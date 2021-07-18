import numpy as np
import torch
from sklearn import metrics
from torch import nn
from prots_utils import makeInputData
#from mainMLP import makeInputData

from sklearn.model_selection import KFold

AMINO_ACIDS = ('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')

# hyper parameters
epochs = 300
lr = 0.02
folds = 10


class LRModel(nn.Module):

    def __init__(self):
        super(LRModel, self).__init__()
        self.fc = nn.Linear(21, 1)

    def forward(self, x):
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


def myEncodeDense(seqs, labels):
    def myEncodeDenseOne(seq):
        return [(seq == AA).sum() for AA in AMINO_ACIDS]

    seqs_encoded = torch.FloatTensor([myEncodeDenseOne(seq) for seq in seqs])
    labels_encoded = torch.FloatTensor(labels)
    return seqs_encoded, labels_encoded

def train():
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = modely(seqs_train)
        loss = loss_f(preds, labels_train)
        loss.backward()
        optimizer.step()


def test():
    with torch.no_grad():
        # prediction
        preds = modely(seqs_test)
        predsLabels = preds.round()
        # accuracy
        curAccuracy = predsLabels.eq(labels_test).sum() / len(labels_test)
        accuracy.append(curAccuracy)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(labels_test, preds)
        curAuc = metrics.auc(fpr, tpr)
        auc.append(curAuc)


if __name__ == "__main__":
    # read and encode all data
    # seqs, labels, stability = makeInputData()
    # seqs, labels = myEncodeDense(seqs, labels)
    seqs, labels = makeInputData()
    labels = labels.reshape(-1,1)
    loss_f = nn.BCELoss()

    # do k-fold
    auc, accuracy = [], []
    cv = KFold(n_splits=folds, random_state=777, shuffle=True)
    for train_index, test_index in cv.split(seqs):
        # init current data and model
        seqs_train, seqs_test, labels_train, labels_test,  = seqs[train_index], seqs[test_index], labels[train_index], labels[test_index]
        modely = LRModel()
        optimizer = torch.optim.Adam(modely.parameters(), lr=lr)
        train()
        test()
    accuracy, auc = np.mean(accuracy), np.mean(auc)
    print("accuracy:  ", accuracy)
    print("auc:  ", auc)

    print('\n')
    for aa, weight in zip(AMINO_ACIDS, modely.fc.weight.data[0]):
        print(aa, '\t', weight.item())
    print("tail\t", modely.fc.weight.data[0][-1].item())