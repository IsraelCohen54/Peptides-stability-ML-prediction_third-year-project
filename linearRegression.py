import math
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

AMINO_ACIDS = ('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')


# read the excel and return lists of sequences and stability labels
def makeInputData():
    # sort seqs according to labels
    def sort(seqs, labels):
        labels, seqs = map(list, zip(*sorted(zip(labels, seqs), reverse=False)))
        return seqs, labels

    # split string to array of its chars
    def split_to_chars(string):
        return [char for char in string]

    # shuffle two given list in the same way
    def myShuffle(a, b):
        # make list of shuffled indices
        inds = np.arange(len(a))
        np.random.shuffle(inds)
        # make new arrays and give them shuffled values
        newA, newB = [], []
        for ind in inds:
            newA.append(a[ind])
            newB.append(b[ind])
        return newA, newB

    exl_data = pd.read_excel("pepsData.xlsx")
    # extract important columns
    seqs_column = exl_data.pop("sequence")
    stability_column = exl_data.pop("stability")
    # transform to lists
    seqs_asterisk = seqs_column.tolist()
    stability = stability_column.tolist()
    # remove asterisk and sort
    seqs = [with_asterisk[:-1] for with_asterisk in seqs_asterisk]
    seqs_sorted, stability_sorted = sort(seqs, stability)

    # split each sequence to array off amino acids
    seqs_splitted = [split_to_chars(seqs_sorted[i]) for i in range(len(seqs_sorted))]

    # shuffle and return np array
    seqs_shuffled, stabilitys_shuffled = myShuffle(seqs_splitted, stability)
    return np.array(seqs_shuffled), np.array(stabilitys_shuffled)


# dense-encode seqs, and transform both to tensors
def myEncode(seqs, stabs):
    # encode given seq to list such as element i is how many times AMINO_ACIDS[i] occurs in the given seq
    def myEncodeDenseOne(seq):
        return [(seq == AA).sum() for AA in AMINO_ACIDS]

    seqs_encoded = torch.FloatTensor([myEncodeDenseOne(seq) for seq in seqs])
    stabs_encoded = torch.FloatTensor(stabs)
    return seqs_encoded, stabs_encoded


if __name__ == "__main__":
    # read and encode data
    seqs, stabs = makeInputData()
    seqs_encoded, stabs_encoded = myEncode(seqs, stabs)
    stabs_encoded = stabs_encoded.reshape(-1, 1)

    # model
    model = nn.Linear(20, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # train
    epochs = 300
    for epoch in range(epochs):
        pred = model(seqs_encoded)
        loss = criterion(pred, stabs_encoded)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 10 == 0:
            print(f"epoch {epoch}     loss {loss.item():.4f}")

    # make prediction
    preds = model(seqs_encoded).detach().numpy()
    preds = np.squeeze(preds)

    # plot
    originalStability = 1
    plt.figure()
    plt.scatter(stabs, preds)
    plt.xlabel("real stability")
    plt.ylabel("predictional stability")
    plt.title("linear regression")
    z = np.polyfit(stabs, preds, 1)
    p = np.poly1d(z)
    plt.plot(stabs, p(stabs), "r--")
    # R = r2_score(preds, p(stabs))
    # plt.text(4.5, 5.5, 'R^2=%0.3f' % R, fontdict={'fontsize': 17})
    plt.savefig("linear regression")
    plt.show()
