import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
import torch
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from sklearn.metrics import mean_squared_error

AMINO_ACIDS = ('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')


# read the excel and return lists of sequences and stability labels
def makeInputData():
    # split string to array of its chars
    def split_to_chars(string):
        return [char for char in string]

    # shuffle two given lists in the same order
    def shuffle(seqs, stabs):
        both = list(zip(seqs, stabs))
        random.shuffle(both)
        seqs, stabs = zip(*both)
        return seqs, stabs

    exl_data = pd.read_excel("pepsData.xlsx")
    # extract important columns
    seqs_column = exl_data.pop("sequence")
    stability_column = exl_data.pop("stability")
    # transform to lists
    seqs_asterisk = seqs_column.tolist()
    stability = stability_column.tolist()
    # remove asterisk and split
    seqs = [with_asterisk[:-1] for with_asterisk in seqs_asterisk]
    seqs_splitted = [split_to_chars(seqs[i]) for i in range(len(seqs))]
    # shuffle
    seqs_splitted, stability = shuffle(seqs_splitted, stability)
    return np.array(seqs_splitted), np.array(stability)


# densely encode given data
def myEncodeDense(train_x, test_x, train_y, test_y):
    # encode given seq to list such as element i is how many times AMINO_ACIDS[i] occures in the given seq
    def myEncodeDenseOne(seq):
        return [(seq == AA).sum() for AA in AMINO_ACIDS]

    train_x_encoded = torch.FloatTensor([myEncodeDenseOne(seq) for seq in train_x])
    test_x_encoded = torch.FloatTensor([myEncodeDenseOne(seq) for seq in test_x])
    train_y_encoded = torch.tensor(train_y)
    test_y_encoded = torch.tensor(test_y)
    return train_x_encoded, test_x_encoded, train_y_encoded, test_y_encoded


# encode. than turn to np array, float, and tensor
def myEncode(train_x, test_x, train_y, test_y):
    # train_x.reshape(-1,1); test_x.reshape(-1,1); validation_x.reshape(-1,1); train_y.reshape(-1,1); test_y.reshape(-1,1); validation_y.reshape(-1,1);
    # make encoder
    enc = OneHotEncoder(sparse=False)
    enc.fit(train_x)
    # samples = transform and convert to float tensor
    encoded_train_x = torch.FloatTensor(enc.transform(train_x))
    encoded_test_x = torch.FloatTensor(enc.transform(test_x))
    # labels - just convert to tensor
    encoded_train_y = torch.tensor(train_y)
    encoded_test_y = torch.tensor(test_y)
    return encoded_train_x, encoded_test_x, encoded_train_y, encoded_test_y


# plot scatter plot of original stability and score1 of machine output, with trend line. save it in given pdf
def plotScatter(stability, preds):
    plt.figure()
    # scatter plot
    plt.scatter(stability, preds)
    plt.xlabel("real stability")
    plt.ylabel("predicted stability")
    plt.title("mlp regression")
    # trend line
    z = np.polyfit(stability, preds, 1)
    p = np.poly1d(z)
    plt.plot(stability, p(stability), "r--")
    plt.savefig("mlp regression")


if __name__ == "__main__":
    seqs, stabs = makeInputData()

    criterion = nn.MSELoss()

    cv = KFold(n_splits=10, random_state=777, shuffle=True)
    preds, correspondingStabs = [], []
    curFold = -1
    for train_index, test_index in cv.split(seqs):
        curFold += 1
        print(f"{curFold=}")
        # define train and test data, encode it
        seqs_train, seqs_test, stabs_train, stabs_test = seqs[train_index], seqs[test_index], stabs[train_index], stabs[test_index]
        seqs_train, seqs_test, stabs_train, stabs_test = myEncodeDense(seqs_train, seqs_test, stabs_train, stabs_test)
        # initiate model, make prediction, save results
        model = MLPRegressor(random_state=1, max_iter=500).fit(seqs_train, stabs_train)
        testPreds = model.predict(seqs_test)
        preds.extend(testPreds)
        correspondingStabs.extend(stabs_test)


    loss = mean_squared_error(correspondingStabs, preds)
    print("loss: ", loss)
    plotScatter(correspondingStabs, preds)