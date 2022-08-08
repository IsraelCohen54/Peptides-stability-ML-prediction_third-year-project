import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import math
import torch
from torch import nn, optim
import torch.nn.functional as F
from xlwt import Workbook
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMINO_ACIDS = ('A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')

# hyper parameters
folds = 6
epochs = 5
lr = 0.0001

# parameters to be written at the results file
params = f"parameters:     folds-{folds}  epochs-{epochs}  lr-{lr}"

# read the excel and return lists of sequences and stability labels
def makeInputData():
    # sort seqs according to labels
    def sort(seqs, labels):
        labels, seqs = map(list, zip(*sorted(zip(labels, seqs), reverse=False)))
        return seqs, labels

    # get highest and lowest sequences, according to labels, and return the labels as well
    def get_highest_and_lowest_quarters(seqs, stab):
        quarter = math.floor(len(seqs) / 4)
        firstQuarterSeqs = seqs[:quarter]
        lastQuarterSeqs = seqs[len(seqs) - quarter:]
        firstQuarterStab = stab[:quarter]
        lastQuarterStab = stab[len(stab) - quarter:]
        return firstQuarterSeqs, lastQuarterSeqs, firstQuarterStab, lastQuarterStab

    # split string to array of its chars
    def split_to_chars(string):
        return [char for char in string]

    # shuffle three arrays the same way
    def shuffleThree(X, Y, Z):
        # give s indecies of X (and Y, Z)
        s = np.arange(0, len(X), 1)
        # shuffle indecies
        np.random.shuffle(s)
        # make new arrays and give them shuffled values
        newX = []
        newY = []
        newZ = []
        for i in range(len(X)):
            newX.append(X[s[i]])
            newY.append(Y[s[i]])
            newZ.append(Z[s[i]])
        return newX, newY, newZ

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
    # get highest and lowest quarters by seq, and by stability or two mid quarters as said before
    first_quarter_seqs, last_quarter_seqs, first_quarter_stab, last_quarter_stab = get_highest_and_lowest_quarters(
        seqs_sorted, stability_sorted)
    # split each sequence to array off amino acids
    first_quarter_seqs_splitted = [split_to_chars(first_quarter_seqs[i]) for i in range(len(first_quarter_seqs))]
    last_quarter_seqs_splitted = [split_to_chars(last_quarter_seqs[i]) for i in range(len(last_quarter_seqs))]
    # make new labels
    first_quarter_labels = [0 for _ in range(len(first_quarter_seqs))]
    last_quarter_labels = [1 for _ in range(len(last_quarter_seqs))]
    # concatenate quarters
    seqs_margins = first_quarter_seqs_splitted + last_quarter_seqs_splitted
    labels_margins = first_quarter_labels + last_quarter_labels
    stability_margins = first_quarter_stab + last_quarter_stab
    # shuffle and return np array. in the kfold we do shuffle=True, but still without the shuffle here the algorithm doesn't work
    seqs_margins_shuffled, labels_margins_shuffled, stability_margins_shuffled = \
        shuffleThree(seqs_margins, labels_margins, stability_margins)
    return np.array(seqs_margins_shuffled), np.array(labels_margins_shuffled), np.array(stability_margins_shuffled)


def myEncodeDense(train_x, test_x, validation_x, train_y, test_y, validation_y):
    # encode given seq to list such as element i is how many times AMINO_ACIDS[i] occures in the given seq
    def myEncodeDenseOne(seq):
        return [(seq == AA).sum() for AA in AMINO_ACIDS]

    train_x_encoded = torch.FloatTensor([myEncodeDenseOne(seq) for seq in train_x])
    test_x_encoded = torch.FloatTensor([myEncodeDenseOne(seq) for seq in test_x])
    validation_x_encoded = torch.FloatTensor([myEncodeDenseOne(seq) for seq in validation_x])
    train_y_encoded = torch.tensor(train_y)
    test_y_encoded = torch.tensor(test_y)
    validation_y_encoded = torch.tensor(validation_y)
    return train_x_encoded, test_x_encoded, validation_x_encoded, train_y_encoded, test_y_encoded, validation_y_encoded


# encode. than turn to np array, float, and tensor
def myEncode(train_x, test_x, validation_x, train_y, test_y, validation_y):
    # train_x.reshape(-1,1); test_x.reshape(-1,1); validation_x.reshape(-1,1); train_y.reshape(-1,1); test_y.reshape(-1,1); validation_y.reshape(-1,1);
    # make encoder
    enc = OneHotEncoder(sparse=False)
    enc.fit(train_x)
    # samples = transform and convert to float tensor
    encoded_train_x = torch.FloatTensor(enc.transform(train_x))
    encoded_test_x = torch.FloatTensor(enc.transform(test_x))
    encoded_validation_x = torch.FloatTensor(enc.transform(validation_x))
    # labels - just convert to tensor
    encoded_train_y = torch.tensor(train_y)
    encoded_test_y = torch.tensor(test_y)
    encoded_validation_y = torch.tensor(validation_y)
    return encoded_train_x, encoded_test_x, encoded_validation_x, encoded_train_y, encoded_test_y, encoded_validation_y


# define net layers
class ourModel(nn.Module):
    def __init__(self, vec_size):
        super(ourModel, self).__init__()
        self.vec_size = vec_size
        self.fc0 = nn.Linear(vec_size, 395)
        self.fc1 = nn.Linear(395, 330)
        self.fc2 = nn.Linear(330, 265)
        self.fc3 = nn.Linear(265, 200)
        self.fc4 = nn.Linear(200, 135)
        self.fc5 = nn.Linear(135, 70)
        self.fc6 = nn.Linear(70, 2)

    def forward(self, x):
        x = x.view(-1, self.vec_size)
        x = tanh(self.fc0(x))
        x = tanh(self.fc1(x))
        x = tanh(self.fc2(x))
        x = tanh(self.fc3(x))
        x = tanh(self.fc4(x))
        x = tanh(self.fc5(x))
        x = self.fc6(x)
        return F.log_softmax(x, dim=1)


# define net layers
class ourModelDense(nn.Module):
    def __init__(self, vec_size):
        super(ourModelDense, self).__init__()
        self.vec_size = vec_size
        self.fc0 = nn.Linear(vec_size, 8)
        self.fc1 = nn.Linear(8, 2)

    def forward(self, x):
        x = x.view(-1, self.vec_size)
        x = tanh(self.fc0(x))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


# normalize with z-score, than do tanh
def tanh(x):
    normalized = (x - torch.mean(x)) / torch.std(x)
    return torch.tanh(normalized)


# train with given samples and labels
def train(samples, labels, model, optimizer):
    model.train()
    for sample, label in zip(samples, labels):
        optimizer.zero_grad()
        output = model(sample)
        t_loss = F.nll_loss(output, label.long().view(1)) #nll_loss: The negative log likelihood loss. It is useful to train a classification problem with C classes. If provided, the optional argument weight should be a 1D Tensor assigning weight to each of the classes. This is particularly useful when you have an unbalanced training set which is our case, as we dont have a balanced number of stabe/unstable IMO using "same" AAs for both to learn...
        t_loss.backward()
        optimizer.step()


def tensorToList(tensorList):
    normalList = []
    for i in range(len(tensorList)):
        normalList.append(tensorList[i].tolist()[0][0])
    return normalList


# do predictions on train data. exactly like validation. we don't need score value of train
def predOnTrain(samples, labels, model):
    auc, accuracy, f1Score, precision, recall, scores1 = validation(samples, labels, model)
    return auc, accuracy, f1Score, precision, recall


# do validation with given samples and labels, return loss
def validation(samples, labels, model):
    # do prediction
    model.eval()
    predictions = []
    scores1 = []
    with torch.no_grad():
        for v_sample, v_label in zip(samples, labels):
            output = model(v_sample)
            predictions.append(output.max(1, keepdim=True)[1])
            scores1.append(output[0][1])
    # take measurements
    predictions = tensorToList(predictions)  # that way the next functions receive two normal lists, not normal list and tensors list
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
    auc = metrics.auc(fpr, tpr)
    accuracy = metrics.accuracy_score(labels, predictions)
    f1Score = metrics.f1_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, zero_division=0)
    recall = metrics.recall_score(labels, predictions, zero_division=0)
    return auc, accuracy, f1Score, precision, recall, scores1


# make all our measurements simple lists, which are mean of all folds
def measurementsMeans():
    global t_auc, t_accuracy, t_f1Score, t_precision, t_recall, v_auc, v_accuracy, v_f1Score, v_precision, v_recall
    t_auc = np.mean(t_auc, axis=0)
    t_accuracy = np.mean(t_accuracy, axis=0)
    t_f1Score = np.mean(t_f1Score, axis=0)
    t_precision = np.mean(t_precision, axis=0)
    t_recall = np.mean(t_recall, axis=0)
    v_auc = np.mean(v_auc, axis=0)
    v_accuracy = np.mean(v_accuracy, axis=0)
    v_f1Score = np.mean(v_f1Score, axis=0)
    v_precision = np.mean(v_precision, axis=0)
    v_recall = np.mean(v_recall, axis=0)
    global test_auc, test_accuracy, test_f1Score, test_precision, test_recall
    test_auc = np.mean(test_auc)
    test_accuracy = np.mean(test_accuracy)
    test_f1Score = np.mean(test_f1Score)
    test_precision = np.mean(test_precision)
    test_recall = np.mean(test_recall)


# write given measurements with given names list on given excel sheet
def writeResults(sheet, names, *measurements):
    # write epochs in column 0
    sheet.write(0, 0, 'epochs')
    for i in range(epochs):
        sheet.write(i + 1, 0, i + 1)  # i+1 because line 0 is occupied with titles
    # write names
    for i in range(len(names)):
        sheet.write(0, i + 1, names[i])
    # write measurements
    for i, measurement in enumerate(measurements):
        for j in range(epochs):
            sheet.write(j + 1, i + 1, measurement[j])


# write given score and original stability on given excel sheet
def writeScores(sheet):
    # write given data, start in startCol
    startCol = 10
    sheet.write(0, startCol, 'scores1')
    for i, score in enumerate(scores1):
        sheet.write(i + 1, startCol, score.item())  # i+1 because line 0 is occupied with titles
    sheet.write(0, startCol + 1, 'Origin_stability')
    for i, stab in enumerate(stability):
        sheet.write(i + 1, startCol + 1, stab)


# write to excel numeric results of each epoch, and final prediction along with original stability
def writeToExel():
    results = Workbook()
    sheet1 = results.add_sheet("validation")
    # writeResults(sheet1,["Auc","Accuracy","F1Score","Precision","Recall"],v_auc,v_accuracy,v_f1Score,v_precision,v_recall)
    writeScores(sheet1)
    sheet1.write(0, 15, "main MLP")
    sheet1.write(1, 15, params)
    results.save('results.xls')


# plot scatter plot of original stability and score1 of machine output, with trend line. save it in given pdf
def plotScatter(pdf):
    plt.figure()
    score1_reversed = 2 ** (np.array(scores1))  # our score is result of log softmax. reverse it to normal softmax
    plt.scatter(originalStability, score1_reversed)
    plt.xlabel("Stability")
    plt.ylabel("Score1")
    plt.title("Predictions analysis")
    z = np.polyfit(originalStability, score1_reversed, 1)
    p = np.poly1d(z)
    plt.plot(originalStability, p(originalStability), "r--")
    pdf.savefig()


def plotMeasurment(trainData, valData, test_data, name, iters, pdf):
    # lines of train and validation
    plt.figure()
    plt.title(name)
    plt.plot(iters, trainData, label="Train")
    plt.plot(iters, valData, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel(name)
    plt.locator_params(axis="x", integer=True, tight=True)  # make x axis to display only whole number (iterations)
    # add line of test - same value again and again
    ts_line_data = []
    for _ in valData:
        ts_line_data.append(test_data)
    plt.plot(iters, ts_line_data, label="test end value")
    plt.legend()
    pdf.savefig()
    ts_line_data.clear()


# plot graph for each measurement, and scatter plot
def plotGraphsToPDF():
    # create pdf for graphs
    pdf = matplotlib.backends.backend_pdf.PdfPages("graphs.pdf")
    # make lists to iterate over them
    names = ["Auc", "Accuracy", "F1Score", "Precision", "Recall"]
    t_measurements = [t_auc, t_accuracy, t_f1Score, t_precision, t_recall]
    v_measurements = [v_auc, v_accuracy, v_f1Score, v_precision, v_recall]
    test_measurements = [test_auc, test_accuracy, test_f1Score, test_precision, test_recall]
    epochsList = [i for i in range(epochs)]
    # plot graph for each measurement
    for i in range(len(names)):
        plotMeasurment(t_measurements[i], v_measurements[i], test_measurements[i], names[i], epochsList, pdf)
    # plot summarising scatter plot
    plotScatter(pdf)
    # close pdf
    pdf.close()


# receive all data, and separate to 0.8 for train, 0.2 for test
def splitTrainValidation(all_samples, all_labels, all_stability):
    v_indexes = np.random.choice(range(len(all_samples)), math.floor(0.2 * len(all_samples)), replace=False)
    v_samples = all_samples[v_indexes];
    v_labels = all_labels[v_indexes];
    v_stability = all_stability[v_indexes]
    t_samples = all_samples[[i for i in range(len(all_samples)) if i not in v_indexes]]
    t_labels = all_labels[[i for i in range(len(all_samples)) if i not in v_indexes]]
    t_stability = all_stability[[i for i in range(len(all_samples)) if i not in v_indexes]]
    return np.array(t_samples), np.array(v_samples), np.array(t_labels), np.array(v_labels), np.array(t_stability), np.array(v_stability)


# do test, save only scores1 for scatter plot against original stability
def test():
    auc, accuracy, f1Score, precision, recall, test_scores1 = validation(test_x, test_y, modely)
    test_auc.append(auc)
    test_accuracy.append(accuracy)
    test_f1Score.append(f1Score)
    test_precision.append(precision)
    test_recall.append(recall)
    global scores1
    scores1 += test_scores1


# collect measurement of current epoch, update the lists saving them
def collectMeasurements():
    # collect measurement on train
    curAuc, curAccuracy, curF1, curPrecision, curRecall = \
        predOnTrain(train_x, train_y, modely)
    t_auc[curFold].append(curAuc)
    t_accuracy[curFold].append(curAccuracy)
    t_f1Score[curFold].append(curF1)
    t_precision[curFold].append(curPrecision)
    t_recall[curFold].append(curRecall)
    # collect measurement on validation
    curAuc, curAccuracy, curF1, curPrecision, curRecall, curScores1 = \
        validation(validation_x, validation_y, modely)
    v_auc[curFold].append(curAuc)
    v_accuracy[curFold].append(curAccuracy)
    v_f1Score[curFold].append(curF1)
    v_precision[curFold].append(curPrecision)
    v_recall[curFold].append(curRecall)

if __name__ == "__main__":
    samples, labels, stability = makeInputData() # read the excel and return (after shuffling) np.array of AAs sequences (seqs_margins_shuffled), and stability labels (0 non stable quarter, 1 stable quarter), original stability grade.
    # lists to hold train results. Next line explanation: auc[i][j] is the auc on fold i at epoch j. 
    t_auc = [[] for i in range(folds)]      #AUC(Area under the ROC Curve): AUC provides an aggregate measure of performance across all possible classification thresholds}
    t_accuracy = [[] for i in range(folds)] #accuracy: correct predictions
    t_f1Score = [[] for i in range(folds)]  #f1score: the harmonic mean between precision and recall. It is used as a statistical measure to rate performance
    t_precision = [[] for i in range(folds)]#precision: number of true positives (said thing correctly) divided by the total number of positive predictions (say things is X overall without checking if it's correct or not)
    t_recall = [[] for i in range(folds)]   #recall: the ratio between the numbers of Positive samples correctly classified as Positive, to the total number of Positive samples. The recall measures the model's ability to detect positive samples. The higher the recall, the more positive samples detected.
    # list to hold validation results
    v_auc = [[] for i in range(folds)]  # auc[i][j] is the auc on fold i at epoch j
    v_accuracy = [[] for i in range(folds)]
    v_f1Score = [[] for i in range(folds)]
    v_precision = [[] for i in range(folds)]
    v_recall = [[] for i in range(folds)]
    # list to hold final prediction of test
    test_auc = []
    test_accuracy = []
    test_f1Score = []
    test_precision = []
    test_recall = []
    scores1 = []
    originalStability = []

    # run kfold cross validation (meaning, if the data is small, we can part it to E.G. 3 folds, to train 3 times over 2 other part and test the third one, then sum overall results. 
    cv = KFold(n_splits=folds, random_state=777, shuffle=True)
    curFold = -1
    for train_index, test_index in cv.split(samples):
        curFold += 1
        print(f"fold {curFold}")  # just to see it running
        # separate all data to train and test according to k-fold. separate train data to 0.8 train and 0.2 validation. encode all according to train
        train_x, test_x, train_y, test_y, train_stab, test_stab = \
            samples[train_index], samples[test_index], labels[train_index], labels[test_index], stability[train_index], stability[test_index]
        train_x, validation_x, train_y, validation_y, train_stab, validation_stab = splitTrainValidation(train_x, train_y, train_stab)
        # train_x, test_x, validation_x, train_y, test_y, validation_y = myEncode(train_x, test_x, validation_x, train_y, test_y, validation_y)
        train_x, test_x, validation_x, train_y, test_y, validation_y = myEncodeDense(train_x, test_x, validation_x, train_y, test_y, validation_y)

        # prepare new model
        # modely = ourModel(vec_size=23 * 20)
        modely = ourModelDense(vec_size=20)
        optimizery = optim.Adam(modely.parameters(), lr=lr) #Adam optimizer results are generally better, faster and require fewer parameters for tuning, than every other optimization algorithms. optimizer - learning way of the nn weights.
        # do epochs - train and collect train and validation measurements
        for epoch in range(epochs):
            collectMeasurements() #applying data to current nn learned weights and saving results to the lists before...
            train(train_x, train_y, modely, optimizery)
        # at the end of fold, do test. also save original stability of tested peptides for comparison
        test()
        originalStability += test_stab.tolist()

    # make from each measurements simple list, mean of all folds
    measurementsMeans()
    # write numeric results to excel, and plot graphs to pdf
    # writeToExel()
    plotGraphsToPDF()
