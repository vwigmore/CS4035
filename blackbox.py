import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import deque
from StringIO import StringIO
from sklearn import metrics
from sklearn import neighbors


def calc_eff(y_predict, y_test):
    """
    This function calculates the performance of a predictor
    using the predicted values and the target values
    """
    TP, FP, FN, TN = 0, 0, 0, 0
    for i in xrange(len(y_predict)):
        if y_test[i] == 1 and y_predict[i] == 1:
            TP += 1
        if y_test[i] == 0 and y_predict[i] == 1:
            FP += 1
        if y_test[i] == 1 and y_predict[i] == 0:
            FN += 1
        if y_test[i] == 0 and y_predict[i] == 0:
            TN += 1
    print('TP: ' + str(TP))
    print('FP: ' + str(FP))
    print('FN: ' + str(FN))
    print('TN: ' + str(TN))


def read_data_smoted(path):
    # reads the smoted data which is the training data. Because of memory issues, only 200000 entries are used
    names = [i for i in range(0, 295)]
    types = {i: np.int8 for i in range(3, 295)}
    types.update({'0': np.float32, '1': np.float32, '2': np.unicode_})
    nr_samples = 100000

    # read the first 100.000 entries which are the non-fraud cases
    dataset = pd.read_csv(path, delimiter=',', nrows=nr_samples, names=names, dtype=types,
                          header=None)
    # read the last 100.000 entries which are the fraud cases (SMOTE)
    with open(path, 'r') as f:
        q = deque(f, nr_samples)
    dataset_bottom = pd.read_csv(StringIO(''.join(q)), header=None)
    dataset = pd.concat([dataset, dataset_bottom])

    # set label from training data as target with chargeback=1 and settled=0
    Targets = []
    for x in dataset.loc[:, 2].values:
        if x == "Chargeback":
            Targets.append(1)
        else:
            Targets.append(0)

    # remove label from training data and one hot encoded label
    dataset = dataset.drop(dataset.columns[2], axis=1)
    dataset = dataset.drop(dataset.columns[286], axis=1)
    dataset = dataset.drop(dataset.columns[285], axis=1)

    Inputs = dataset.values
    dataset = None
    return Inputs, Targets


def read_data_test(path):
    dataset = pd.read_csv(path, delimiter=',', header=None)

    # set label from test data as target with chargeback=1 and settled=0
    Targets = []
    for x in dataset.loc[:, 4].values:
        if x == "Chargeback":
            Targets.append(1)
        else:
            Targets.append(0)
    # forgot to remove unnecessary columns from dataset in KNIME
    dataset = dataset.drop([dataset.columns[4], dataset.columns[0], dataset.columns[3], dataset.columns[5], dataset.columns[7], dataset.columns[8], dataset.columns[9], dataset.columns[10]], axis=1)
    dataset = dataset.drop(dataset.columns[286], axis=1)
    dataset = dataset.drop(dataset.columns[285], axis=1)
    Inputs = dataset.values
    dataset = None
    return Inputs, Targets


if __name__ == "__main__":

    # arrays to store values in for each of the 10 iterations.
    y_prob_tot = []
    y_targ_tot = []
    threshold = 0.8  # threshold for probability above which an entry is marked as fraud

    for i in range(0, 10):
        Inp_test, Targ_test = read_data_test('datasets/test' + str(i) + '.csv')
        Inp_train, Targ_train = read_data_smoted('datasets/tsmote' + str(i) + '.csv')
        print("files loaded in iteration number: " + str(i))

        # used classifier is the K-neighbors algorithm
        clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
        clf.fit(Inp_train, Targ_train)

        # take the probabilities of an entry being fraud
        y_pred_prob = clf.predict_proba(Inp_test)[:, 1]

        # add probabilities and targets to array
        y_prob_tot.extend(y_pred_prob)
        y_targ_tot.extend(Targ_test)

    # calculate values for roc curve
    fpr, tpr, t = metrics.roc_curve(Targ_test, y_pred_prob)

    # classify entries based on probabilities and chosen threshold
    y_pred_tot = list(map(lambda x: 1 if x > threshold else 0, y_prob_tot))
    calc_eff(y_pred_tot, y_targ_tot)

    plt.figure(1)
    plt.plot(fpr, tpr, 'r', label="K-nearest neighbor")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.show()
