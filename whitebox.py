# coding: utf-8

import datetime
import time
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTENC
import pandas as pd
from collections import deque
from StringIO import StringIO

import numpy as np

def read_data_smote(str):
    #names = [i for i in range(0, 295)]
    #types = {i: np.int8 for i in range(3, 295)}
    #types.update({'0': np.float32, '1': np.float32, '2': np.unicode_})

    nr_samples = 25000
    dataset = pd.read_csv(str, delimiter=',',nrows=nr_samples, header=None)

    with open(str, 'r') as f:
        q = deque(f, nr_samples)
    dataset_bottom = pd.read_csv(StringIO(''.join(q)), header=None)
    dataset = pd.concat([dataset, dataset_bottom])

    Targets = []
    for x in dataset.loc[:, 2].values:
       if x == "Chargeback":
          Targets.append(1)
       else:
          Targets.append(0)

    dataset = dataset.drop(dataset.columns[2], axis=1)
    Inputs = dataset.values
    dataset = None
    return Inputs, Targets

def read_data_test(str):
    dataset = pd.read_csv(str, delimiter=',', header=None)
    Targets = []
    for x in dataset.loc[:, 4].values:
        if x == "Chargeback":
            Targets.append(1)
        else:
            Targets.append(0)

    dataset = dataset.drop([dataset.columns[4], dataset.columns[0], dataset.columns[3], dataset.columns[5], dataset.columns[7], dataset.columns[8], dataset.columns[9], dataset.columns[10]], axis=1)
    Inputs = dataset.values
    print(Inputs[0])
    dataset = None
    return Inputs, Targets


#def RForest(x, y, s):
    #x_array = np.array(x)
    #y_array = np.array(y)
    #usx = x_array
    #usy = y_array
    #x_train, x_test, y_train, y_test = train_test_split(usx, usy, test_size=s)
    #clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    #clf.fit(x_train, y_train)
    #y_predict = clf.predict(x_test)
    #y_pred_prob = clf.predict_proba(x_test)[:, 1]
    #return y_predict, y_test, y_pred_prob


def calc_eff(y_predict, y_test):
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

def visualize_decision(rf, sample_id, X_train):
    for j, tree in enumerate(rf.estimators_):

        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold

        print("Decision path for DecisionTree {0}".format(j))
        node_indicator = tree.decision_path(X_train)
        leave_id = tree.apply(X_train)
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]

        print('Rules used to predict sample %s: ' % sample_id)
        for node_id in node_index:
            if leave_id[sample_id] != node_id:
                continue

            if (X_train[sample_id, feature[node_id]] <= threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            print("decision id node %s : (X_train[%s, %s] (= %s) %s %s)"
                  % (node_id,
                     sample_id,
                     feature[node_id],
                     X_train[sample_id, feature[node_id]],
                     threshold_sign,
                     threshold[node_id]))

if __name__ == "__main__":
    train_x, train_y = read_data_smote('datasets/tsmote0.csv')
    test_x, test_y = read_data_test('datasets/test0.csv')
    print(len(test_y))

    #x_train = np.array(train_x)
    #y_train = np.array(train_y)

    #x_test = np.array(test_x)
    #y_test = np.array(test_y)

    clf = RandomForestClassifier(n_estimators=100,max_depth=5)
    clf.fit(train_x, train_y)
    y_predict = clf.predict(test_x)
    calc_eff(y_predict, test_y)

    y_pred_prob = clf.predict_proba(test_x)[:, 1]

    print(y_pred_prob)
    fpr, tpr, thresholds = metrics.roc_curve(test_y, y_pred_prob)
    print(thresholds)

    plt.figure(3)
    plt.plot(fpr, tpr, 'b', label="SMOTED Random Forests")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()

    plt.show()

    #decisionpath, n_node_ptr = clf.decision_path(test_x)
    #visualize_decision(clf, 0, test_x)

    #stratisfied kfold means that each fold has roughly the same number
    #of fraudulent cases in the test fold.
    #kf = StratifiedKFold(n_splits=10)

    #for train, test in kf.split(x_array, y_array):
       #smote the data set
       #sm = SMOTENC(categorical_features=[0, 1, 4, 5, 6, 7, 10, 11, 12, 13])

       #I assume it works but it takes forever
       #resampledX, resampledY = sm.fit_resample(x_array[train], y_array[train])


       #clf = RandomForestClassifier(n_estimators=100, max_depth=2)
       #clf.fit(x_array[train], y_array[train])
       #y_predict = clf.predict(x_array[test])
       #calc_eff(y_predict, y_array[test])
       #y_pred_prob = clf.predict_proba(x_array[test])[:, 1]

       #decisionpath: Return a node indicator matrix where non zero elements indicates
       # that the samples goes through the nodes.

       #n_node_ptr: The columns from
       # indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]] gives the indicator value for the i-th estimator.
       #decisionpath, n_node_ptr = clf.decision_path(x_array[test])
       #visualize_decision(clf, 0, x_array[test])





