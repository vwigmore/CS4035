# coding: utf-8

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from collections import deque
from StringIO import StringIO


import numpy as np

#The method used to read in the smoted training set.
def read_data_smoted(str):

    #The full set is too large to read in (it leads to memory errors)
    #Therefore the first 25000 lines of the smoted set are read in here.
    #These are all non fraudelent cases and all the fraudulent non-smoted cases
    nr_samples = 25000
    dataset = pd.read_csv(str, delimiter=',',nrows=nr_samples, header=None)

    #Here the last 25000 lines of the smoted set are read in these are all smoted fraudulent cases.
    with open(str, 'r') as f:
        q = deque(f, nr_samples)
    dataset_bottom = pd.read_csv(StringIO(''.join(q)), header=None)
    #The two sets are here concatenated.
    dataset = pd.concat([dataset, dataset_bottom])

    #read all the chargebacks and create the test labels
    Targets = []
    for x in dataset.loc[:, 2].values:
       if x == "Chargeback":
          Targets.append(1)
       else:
          Targets.append(0)

    #The labels and other features that should not be used in training are removed.
    dataset = dataset.drop([dataset.columns[2], dataset.columns[285], dataset.columns[286]], axis=1)
    Inputs = dataset.values
    dataset = None
    return Inputs, Targets

#The method used to read in the test set.
def read_data_test(str):

    dataset = pd.read_csv(str, delimiter=',', header=None)
    Targets = []
    for x in dataset.loc[:, 4].values:
        if x == "Chargeback":
            Targets.append(1)
        else:
            Targets.append(0)

    #all unusable features are removed here
    dataset = dataset.drop([dataset.columns[4], dataset.columns[0], dataset.columns[3], dataset.columns[5],
                            dataset.columns[7], dataset.columns[8], dataset.columns[9], dataset.columns[10],
                            dataset.columns[285], dataset.columns[286]], axis=1)
    Inputs = dataset.values
    dataset = None
    return Inputs, Targets

#The method used to calculate the number of True positives, True negatives, False positives and False negatives.
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

#This method creates a dictionary that maps the column numbers to the features stored on those columns.
def dictionary():
    header = {}

    with open("datasets/header.txt", "r") as filestream:
        for line in filestream:
            currentline = line.split(",")
            for i, element in enumerate(currentline):
                header[i] = element
    header[0] = 'bin'
    header[291] = 'SwedenAccount_accountcode'
    return header

#Sub method of visualisation that is called to visualise the trees used in the random forest
def visualize_tree(Tree, index):
    n_nodes = Tree.tree_.node_count
    children_left = Tree.tree_.children_left
    children_right = Tree.tree_.children_right
    feature = Tree.tree_.feature
    threshold = Tree.tree_.threshold

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True


    print("This is tree number: %s in the forest, the tree has %s nodes and has "
          "the following tree structure:"
          % (index, n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))
    print

#Sub method of visualisation that is used to visualise the decision path the transaction makes through the different
#in the random forest.
def visualize_decision_path(Tree, transaction, simplified, Tree_n):

    feature = Tree.tree_.feature
    threshold = Tree.tree_.threshold

    node_indicator = Tree.decision_path(transaction)

    leave_id = Tree.apply(transaction)

    sample_id = 0
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]
    featureset = []
    thresholdsignset = []
    thresholdset = []
    valueset = []
    nodeset = []

    #This prints out the decision path using columns and thresholds of each node in the tree.
    #only called if the visualisation is not simplified.
    if not simplified:

       print('Rules used to predict sample %s in tree number %s of the random forest: ' % (sample_id, Tree_n))
       for node_id in node_index:
           if leave_id[sample_id] == node_id:
               print('Final leave node id: %s' % node_id)
               continue

           if (transaction[sample_id, feature[node_id]] <= threshold[node_id]):
               threshold_sign = "<="
           else:
               threshold_sign = ">"

           print("decision id node %s : (transaction[:, %s] (= %s) %s %s)"
                 % (node_id,
                    #sample_id,
                    feature[node_id],
                    transaction[sample_id, feature[node_id]],
                    threshold_sign,
                    threshold[node_id]))
       print

    #code used to visualise the decision path each column is matched to the feature on the column.
    #To show what features where used in what order by the tree to determine if the transaction is fraudulent.
    for node_id in node_index:
        if leave_id[sample_id] == node_id:
            continue

        if (transaction[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"
        featureset.append(feature[node_id])
        thresholdsignset.append(threshold_sign)
        valueset.append(transaction[sample_id, feature[node_id]])
        thresholdset.append(threshold[node_id])
        nodeset.append([node_id])

    header = dictionary()

    print('decision path of the transaction in tree %s of the random forest' %Tree_n)
    for i, e in enumerate(featureset):
        string = 'node %s of the tree made the decision because ' %nodeset[i]
        if e == 0 or e == 1 or e == 2:
           if thresholdsignset[i] == '<=':
               string += '%s of the transaction was %s which is less than the threshold of %s' %(header[e], valueset[i], thresholdset[i])
           else:
               string += '%s of the transaction was %s which is more than the threshold of %s' %(header[e], valueset[i], thresholdset[i])
           print(string)
           continue
        if thresholdsignset[i] == '<=':
            string += 'transaction did not match the condition of %s' %header[e]
        else:
            string += 'transaction did match the condition of %s' %header[e]
        print(string)
    print

#The main method used to visualise how a transaction is labeled as fraudulent by the random forest.
#Simplified is a boolean if it is true only the decision path of the transaction through the trees is revealed.
#If Simplified is false each tree structure is visualised as well as the decision path of the transaction through the tree.
def visualize_decision(rf, transaction, actual_result, simplified):
    trees = rf.estimators_
    The_trees = []

    if actual_result == 1:
        print("the transaction was actually fraudulent")
    else:
        print("the transaction was not fraudulent")

    #only trees that predict that transaction to be fraudulent are visualised.
    for j, tree in enumerate(trees):
        y_predict = tree.predict(transaction)
        if y_predict[0] == 1:
            The_trees.append([tree, j])

    print("the amount of trees that decided the transaction was fraudulent is %s out of %s" %(len(The_trees), len(trees)))
    print('')

    #calls the submethods that print out the tree structure as well as how
    for x in The_trees:
        if not simplified:
            visualize_tree(x[0], x[1])
        visualize_decision_path(x[0], transaction, simplified, x[1])

if __name__ == "__main__":

    # arrays to store values in for each of the 10 iterations.
    y_prob_tot = []
    y_pred_tot = []
    y_actual_tot = []
    #Decision Threshold used.
    Thr = 0.7


    for i in range(0, 10):
        #read in the data set.
        test_x, test_y = read_data_test('datasets/test' + str(i) + '.csv')
        train_x, train_y = read_data_smoted('datasets/tsmote' + str(i) + '.csv')
        print("files loaded in iteration number: " + str(i))

        # used classifier is the random forest algorithm
        clf = RandomForestClassifier(n_estimators=20, max_depth=5)
        clf.fit(train_x, train_y)
        y_predict = clf.predict(test_x)

        y_pred_prob = clf.predict_proba(test_x)[:, 1]

        y_pred_tot.extend(y_predict)
        y_actual_tot.extend(test_y)
        y_prob_tot.extend(y_pred_prob)

        #takes one case that is determined to be fraudulent and visualises the random forests decision making.
        if i == 1:
            for x, y in enumerate(y_predict):
                if y == 1:
                    visualize_decision(clf, np.array([test_x[i]]), test_y[i], True)
                    break

    #takes the Threshold to determine if a case is fraudluent.
    y_pred_tot2 = list(map(lambda x: 1 if x > Thr else 0, y_prob_tot))
    calc_eff(y_pred_tot2, y_actual_tot)

    #Roc curve of the data used.
    fpr, tpr, thresholds = metrics.roc_curve(y_actual_tot, y_prob_tot)

    plt.figure(3)
    plt.plot(fpr, tpr, 'b', label="10Fold Random Forests")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()

    plt.show()


