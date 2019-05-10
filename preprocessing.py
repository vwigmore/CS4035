# coding: utf-8

import numpy as np
import csv

"""
In this file the one hot encoded data is split for k-fold cross validation with k=10
"""
if __name__ == "__main__":

    chargeback = []
    settled = []

    # separate the fraud from the non-fraud cases
    ah = open('datasets/enc.csv', 'r')
    for line_ah in ah:
        line_ahs = line_ah.replace('"', '')
        if line_ahs.strip().split(',')[4] == 'Chargeback':
            chargeback.append(line_ahs.strip().split(','))
        elif line_ahs.strip().split(',')[4] == 'Settled':
            settled.append(line_ahs.strip().split(','))

    # all chargeback entries are divided over 10 arrays
    ch = [0 for i in range(0, 10)]
    for i in range(0, len(chargeback)):
        index = i % 10
        if ch[index] == 0:
            ch[index] = [chargeback[i]]
        else:
            ch[index].append(chargeback[i])
    chargeback = None

    # all settled entries are divided over 10 arrays
    st = [0 for i in range(0, 10)]
    for i in range(0, len(settled)):
        index = i % 10
        if st[index] == 0:
            st[index] = [settled[i]]
        else:
            st[index].append(settled[i])
    settled = None

    # use the 10 arrays constructed to create 10 different sets for training and testing
    for i in range(0, 10):
        test = ch[i] + st[i]
        test = np.array(test)

        # write validation set to file
        with open("datasets/test" + str(i) + ".csv", "w+") as csv_file:
            csvW = csv.writer(csv_file, delimiter=",")
            csvW.writerows(test)

        # training data consists of all chargebacks and settled entries not in current test set
        train = []
        for x in ch[:i]:
            for y in x:
                train.append(y)
        for x in ch[i:]:
            for y in x:
                train.append(y)
        for x in st[i:]:
            for y in x:
                train.append(y)
        for x in st[:i]:
            for y in x:
                train.append(y)

        train = np.array(train)

        # write training set to file
        with open("datasets/train" + str(i) + ".csv", "w+") as csv_file:
            csvW = csv.writer(csv_file, delimiter=",")
            csvW.writerows(train)

