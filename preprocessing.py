# coding: utf-8

import numpy as np
import csv


if __name__ == "__main__":

    chargeback = []
    settled = []

    ah = open('datasets/enc.csv', 'r')
    ah.readline()
    for line_ah in ah:
        line_ahs = line_ah.replace('"', '')
        if line_ahs.strip().split(',')[4] == 'Chargeback':
            chargeback.append(line_ahs.strip().split(','))
        elif line_ahs.strip().split(',')[4] == 'Settled':
            settled.append(line_ahs.strip().split(','))

    ch = [0 for i in range(0, 10)]
    for i in range(0, len(chargeback)):
        index = i % 10
        if ch[index] == 0:
            ch[index] = [chargeback[i]]
        else:
            ch[index].append(chargeback[i])
    chargeback = None
    st = [0 for i in range(0, 10)]
    for i in range(0, len(settled)):
        index = i % 10
        if st[index] == 0:
            st[index] = [settled[i]]
        else:
            st[index].append(settled[i])
    settled = None
    for i in range(0, 10):
        test = ch[i] + st[i]
        test = np.array(test)

        with open("datasets/test" + str(i) + ".csv", "w+") as csv_file:
            csvW = csv.writer(csv_file, delimiter=",")
            csvW.writerows(test)

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

        with open("datasets/train" + str(i) + ".csv", "w+") as csv_file:
            csvW = csv.writer(csv_file, delimiter=",")
            csvW.writerows(train)

