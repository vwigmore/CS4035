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

import numpy as np


def string_to_timestamp(date_string):#convert time string to float value
    time_stamp = time.strptime(date_string.replace('"', ''), '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)


def read_data(path, s=0):
    ah = open(path, 'r')

    data = []
    datah = []
    color = []
    (issuercountry_set, txvariantcode_set, currencycode_set, shoppercountry_set, interaction_set,
     verification_set, accountcode_set, mail_id_set, ip_id_set, card_id_set) = [set() for _ in xrange(10)]
    (issuercountry_dict, txvariantcode_dict, currencycode_dict, shoppercountry_dict, interaction_dict,
     verification_dict, accountcode_dict, mail_id_dict, ip_id_dict, card_id_dict) = [{} for _ in xrange(10)]
    # label_set
    # cvcresponse_set = set()
    ah.readline()  # skip first line
    for line_ah in ah:
        line_ahs = line_ah.replace('"', '')
        if line_ahs.strip().split(',')[9-s] == 'Refused':  # remove the row with 'refused' label, since it's uncertain about fraud
            continue
        if 'na' in str(line_ahs.strip().split(',')[14-s]).lower() or 'na' in str(line_ahs.strip().split(',')[4-s].lower()):
            continue
        bookingdate = string_to_timestamp(line_ahs.strip().split(',')[1-s])  # date reported flaud
        issuercountry = line_ahs.strip().split(',')[2-s]  # country code
        issuercountry_set.add(issuercountry)
        txvariantcode = line_ahs.strip().split(',')[3-s]  # type of card: visa/master
        txvariantcode_set.add(txvariantcode)
        issuer_id = float(line_ahs.strip().split(',')[4-s])  # bin card issuer identifier
        amount = float(line_ahs.strip().split(',')[5-s])  # transaction amount in minor units
        currencycode = line_ahs.strip().split(',')[6-s]
        currencycode_set.add(currencycode)
        shoppercountry = line_ahs.strip().split(',')[7-s]  # country code
        shoppercountry_set.add(shoppercountry)
        interaction = line_ahs.strip().split(',')[8-s]  # online transaction or subscription
        interaction_set.add(interaction)
        if line_ahs.strip().split(',')[9-s] == 'Chargeback':
            label = 1  # label fraud
        else:
            label = 0  # label save
        verification = line_ahs.strip().split(',')[10-s]  # shopper provide CVC code or not
        verification_set.add(verification)
        cvcresponse = line_ahs.strip().split(',')[11-s]  # 0 = Unknown, 1=Match, 2=No Match, 3-6=Not checked
        if cvcresponse > 2:
            cvcresponse = 3
        year_info = datetime.datetime.strptime(line_ahs.strip().split(',')[12-s], '%Y-%m-%d %H:%M:%S').year
        month_info = datetime.datetime.strptime(line_ahs.strip().split(',')[12-s], '%Y-%m-%d %H:%M:%S').month
        day_info = datetime.datetime.strptime(line_ahs.strip().split(',')[12-s], '%Y-%m-%d %H:%M:%S').day
        creationdate = str(year_info) + '-' + str(month_info) + '-' + str(day_info)  # Date of transaction
        creationdate_stamp = string_to_timestamp(line_ahs.strip().split(',')[12-s])  # Date of transaction-time stamp
        accountcode = line_ahs.strip().split(',')[13-s]  # merchant’s webshop
        accountcode_set.add(accountcode)
        mail_id = int(float(line_ahs.strip().split(',')[14-s].replace('email', '')))  # mail
        mail_id_set.add(mail_id)
        ip_id = int(float(line_ahs.strip().split(',')[15-s].replace('ip', '')))  # ip
        ip_id_set.add(ip_id)
        card_id = int(float(line_ahs.strip().split(',')[16-s].replace('card', '')))  # card
        card_id_set.add(card_id)
        data.append([issuercountry, txvariantcode, issuer_id, amount, currencycode,
                     shoppercountry, interaction, verification, cvcresponse, creationdate_stamp,
                     accountcode, mail_id, ip_id, card_id, label, creationdate])  # add the interested features here
        # y.append(label)# add the labels
    data = sorted(data, key=lambda k: k[-1])

    enc1 = OneHotEncoder(categories=issuercountry_set)
    enc2 = OneHotEncoder(categories=txvariantcode_set)
    enc3 = OneHotEncoder(categories=currencycode_set)
    enc4 = OneHotEncoder(categories=shoppercountry_set)
    enc5 = OneHotEncoder(categories=accountcode_set)
    data1 = enc1.transform(data[:, 0])
    data2 = enc2.transform(data[:, 1])
    data3 = enc3.transform(data[:, 4])
    data4 = enc4.transform(data[:, 5])
    data5 = enc5.transform(data[:, 10])
    datas = np.c_[data1, data2, data[:, [2, 3]], data3, data4, data[:, [6, 7, 8, 9]], data5, data[:, [11, 12, 13, 14, 15]]]
    xh = []
    for item in datas:
        xh.append(item[0:-2])

    x = []  # contains features
    y = []  # contains labels

    for item in data:  # split data into x,y
        x.append(item[0:-2])
        y.append(item[-2])

    '''map number to each categorial feature'''
    for item in list(issuercountry_set):
        issuercountry_dict[item] = list(issuercountry_set).index(item)
    for item in list(txvariantcode_set):
        txvariantcode_dict[item] = list(txvariantcode_set).index(item)
    for item in list(currencycode_set):
        currencycode_dict[item] = list(currencycode_set).index(item)
    for item in list(shoppercountry_set):
        shoppercountry_dict[item] = list(shoppercountry_set).index(item)
    for item in list(interaction_set):
        interaction_dict[item] = list(interaction_set).index(item)
    for item in list(verification_set):
        verification_dict[item] = list(verification_set).index(item)
    for item in list(accountcode_set):
        accountcode_dict[item] = list(accountcode_set).index(item)
    print(len(list(card_id_set)))
    # for item in list(card_id_set):
    #    card_id_dict[item] = list(card_id_set).index(item)
    '''modify categorial feature to number in data set'''
    for item in x:
        item[0] = issuercountry_dict[item[0]]
        item[1] = txvariantcode_dict[item[1]]
        item[4] = currencycode_dict[item[4]]
        item[5] = shoppercountry_dict[item[5]]
        item[6] = interaction_dict[item[6]]
        item[7] = verification_dict[item[7]]
        item[10] = accountcode_dict[item[10]]
    return x, y, xh


def log_reg(x, y, s):
    x_array = np.array(x)
    y_array = np.array(y)
    usx = x_array
    usy = y_array
    x_train, x_test, y_train, y_test = train_test_split(usx, usy, test_size=s)

    cls_log = LogisticRegression()
    cls_log.fit(x_train, y_train)
    y_predict = cls_log.predict(x_test)
    y_pred_prob = cls_log.predict_proba(x_test)[:, 1]
    return y_predict, y_test, y_pred_prob


def kNN(x, y, s):
    x_array = np.array(x)
    y_array = np.array(y)
    usx = x_array
    usy = y_array
    x_train, x_test, y_train, y_test = train_test_split(usx, usy, test_size=s)

    clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    y_pred_prob = clf.predict_proba(x_test)[:, 1]
    return y_predict, y_test, y_pred_prob


def RForest(x, y, s):
    x_array = np.array(x)
    y_array = np.array(y)
    usx = x_array
    usy = y_array
    x_train, x_test, y_train, y_test = train_test_split(usx, usy, test_size=s)
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    y_pred_prob = clf.predict_proba(x_test)[:, 1]
    return y_predict, y_test, y_pred_prob


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


if __name__ == "__main__":
    og_x, og_y = read_data('data_for_student_case.csv')

    # y_predict_log, y_test, y_prob = log_reg(og_x, og_y, 0.2)
    # calc_eff(y_predict_log, y_test)
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
    # plt.figure(1)
    # plt.plot(fpr, tpr, 'r', label="logistic regression")

    y_predict_knn, y_test, y_prob = kNN(og_x, og_y, 0.2)
    calc_eff(y_predict_knn, y_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
    plt.figure(2)
    plt.plot(fpr, tpr, 'r', label="kNeighbors")

    # y_predict_rf, y_test, y_prob = RForest(og_x, og_y, 0.2)
    # calc_eff(y_predict_rf, y_test)
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
    # plt.figure(3)
    # plt.plot(fpr, tpr, 'r', label="Random Forests")
    og_x = None

    sm_x, sm_y = read_data('SMOTED_data.csv', s=1)
    # y_predict_sm_log, y_sm_test, y_sm_prob = log_reg(sm_x, sm_y, 0.2)
    # calc_eff(y_predict_sm_log, y_sm_test)
    # fpr, tpr, thresholds = metrics.roc_curve(y_sm_test, y_sm_prob)
    # plt.figure(1)
    # plt.plot(fpr, tpr, 'b', label="SMOTED logistic regression")
    # plt.xlabel("False positive rate")
    # plt.ylabel("True positive rate")
    # plt.legend()

    y_predict_sm_knn, y_sm_test, y_sm_prob = kNN(sm_x, sm_y, 0.2)
    calc_eff(y_predict_sm_knn, y_sm_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_sm_test, y_sm_prob)
    plt.figure(2)
    plt.plot(fpr, tpr, 'b', label="SMOTED kNeighbors")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()

    # y_predict_sm_rf, y_sm_test, y_sm_prob = RForest(sm_x, sm_y, 0.2)
    # calc_eff(y_predict_sm_rf, y_sm_test)
    # fpr, tpr, thresholds = metrics.roc_curve(y_sm_test, y_sm_prob)
    # plt.figure(3)
    # plt.plot(fpr, tpr, 'b', label="SMOTED Random Forests")
    # plt.xlabel("False positive rate")
    # plt.ylabel("True positive rate")
    # plt.legend()

    plt.show()


