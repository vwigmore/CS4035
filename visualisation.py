# coding: utf-8

import datetime
import time
import seaborn as sns
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np


def string_to_timestamp(date_string):#convert time string to float value
    time_stamp = time.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    return time.mktime(time_stamp)


if __name__ == "__main__":

    src = 'data_for_student_case.csv'
    ah = open(src, 'r')

    x = []  # contains features
    y = []  # contains labels
    data = []
    color = []
    (issuercountry_set, txvariantcode_set, currencycode_set, shoppercountry_set, interaction_set,
     verification_set, accountcode_set, mail_id_set, ip_id_set, card_id_set) = [set() for _ in xrange(10)]
    (issuercountry_dict, txvariantcode_dict, currencycode_dict, shoppercountry_dict, interaction_dict,
     verification_dict, accountcode_dict, mail_id_dict, ip_id_dict, card_id_dict) = [{} for _ in xrange(10)]
    # label_set
    ah.readline()  # skip first line
    count = 0
    for line_ah in ah:
        if line_ah.strip().split(',')[9]=='Refused':# remove the row with 'refused' label, since it's uncertain about fraud
            continue
        if 'na' in str(line_ah.strip().split(',')[14]).lower() or 'na' in str(line_ah.strip().split(',')[4].lower()):
            continue
        bookingdate = string_to_timestamp(line_ah.strip().split(',')[1])# date reported flaud
        issuercountry = line_ah.strip().split(',')[2]#country code
        issuercountry_set.add(issuercountry)
        txvariantcode = line_ah.strip().split(',')[3]#type of card: visa/master
        txvariantcode_set.add(txvariantcode)
        issuer_id = float(line_ah.strip().split(',')[4])#bin card issuer identifier
        amount = float(line_ah.strip().split(',')[5])#transaction amount in minor units
        currencycode = line_ah.strip().split(',')[6]
        currencycode_set.add(currencycode)
        shoppercountry = line_ah.strip().split(',')[7]#country code
        shoppercountry_set.add(shoppercountry)
        interaction = line_ah.strip().split(',')[8]#online transaction or subscription
        interaction_set.add(interaction)
        if line_ah.strip().split(',')[9] == 'Chargeback':
            label = 1#label fraud
            count += 1
        else:
            label = 0#label save
        verification = line_ah.strip().split(',')[10]#shopper provide CVC code or not
        verification_set.add(verification)
        cvcresponse = line_ah.strip().split(',')[11]#0 = Unknown, 1=Match, 2=No Match, 3-6=Not checked
        if cvcresponse > 2:
            cvcresponse = 3
        year_info = datetime.datetime.strptime(line_ah.strip().split(',')[12],'%Y-%m-%d %H:%M:%S').year
        month_info = datetime.datetime.strptime(line_ah.strip().split(',')[12],'%Y-%m-%d %H:%M:%S').month
        day_info = datetime.datetime.strptime(line_ah.strip().split(',')[12],'%Y-%m-%d %H:%M:%S').day
        creationdate = str(year_info)+'-'+str(month_info)+'-'+str(day_info)#Date of transaction
        creationdate_stamp = string_to_timestamp(line_ah.strip().split(',')[12])#Date of transaction-time stamp
        accountcode = line_ah.strip().split(',')[13]#merchant’s webshop
        accountcode_set.add(accountcode)
        mail_id = int(float(line_ah.strip().split(',')[14].replace('email','')))#mail
        mail_id_set.add(mail_id)
        ip_id = int(float(line_ah.strip().split(',')[15].replace('ip','')))#ip
        ip_id_set.add(ip_id)
        card_id = int(float(line_ah.strip().split(',')[16].replace('card','')))#card
        card_id_set.add(card_id)
        data.append([issuercountry, txvariantcode, issuer_id, amount, currencycode,
                    shoppercountry, interaction, verification, cvcresponse, creationdate_stamp,
                     accountcode, mail_id, ip_id, card_id, label, creationdate])# add the interested features here

    # Create a HeatMap of the number of fraud cases
    par1 = txvariantcode_set
    par2 = currencycode_set

    heat_map = np.zeros((len(par1), len(par2)))
    plist1 = list(par1)
    plist2 = list(par2)

    # count the number of frauds in each of the chosen categories
    for [issuercountry, txvariantcode, issuer_id, amount, currencycode,
                    shoppercountry, interaction, verification, cvcresponse, creationdate_stamp,
                     accountcode, mail_id, ip_id, card_id, label, creationdate] in data:
        if label == 1:
            i = plist1.index(txvariantcode)
            j = plist2.index(currencycode)
            heat_map[i][j] = heat_map[i][j] + 1

    plt.figure(1)
    sns.heatmap(heat_map, xticklabels=plist2, yticklabels=plist1)

    # Create a scatterplot of the average transaction and the deviation per transaction
    data2 = sorted(data, key=itemgetter(9))
    dictionary = {}
    maxelements = 0

    for [issuercountry, txvariantcode, issuer_id, amount, currencycode,
         shoppercountry, interaction, verification, cvcresponse, creationdate_stamp,
         accountcode, mail_id, ip_id, card_id, label, creationdate] in data2:

        if str(card_id) not in dictionary:
           newArray = []
           templist = []

           templist.append(amount)
           templist.append(currencycode)
           templist.append(label)

           newArray.append(templist)

           dictionary.update({str(card_id): newArray})
        else:
            templist = []
            array = dictionary.get(str(card_id))

            templist.append(amount)
            templist.append(currencycode)
            templist.append(label)

            array.append(templist)
            dictionary.update({str(card_id): array})

    Xgood = []
    Xbad = []
    Ygood = []
    Ybad = []

    # print("at the scatter")
    count = 0

    for key in dictionary:
        list = dictionary[key]

        el = len(list)
        if len(list) < 2:
            continue

        conversion_dict = {'SEK': 0.09703, 'MXN': 0.04358, 'AUD': 0.63161, 'NZD': 0.58377, 'GBP': 1.13355}
        total = 0
        for x in list:
            Rate = conversion_dict[x[1]]

            total += total + x[0] * Rate

        for x in list:
            if x[2] is 1:

                Rate = conversion_dict[x[1]]

                avg = (total - x[0] * Rate) / 100
                avg = avg / (el - 1)
                Xbad.append(avg)
                Ybad.append((x[0] * Rate) / 100)
                count += 1
            else:
                Rate = conversion_dict[x[1]]

                avg = (total - x[0] * Rate) / 100
                avg = avg / (el - 1)

                if avg < 15000:
                    Xgood.append(avg)
                    Ygood.append((x[0] * Rate) / 100)

    plt.figure(2)
    plt.scatter(Xgood, Ygood, c='blue', label='non-fraudulent transactions')
    plt.scatter(Xbad, Ybad, c='red', label='fraudulent transactions')

    plt.title("Comparison transactions deviating from the average amount")
    plt.xlabel("Average transaction amount - transaction t")
    plt.ylabel("transaction t")
    plt.legend()

    plt.show()
    print(count)






