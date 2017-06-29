# -*- coding: UTF-8 -*-
import math
import pandas as pd
import numpy as np
import utm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

gsm_train = pd.read_csv('GSM/new2gtrain.csv')
gsm_test = pd.read_csv('GSM/new2gtest.csv')

lte_train = pd.read_csv('LTE/new4gtrain.csv')
lte_test = pd.read_csv('LTE/new4gtrain.csv')

class RandomTree(object):

    def __init__(self, train, test):
        '''
        prepare datas
        :param train: training data
        :param test: testing data
        '''
        self.train = train
        self.test = test

        # prepare data
        for i in range(1, 7):
            self.train['RSSI_' + str(i)] = abs(self.train['RSCP_' + str(i)] - self.train['EcNo_' + str(i)])
            self.test['RSSI_' + str(i)] = abs(self.test['RSCP_' + str(i)] - self.test['EcNo_' + str(i)])

        self.total = self.test.append(self.train, ignore_index=True)
        self.tests = self.total.ix[:, [u'SRNCID', u'BestCellID', u'RSSI_1', u'RSSI_2', u'RSSI_3', u'RSSI_4', u'RSSI_5', u'RSSI_6']]

        self.regressor_y = self.total[['Longitude', 'Latitude']]
        self.classifier_y = self.total['GridID']

        self.regressor = RandomForestRegressor(random_state=0, n_estimators=10)
        self.classifier = RandomForestClassifier(n_estimators=10)

        self.classifier_train_x, self.classifier_test_x, self.classifier_train_y, self.classifier_test_y = \
            train_test_split(self.tests, self.classifier_y, test_size=0.2)
        self.regressor_train_x, self.regressor_test_x, self.regressor_train_y, self.regressor_test_y = \
            train_test_split(self.tests, self.regressor_y, test_size=0.2)

        # calculate center mark
        self.min_location = utm.from_latlon(self.total['Latitude'].min(), self.total['Longitude'].min())
        self.max_location = utm.from_latlon(self.total['Latitude'].max(), self.total['Longitude'].max())
        width = self.max_location[1] - self.min_location[1]
        height = self.max_location[0] - self.min_location[0]
        self.grid_x = math.ceil(width / 20)
        self.grid_y = math.ceil(height / 20)

    def findCenter(self, num):
        '''
        find the center
        :param num: grid id
        :return: center mark
        '''
        dr = math.ceil(num / self.grid_x)
        dc = num % self.grid_y
        c_x = self.min_location[1] + dc * 20 - 10
        c_y = self.min_location[0] + dr * 20 - 10
        return [c_x, c_y]

    def distance(self, lo1, la1, lo2, la2):
        '''
        calculate distance
        :param lo1: longitude1
        :param la1: latitude1
        :param lo2: longitude2
        :param la2: latitude2
        :return: distance
        '''
        dlon = lo2 - lo1
        dlat = la2 - la1
        return math.sqrt(dlon * dlon + dlat * dlat)

    def predict(self):
        '''
        train ans predict
        :return: regressor_res and classifier_res
        '''
        self.classifier_train_x, self.classifier_test_x, self.classifier_train_y, self.classifier_test_y = \
            train_test_split(self.tests, self.classifier_y, test_size=0.2)
        self.regressor_train_x, self.regressor_test_x, self.regressor_train_y, self.regressor_test_y = \
            train_test_split(self.tests, self.regressor_y, test_size=0.2)

        self.regressor.fit(self.regressor_train_x, self.regressor_train_y)
        self.classifier.fit(self.classifier_train_x, self.classifier_train_y)

        regressor_res = self.regressor.predict(self.regressor_test_x)
        classifier_res = self.classifier.predict(self.classifier_test_x)
        r_score = self.regressor.score(self.regressor_test_x, self.regressor_test_y)
        c_score = self.classifier.score(self.classifier_test_x, self.classifier_test_y)
        print 'regressor score: ' + str(r_score)
        print 'classifer score: ' + str(c_score)

        return regressor_res, classifier_res, r_score, c_score

    def compare(self):
        '''
        compare
        :return:
        '''
        regressor_com = []
        classifier_com = []
        r_score = []
        c_score = []
        for i in range(0, 10):
            regressor_res, classifier_res, r, c = self.predict()
            r_score.append(r)
            c_score.append(c)
            self.regressor_test_y.index = range(0, len(self.regressor_test_y))
            regressor_res = pd.DataFrame(regressor_res, columns=['PLO', 'PLA'])

            c_ls = []
            for i in range(len(classifier_res)):
                center = self.findCenter(classifier_res[0])
                c_ls.append(utm.to_latlon(center[1], center[0], self.min_location[2], self.min_location[3]))

            c_ls = pd.DataFrame(c_ls, columns=['PLA', 'PLO'])

            r_eval = pd.concat([self.regressor_test_y, regressor_res], axis=1)
            c_eval = pd.concat([self.regressor_test_y, c_ls], axis=1)

            for i in range(0, len(r_eval)):
                r_dis = self.distance(r_eval.loc[i, 'Longitude'], r_eval.loc[i, 'Latitude'], r_eval.loc[i, 'PLO'],
                                    r_eval.loc[i, 'PLA'])
                c_dis = self.distance(c_eval.loc[i, 'Longitude'], c_eval.loc[i, 'Latitude'], c_eval.loc[i, 'PLO'],
                                      c_eval.loc[i, 'PLA'])
                regressor_com.append(r_dis)
                classifier_com.append(c_dis)


        # r_score = np.average(r_score)
        # c_score = np.average(c_score)

        plt.plot(r_score, color='red')
        plt.xlabel('time')
        plt.ylabel('average')
        plt.show()

        plt.plot(c_score, color='blue')
        plt.xlabel('time')
        plt.ylabel('average')
        plt.show()

        regressor_com.sort()
        classifier_com.sort()

        plt.plot(regressor_com, color='red')
        plt.xlabel('index')
        plt.ylabel('distance')
        plt.show()

        plt.plot(classifier_com, color='blue')
        plt.xlabel('index')
        plt.ylabel('distance')
        plt.show()



if __name__ == '__main__':
    print '2G'
    g_test = RandomTree(train=gsm_train, test=gsm_test)
    g_test.compare()
    print '4G'
    l_test = RandomTree(train=lte_train, test=lte_test)
    l_test.compare()
