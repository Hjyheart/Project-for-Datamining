# coding: UTF-8

from sklearn.ensemble import RandomForestClassifier
from buffer.DataBuffer import DataBuffer as buffer
from utils.CSVReader import CSVReader
from utils.RFDataProcessor import RFDataProcessor
from sklearn.model_selection import train_test_split
import utm
from math import sqrt
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from utils.GridCutUtil import GridCutUtil as gutil


class RFCProcessor(object):
    predictors = [u'SRNCID', u'BestCellID', u'RSSI_1', u'RSSI_2', u'RSSI_3', u'RSSI_4', u'RSSI_5', u'RSSI_6']

    def __init__(self):
        self.data_set_2g = buffer().train_set_2g.append(buffer().test_set_2g, ignore_index=True)
        self.data_set_4g = buffer().train_set_4g.append(buffer().test_set_4g, ignore_index=True)
        self.X_2g = self.data_set_2g.ix[:, self.predictors]
        self.y_2g = self.data_set_2g['GridID']
        self.X_4g = self.data_set_4g.ix[:, self.predictors]
        self.y_4g = self.data_set_4g['GridID']
        self.processor = RandomForestClassifier(n_estimators=10, n_jobs=-1)

    def split_dataset(self, X, y, test_size):
        return train_test_split(X, y, test_size=test_size)

    def classify_2g(self):
        X_train_2g, X_test_2g, y_train_2g, y_test_2g = self.split_dataset(self.X_2g, self.y_2g, 0.2)
        self.processor.fit(X_train_2g, y_train_2g)
        return self.processor.predict(X_test_2g), y_test_2g, self.processor.score(X_test_2g, y_test_2g)

    def classify_4g(self):
        X_train_4g, X_test_4g, y_train_4g, y_test_4g = self.split_dataset(self.X_4g, self.y_4g, 0.2)
        self.processor.fit(X_train_4g, y_train_4g)
        return self.processor.predict(X_test_4g), y_test_4g, self.processor.score(X_test_4g, y_test_4g)

    def cal_deviation(self, dataset, result, test):
        center_ls = []
        g = gutil(dataset)
        for i in range(len(result)):
            center = g.cal_center(result[0])
            center_ls.append(utm.to_latlon(center[1], center[0], g._min[2], g._min[3]))
        center_ls = np.array(center_ls)
        test.index = range(0, len(test))
        center_ls = pd.DataFrame(center_ls, columns=['Pre_Latitude', 'Pre_Longitude'])
        cls_eval = pd.concat([test, center_ls], axis=1)
        dis=0
        for i in range(0, len(cls_eval)):
            dis += g.haversine(dataset.loc[i, "Longitude"], dataset.loc[i, "Latitude"],
                               cls_eval.loc[i, "Pre_Longitude"], cls_eval.loc[i, "Pre_Latitude"])
        return sqrt(dis)

if __name__ == '__main__':
    reader = CSVReader()
    RFDataProcessor()
    r = RFCProcessor()

    dev_2g, dev_4g = [], []
    for i in range(10):
        result_2g, test_2g, score_2g = r.classify_2g()
        dev_2g.append(r.cal_deviation(r.data_set_2g, result_2g, test_2g))
    plt.plot(dev_2g)
    plt.show()
    for i in range(10):
        result_4g, test_4g, score_4g = r.classify_4g()
        dev_4g.append(r.cal_deviation(r.data_set_4g, result_4g, test_4g))
    plt.plot(dev_4g)
    plt.show()
    # f = open("rf_classify.txt", "w")
    # f.write('\n'.join(map(str,y_test)))
    # print r.score(y_test)
    # print (y_test == buffer().test_set['GridID'][:]).mean()
