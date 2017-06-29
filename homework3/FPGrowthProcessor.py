# coding=utf-8
import pandas as pd
import pyfpgrowth
import sys

import time


class FPGrowthProcessor(object):
    def fp_growth(self, transactions, support):
        return pyfpgrowth.find_frequent_patterns(transactions, support)


if __name__ == '__main__':
    sys.setrecursionlimit(80000)
    train_set = pd.read_csv('GSM/new2gtrain.csv')
    train = train_set.head(800).groupby('IMSI')['GridID'].apply(list)
    train = map(lambda a: list(set(a)), train)
    a = FPGrowthProcessor()
    start = time.time()
    res = a.fp_growth(train, 1)
    end = time.time()
    print (end - start)
    print '\nfrequent itemset:\n', res