# -*- coding: UTF-8 -*-
import pandas as pd
import time
from apriori import apriori
import pyfpgrowth
import matplotlib.pyplot as plt
import sys

class FPGrowthProcessor(object):
    def fp_growth(self, transactions, support):
        return pyfpgrowth.find_frequent_patterns(transactions, support)


if __name__ == '__main__':
    sys.setrecursionlimit(80000)
    train_set = pd.read_csv('GSM/new2gtrain.csv')

    a_time = []
    f_time = []
    for i in range(1, 10):
        train = train_set.head(i * 100).groupby('IMSI')['GridID'].apply(list)
        train = map(lambda a: list(set(a)), train)

        a_start = time.time()
        apriori(train, minSupport=1.0)
        a_end = time.time()
        print a_end - a_start
        a_time.append((a_end - a_start) * 1000)

        f = FPGrowthProcessor()
        f_start = time.time()
        f.fp_growth(train, 1)
        f_end = time.time()
        print f_end - f_start
        f_time.append((f_end - f_start) * 1000)

    x = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    plt.figure(figsize=(8, 4))
    plt.plot(x, a_time, label="apriori", color="red", linewidth=2)
    plt.plot(x, f_time, color='blue', label="fpgrowth")
    plt.xlabel("number of data")
    plt.ylabel("time(ms)")
    plt.title("apriori and FPGrowth")
    plt.legend()
    plt.show()