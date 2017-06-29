# coding=utf-8

import pandas as pd
import time
from apriori import apriori


if __name__ == '__main__':
    train_set = pd.read_csv('GSM/new2gtrain.csv')
    train = train_set.head(800).groupby('IMSI')['GridID'].apply(list)
    train = map(lambda a: list(set(a)), train)
    start = time.time()
    res = apriori(train, minSupport=1.0)
    end = time.time()
    print end - start
    print list(res)
