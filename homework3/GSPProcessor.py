# coding=utf-8
import pandas as pd


if __name__ == '__main__':
    train_set = pd.read_csv('GSM/new2gtrain.csv').head(80)
    data = train_set.sort(['MRTime'], ascending=True).groupby(['MRTime']).apply(lambda g: g.groupby('IMSI')['GridID'].apply(list))
    #
    # data = {0:[[3,4],[1,2,3],[1,2,6],[1,3,4,6]],
    #         1:[[1,2,6],[5]],
    #         2:[[1,2,6]],
    #         3:[[4,7,8],[2,6],[1,7,8]]}
    tmp = map(lambda (id, item): {id: item[1].values()}, enumerate(data.to_dict().items()))

    f = open("gsp.txt", "w")
    for d in tmp:
        for v in d.values():
            s = ''
            for i in range(len(v)):
                if i != len(v) - 1:
                    s += ' '.join(str(e) for e in set(v[i])) + ' -1 '
                else:
                    s += ' '.join(str(e) for e in set(v[i])) + ' -2\n'
            f.write(s)

    # starttime = datetime.datetime.now()
    # s = GSPProcessor()
    # freq1 = s.freq1(tmp, 2)
    # s.freq_more(tmp, freq1)
    # endtime = datetime.datetime.now()
    # print(endtime - starttime)