# -*- coding:UTF-8 -*-
import pandas as pd
import numpy as np
from lshash import LSHash
from CoordTransform import wgs84togcj02
import utm
from sklearn.neighbors import NearestNeighbors


def pack_data(index):
    '''
    将原先的轨迹打包装入列表中
    :param index: 路径编号
    :return: 打包完的路径
    '''
    data = []
    doc = pd.read_csv('Traj_1000_SH_UTM')
    doc = doc.groupby(doc['Tid'])
    for line in doc.get_group(index).iterrows():
        point = utm.to_latlon(line[1][2], line[1][3], 51, 'U')
        point = wgs84togcj02(point[1], point[0])
        data.append([point[0], point[1]])
    return data


def write_json_lsh(hash_size, grid):
    '''
    将生成的lsh路径放入json并存储
    :param hash_size: hash size列表
    :param grid: 处理完的栅格数组
    :return: none
    '''
    data_lsh = {}
    for size in hash_size:
        print size
        print 'list'
        data_lsh[size] = []
        lsh = LSHash(size, 44107)
        count = 0
        for line in grid:
            lsh.index(line, extra_data=count)
            count += 1
        for id in road_id:
            roads = []
            res = lsh.query(grid[id])
            print len(res)
            for r in res:
                roads.append(pack_data(r[0][1]))
            data_lsh[size].append({id: roads})

    with open('result_lsh.json', 'w') as f:
        f.write(str(data_lsh))


def write_json_nn(knn, grid):
    '''
    将生成的knn路径放入json并存储
    :param knn: k的数值列表
    :param grid: 处理完的栅格数组
    :return: none
    '''
    for nn in knn:
        data_knn = []
        neigh = NearestNeighbors(n_neighbors=nn)
        neigh.fit(grid)
        print 'nn:' + str(nn)
        for id in road_id:
            print 'id:' + str(id)
            roads = []
            distances, indices = neigh.kneighbors(grid[id])
            for r in indices[0]:
                roads.append(pack_data(r))
            data_knn.append({id: roads})

        with open('result_knn' + str(nn) + '.json', 'w') as f:
            f.write(str(data_knn))


if __name__ == '__main__':
    df = pd.read_csv('Traj_1000_SH_UTM')
    df['X'] = ((df['X'] - 346000) / 20).astype(int)
    df['Y'] = ((df['Y'] - 362800) / 20).astype(int)

    storage1 = np.zeros((3448600 / 20, 3463800 / 20))

    k = df.groupby((df['X'], df['Y']))
    grid = np.zeros((44107, 1000))

    index = 0
    for line in k:
        storage1[line[0][0], line[0][1]] = index
        index += 1

    for line in df.iterrows():
        grid[storage1[line[1]['X'], line[1]['Y']], line[1]['Tid'] - 1] = 1

    grid = grid.T

    # 14, 250，480，690，900
    hash_size = [10, 11, 12, 13, 14, 15]
    road_id = [14, 249, 479, 689, 899]
    knn = [3, 4, 5]

    data_knn = {}

    # LSH
    write_json_lsh(hash_size, grid)

    # NN
    write_json_nn(knn, grid)



