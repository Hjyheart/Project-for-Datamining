# -*- coding: UTF-8 -*-
import itertools
# TODO: 对数据扫描出来的结果进行排序
# TODO: napkin bug
# TODO: 除去一个之后递归挖掘到只剩一条路径就行了
class Fptree(object):

    def __init__(self, support):
        self.tree = [{'value': None, 'count': -1}]
        self.support = support
        self.f = open('res.txt', 'a+')

    def finish(self):
        self.f.close()

    def build(self, start_node, data):
        out = []
        order, data = self.sort_data(data)
        print 'sort data:' + str(data)
        record_table = []
        for line in data:
            if len(line) > 0:
                start_node, record_table = self.add_node(line, start_node, 0, record_table)

        while len(order) > 0:
            l = sorted(order.iteritems(), key=lambda a: a[1], reverse=True)
            min_item_name = l[len(l) - 1][0]
            print 'min item name: ' + min_item_name
            new_data = []
            for r in record_table:
                if len(r['key']) > 0 and r['key'][len(r['key']) - 1] == min_item_name:
                    for i in range(0, r['count']):
                        new_data.append(r['key'][0:len(r['key']) - 1])
            print 'new data: ' + str(new_data)
            if len(order) >= 0:
                out = self.sub_build([{'value': None, 'count': -1}], new_data, min_item_name)
                order.pop(min_item_name)
                self.f.write('{' + min_item_name + ':')
                self.f.write(str(out)[1:-1])
                self.f.write('}')

    def sub_build(self, node, data, last_item):
        out = []
        order, data = self.sort_data(data)
        record_table = []
        for line in data:
            if len(line) > 0:
                start_node, record_table = self.add_node(line, node, 0, record_table)

        if self.judge_single(node) and last_item != None:
            return self.combination(node, last_item)
        else:
            l = sorted(order.iteritems(), key=lambda a: a[1], reverse=True)
            min_item_name = l[len(l) - 1][0]
            new_data = []
            for r in record_table:
                if len(r['key']) > 0 and r['key'][len(r['key']) - 1] == min_item_name:
                    for i in range(0, r['count']):
                        new_data.append(r['key'][0:len(r['key']) - 1])

            return self.sub_build([{'value': None, 'count': -1}], new_data, min_item_name)


    def add_node(self, line, node, index, record_table):
        point = line[index]
        for item in node:
            if isinstance(item, list):
                if item[0]['value'] == point:
                    item[0]['count'] += 1
                    for record in record_table:
                        if line[0:index + 1] == record['key']:
                            record['count'] += 1

                    if index < len(line) - 1:
                        item, record_table = self.add_node(line, item, index + 1, record_table)
                        return node, record_table
                    else:
                        return node, record_table
        node.append([{'value': point, 'count': 1}])
        record_table.append({'key': line[0:index + 1], 'count': 1})

        if index < len(line) - 1:
            node[len(node) - 1], record_table = self.add_node(line, node[len(node) - 1], index + 1, record_table)
            return node, record_table
        else:
            return node, record_table

    def sort_data(self, data):
        res = {}
        for line in data:
            for d in line:
                if res.has_key(d):
                    res[d] += 1
                else:
                    res[d] = 1
        new_res = {}
        for line in res.iteritems():
            if line[1] >= self.support:
                new_res[line[0]] = line[1]

        new_data = []
        for line in data:
            tmp = []
            for item in line:
                if res[item] >= self.support:
                    tmp.append(item)
            new_data.append(tmp)

        for line in new_data:
            for i in range(0, len(line) - 1):
                for k in range(i + 1, len(line)):
                    if res[line[k]] > res[line[i]]:
                        tmp = line[k]
                        line[k] = line[i]
                        line[i] = tmp
        return new_res, new_data

    def judge_single(self, node):
        if len(node) == 1:
            return True
        while len(node) == 2:
            node = node[1]
        if len(node) > 2:
            return False
        return True

    def combination(self, node, last_item_name):
        list1 = []
        res = []
        while len(node) == 2:
            if node[0].has_key('value') and node[0]['value'] != None:
                list1.append(node[0]['value'])
            node = node[1]
        if node[0]['value'] != None:
            list1.append(node[0]['value'])

        for i in range(1, len(list1) + 1):
            iter = itertools.combinations(list1, i)
            for line in list(iter):
                line += (last_item_name,)
                res.append(line)
        res.append((last_item_name,))
        print 'conbine:' + str(res)
        return res

test = Fptree(3)

test.build([{'value': None, 'count': -1}], [
    ['bread', 'milk'],
    ['milk', 'napkin', 'peer', 'egg'],
    ['bread', 'napkin', 'peer', 'cola'],
    ['milk', 'bread', 'napkin', 'peer'],
    ['milk', 'bread', 'napkin', 'cola']
])
test.finish()
