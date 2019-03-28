#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/21 14:10
# @Author  : ZhangChaowei

from math import log


data_set = [
    [1, 1, 'yes'],
    [1, 1, 'yes'],
    [1, 0, 'no'],
    [0, 1, 'no'],
    [0, 1, 'no']
]

labels = ['no surfacing', 'flippers']


def Ent(data):
    # 计算熵
    data_set_length = len(data)
    data_count = {}
    for i in data:
        sample_label = i[-1]
        if sample_label in data_count.keys():
            data_count[i[-1]] += 1
        else:
            data_count[i[-1]] = 1
    # print(data_count)
    values = 0
    for k, v in data_count.items():
        p = v / data_set_length
        values -= p * log(p, 2)
    return values


def split_data(data, axis, value):  # 传入数据集，维数，值
    # 划分数据集
    new_data = []
    for i in data:
        if i[axis] == value:
            part1 = i[:axis]
            part2 = i[axis+1:]
            part1.extend(part2)
            new_data.append(part1)
    return new_data


def best_split_data(data):
    # 选择最好的数据集划分
    num_feature = len(data[0]) - 1  # 列数-1，少了最后一列
    base_ent = Ent(data)   # 初始化时的熵值
    best_info = 0.0; best_feature = -1
    for i in range(num_feature):
        feature_list = [example[i] for example in data]  # 取第一列到第n-1列
        unique_val = set(feature_list)     # 进行去重处理
        new_ent = 0.0
        for v in unique_val:   # 对去重后的值进行遍历
            sub_data = split_data(data, i, v)
            p = len(sub_data) / float(len(data))
            new_ent += p * Ent(sub_data)
        info_gain = base_ent - new_ent
        if info_gain > best_info:
            best_info = info_gain
            best_feature = i
    return best_feature


def major_cnt(class_list):
    """处理最后一列"""
    class_count = {}
    for i in class_list:
        if i not in class_count.keys():
            class_count[i] = 1
        else:
            class_count[i] += 1
    sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_class_count


def create_tree(data, labels):
    class_list = [example[-1] for example in data]      # 取出最后一列标签
    if class_list.count(class_list[0]) == len(class_list):   # 类别完全相同，跳出迭代
        return class_list[0]
    if len(data[0]) == 1:         # 无法再次进行划分时，跳出迭代
        return major_cnt(class_list)
    best_feature = best_split_data(data)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label: {}}
    del labels[best_feature]
    feat_value = [example[best_feature] for example in data]
    unique_val = set(feat_value)
    for val in unique_val:
        sub_label = labels[:]
        my_tree[best_feature_label][val] = create_tree(split_data(data, best_feature, val), sub_label)
    return my_tree


if __name__ == '__main__':
    # print(Ent(data_set))
    # print(split_data(data_set, 0, 1))
    # print(best_split_data(data_set))
    # class_list = [example[-1] for example in data_set]
    # print(major_cnt(class_list))
    print(create_tree(data_set, labels))
