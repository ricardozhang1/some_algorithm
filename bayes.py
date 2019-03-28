#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/26 16:47
# @Author  : ZhangChaowei


from numpy import *
from pprint import pprint


def load_data_set():
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def create_vocab_list(data_set):
    """对样本数据进行集合并运算"""
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_words_vec(vocab_list, input_set):
    """将词典的样本数据转换成矩阵数据"""
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print(f'the word: {word} is not in my vocabulary')
    return return_vec


def train_naive_bayes(train_matrix, train_category):
    """将矩阵和标记值传入,
        得到的是一个概率矩阵, 和一个概率值"""
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    preb_A = sum(train_category) / float(num_train_docs)
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vec = log(p1_num / p1_denom)
    p0_vec = log(p0_num / p0_denom)
    return p0_vec, p1_vec, preb_A


def classify_naive_bayes(vec2_classify, p0_vec, p1_vec, p_class1):
    """贝叶斯分类器"""
    p1 = sum(vec2_classify * p1_vec) + log(p_class1)
    p0 = sum(vec2_classify * p0_vec) + log(1 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def testing_naive_bayes():
    list_post, list_class = load_data_set()
    my_vocabulary = create_vocab_list(list_post)
    train_mat = []
    for postin_doc in list_post:
        train_mat.append(set_words_vec(my_vocabulary, postin_doc))
    p0_vec, p1_vec, preb_A = train_naive_bayes(train_mat, list_class)

    test_ent = ['love', 'my', 'dalmation']
    this_doc = array(set_words_vec(my_vocabulary, test_ent))  # 返回的是一个矩阵
    print(test_ent, 'classified as: ', classify_naive_bayes(this_doc, p0_vec, p1_vec, preb_A))

    test_ent = ['stupid', 'garbage']
    this_doc = array(set_words_vec(my_vocabulary, test_ent))
    print(test_ent, 'classified as: ', classify_naive_bayes(this_doc, p0_vec, p1_vec, preb_A))


if __name__ == '__main__':
    testing_naive_bayes()





