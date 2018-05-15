# -*- coding:utf-8 -*-
"""
 ----------------
< Developer 宋馥呈 >
 ----------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||
"""
import csv
from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.cross_validation import train_test_split

dfa = pd.read_csv('./training_data_a.csv')
dfa_cloumns = dfa.columns.values
dfb = pd.read_csv('./training_data_b.csv')
dfb_cloumns = dfb.columns.values

# 拆分数据集
dfa_X = dfa.iloc[:, 0:-1]
dfa_y = dfa.iloc[:, -1]
dfb_X = dfb.iloc[:, 0:-1]
dfb_y = dfb.iloc[:, -1]
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(dfa_X, dfa_y, test_size=0.2, random_state=1)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(dfb_X, dfb_y, test_size=0.2, random_state=1)
# ((1006, 65), (252, 65), (1006,), (252,))
# ((970, 65), (243, 65), (970,), (243,))

column_names = [dfa_cloumns, dfb_cloumns]
models = ['Ridge', 'LassoLars']
alphas = [0, exp(-10), exp(-5), exp(-2), exp(-1), 1, 10]
datasets = [[[X_train_a, y_train_a], [X_test_a, y_test_a], 'training_data_a_'],
            [[X_train_b, y_train_b], [X_test_b, y_test_b], 'training_data_b_']]


def save_result(content, filepath='./model_result.csv', mode='w'):
    with open(filepath, mode) as f:
        writer = csv.writer(f)
        for i in content:
            writer.writerow(i)


def get_n_min_max(lst, N=5, columns=None):
    """
    返回列表中最小的N个值和最大的N个值以及它们对应的索引
    """
    lst = [abs(i) for i in lst]
    np_lst = np.array(lst)
    _sort = np.argsort(np_lst)
    n_min_index = _sort[:N]
    n_max_index = _sort[-N:]
    result = []
    for i in n_min_index:
        result.append((lst[i], columns[i]))
    for i in n_max_index:
        result.append((lst[i], columns[i]))
    return result


def build_model(models, alphas, datasets, column_names):
    content_all = []
    for model in models:
        for alpha in alphas:
            for dataset in datasets:
                if model == 'Ridge':
                    index = 0
                    model_name = model
                    clf = linear_model.Ridge(alpha=alpha)
                else:
                    index = 1
                    model_name = model
                    clf = linear_model.LassoLars(alpha=alpha)
                clf.fit(dataset[0][0], dataset[0][1])
                clf_score = clf.score(dataset[0][0], dataset[0][1])
                clf_coef = clf.coef_
                clf_min_max_five = get_n_min_max(clf_coef, columns=column_names[index])
                y_predict = clf.predict(dataset[1][0])
                plt.plot(range(len(dataset[1][1])), dataset[1][1], label='real')
                plt.plot(range(len(dataset[1][1])), y_predict, label='predict')
                plt.title(dataset[-1] + model_name + '_alpha=' + str(alpha))
                plt.legend(loc='upper right')
                plt.savefig('./picture/' + dataset[-1] + model_name + '_alpha=' + str(alpha) + '.png')
                plt.close()
                content = [model_name, dataset[-1], alpha, clf_score]
                content.extend(clf_min_max_five)
                content_all.append(content)
    save_result(content_all)


build_model(models, alphas, datasets, column_names)
