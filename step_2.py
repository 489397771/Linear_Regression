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
import numpy as np
import pandas as pd


def split_list(lst=None, block=5):
    """
    将列表平均分割, 返回[[], [], ...]形式的二维列表
    """
    length = len(lst)
    sub_lenght = length / block
    last_list = []
    for i in range(block):
        last_list.append(lst[:sub_lenght])
        del lst[:sub_lenght]
    return last_list


def sum_sub_lst(lst=None):
    """
    计算子列表的和的平均值
    """
    return [sum(i) / len(i) for i in lst]


def data_process(df, columns, rows):
    list_all = []
    for column in columns:
        my_list = []
        for i in range(1, rows - 49):
            list_temp = []
            for j in range(i, i + 50):
                list_temp.append(df[column][j])
            sub_list_temp = sum_sub_lst(split_list(list_temp))
            my_list.append(sub_list_temp)
        list_all.append(my_list)
    return list_all


def modify_columns(columns, column_length=65):
    columns = list(columns)
    c = [[column + '_' + str(i) for i in range(1, 6)] for column in columns]
    return list(np.array(c).reshape(1, -1))


filepath = './Step1_dropna.csv'
df = pd.read_csv(filepath)
columns = df.columns.values  # <type 'numpy.ndarray'>
rows = len(df)  # 1263
y = df['CLOSE_USDCNY.FX']
y = pd.DataFrame(y)[:rows]
numpy_array = np.array(data_process(df=df, columns=columns, rows=rows))  # (13, 1213, 5)
lst = []
for i in range(numpy_array.shape[1]):
    lst.append(numpy_array[:, i, :].ravel())
last_array = np.array(lst)
df = pd.DataFrame(last_array, columns=modify_columns(columns=columns))  # (1213, 65)
df = df.join(y).rename(columns={'CLOSE_USDCNY.FX': 'y_CLOSE_USDCNY.FX'})
df.to_csv('./training_data_b.csv', index=None)
