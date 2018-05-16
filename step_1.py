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
import pandas as pd
from pandas import DataFrame

pd.set_option('display.max_rows', None)
__author__ = 'FC_Song'

# 1.准备训练数据a: 利用前5天所有汇率的收市价预测明日人名币兑美元收市价格
filepath = './currency_exchange_rates_output.csv'
datasets = pd.read_csv(filepath)[['TIME', 'NAME', 'CLOSE']]
# 获取13种不同汇率
name_values = datasets['NAME'].unique()  
# 获取所有日期
time_values = datasets['TIME'].unique() 
# 挑选出1307条不同的时间转换为DataFrame结构
time_df = DataFrame(time_values)  
df_list = [datasets.where(datasets['NAME'] == name).dropna().rename
           (columns={'CLOSE': 'CLOSE' + '_' + name}).iloc
           [:, [-1]].reset_index(drop=True) for name in name_values]
df = df_list[0]
for i in df_list[1:]:
    df = df.join(i)
# df = time_df.join(df).rename(columns={0: 'TIME'})  # 在汇率前加一列时间
# 第一种版本: 替换缺失值, index从1263开始没有值, 用最下面的值6.223填充NaN
df1 = df.fillna(method='pad')
df1.to_csv('./step1_fillna.csv', index=None)
# 第二种版本: 过滤掉44行缺失值, 共1263行
df2 = df.iloc[:-44]
df2.to_csv('./step1_dropna.csv', index=None)

##############################################
# 修改65个列名
mylist = []
for i in range(4, -1, -1):
    columns_list = []
    for j in list(name_values):
        columns_list.append('CLOSE' + '_' + j + '_' + str(i))
    mylist.extend(columns_list)
mydict = dict(zip([i for i in range(65)], mylist))
##############################################

df0 = df2
for i in range(1, 2):
    tem_list = [df0.iloc[j] for j in range(i, i + 5)]
    series_ = tem_list[0]
    for i in tem_list[1:]:
        series_ = series_.append(i)
    series_ = series_.values.reshape(1, -1)
dfs = DataFrame(series_)

for i in range(2, len(df0) - 4):
    tem_list = [df0.iloc[j] for j in range(i, i + 5)]
    series_ = tem_list[0]
    for i in tem_list[1:]:
        series_ = series_.append(i)
    series_ = series_.values.reshape(1, -1)
    dfs = dfs.append(DataFrame(series_))
dfs = dfs.reset_index(drop=True)  # 1258
y_usdcny = DataFrame(df0['CLOSE_USDCNY.FX'][:len(dfs)]).rename(columns={'CLOSE_USDCNY.FX': 'y_CLOSE_USDCNY.FX'})
dfs = pd.concat([dfs, y_usdcny], axis=1)
# time = DataFrame(time_values[: len(dfs)]).rename(columns={0: 'TIME'})
# dfs = pd.concat([time, dfs], axis=1)
dfs = dfs.rename(columns=mydict)
dfs.to_csv('./training_data_a.csv', index=None)
