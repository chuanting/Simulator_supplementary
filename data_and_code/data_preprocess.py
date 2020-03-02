# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: data_preprocess.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-03-02 (YYYY-MM-DD)
-----------------------------------------------
"""
import pandas as pd
from collections import defaultdict
import numpy as np

names = ['idx', 'throughput', 'goodput', 'energy_cost', 'time_cost', 'packet_loss', 'loss_threshold']
indicators = ['throughput', 'energy_cost', 'time_cost', 'packet_loss']
# df_data = pd.read_csv('D:/OneDrive - mail.sdu.edu.cn/CodeProject/'
#                             'FLinMEN/Simulator/toy_examples/results/different_batch_size.csv', header=0, names=names)
# df_data = pd.read_csv('D:/OneDrive - mail.sdu.edu.cn/CodeProject/'
#                       'FLinMEN/Simulator/toy_examples/results/different_clients.csv', header=0, names=names)
df_data = pd.read_csv('D:/OneDrive - mail.sdu.edu.cn/CodeProject/'
                      'FLinMEN/Simulator/toy_examples/results/different_epsilon.csv', header=0, names=names)

df_data.drop(['idx'], inplace=True, axis=1)
#
# results = defaultdict(dict)
# clients = [10, 20, 40, 80, 160]
# df_list = []
# loss_threshold = df_data['loss_threshold'].unique()
#
# df = pd.DataFrame()
# df['client'] = clients * 3
# df['threshold'] = np.repeat(loss_threshold, 5)
# for ind in indicators:
#     x = []
#     n_rows = df_data[ind].shape[0]
#     for row in range(n_rows):
#         values = df_data[ind][row].strip('][').split(', ')
#         row_values = [float(v) for v in values]
#         x.extend(row_values)
#     df[ind] = x
#
# df.to_csv('d:/sys_performance_clients.csv', index=False)


# different epsilon values
results = defaultdict(dict)
epsilons = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
df_list = []
loss_threshold = df_data['loss_threshold'].unique()

df = pd.DataFrame()
df['epsilon'] = epsilons * 3
df['threshold'] = np.repeat(loss_threshold, 7)
for ind in indicators:
    x = []
    n_rows = df_data[ind].shape[0]
    for row in range(n_rows):
        values = df_data[ind][row].strip('][').split(', ')
        row_values = [float(v) for v in values]
        x.extend(row_values)
    df[ind] = x
print(df)
df.to_csv('d:/sys_performance_epsilons.csv', index=False)
