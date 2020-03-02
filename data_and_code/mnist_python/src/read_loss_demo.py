# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: read_loss_demo.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-03-01 (YYYY-MM-DD)
-----------------------------------------------
"""
import pickle
import pandas as pd
import numpy as np

with open('../save/objects/mnist_cnn_50_C[1]_iid[1]_E[1]_B[64].pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)

    df = pd.DataFrame(np.array(data[:-1]).T, columns=['train_loss', 'train_acc'])
    df.to_csv('d:/mnist_fl_guoqing.csv', index=False)