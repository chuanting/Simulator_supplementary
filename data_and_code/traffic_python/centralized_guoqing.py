# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: centralized_guoqing.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-03-01 (YYYY-MM-DD)
-----------------------------------------------
"""
import numpy as np
import h5py
import datetime
import torch
import sys
sys.path.append('../../')
from FedMMoE.utils.misc_v2 import args_parser
from FedMMoE.utils.misc_v2 import get_data, process_centralized
from FedMMoE.utils.models import LSTM
from torch.utils.data import DataLoader
from sklearn import metrics
import pandas as pd
np.random.seed(2020)
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, _, _, mean, std, _, _ = get_data(args)
    parameter_list = 'Centralized-data-{:}-type-{:}-seed-{:}'.format(args.file, args.type, args.seed)
    log_id = args.directory + parameter_list

    # train = (xc, xp, y), test = (xc, xp, y). Elements of test are dicts
    train, val, test = process_centralized(args, data)
    print(test[0].shape, test[2].shape)

    train_data = list(zip(*train))  # I am not quit sure the * operation
    test_data = list(zip(*test))

    train_loader = DataLoader(train_data, shuffle=False, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=args.batch_size)

    device = 'cuda' if args.gpu else 'cpu'

    model = LSTM(args).to(device)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=args.momentum)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    train_loss = []
    train_truth, train_y_hat = [], []
    results = []
    for epoch in range(50):
        model.train()
        batch_loss = []
        for idx, (xc, xp, y) in enumerate(train_loader):
            xc, xp = xc.float().to(device), xp.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()

            pred = model(xc, xp)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            if idx % 30 == 0:
                print('Epoch {:} [{:}/{:} ({:.2f}%)]\t Loss: {:.4f}'.format(epoch, idx, len(train_loader),
                                                                            idx/len(train_loader)*100, loss.item()))
            batch_loss.append(loss.item())

        avg_batch = sum(batch_loss)/len(batch_loss)
        print('Epoch {:}, Avg loss {:.4f}'.format(epoch, avg_batch))
        train_loss.append(avg_batch)

        model.eval()
        val_loss = []
        truth, pred = [], []
        for idx, (xc, xp, y) in enumerate(test_loader):
            xc, xp = xc.float().to(device), xp.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()
            y_hat = model(xc, xp)

            loss = criterion(y_hat, y)
            val_loss.append(loss.item())
            truth.append(y.detach().cpu())
            pred.append(y_hat.detach().cpu())

        avg_loss_test = sum(val_loss) / len(val_loss)
        truth_arr = np.concatenate(truth).reshape((-1, args.test_days * 24)).T
        pred_arr = np.concatenate(pred).reshape((-1, args.test_days * 24)).T
        val_mse = metrics.mean_squared_error(truth_arr.ravel(), pred_arr.ravel())
        val_mae = metrics.mean_absolute_error(truth_arr.ravel(), pred_arr.ravel())

        results.append((avg_batch, avg_loss_test, val_mse, val_mae))

    df_results = pd.DataFrame(results, columns=['train_loss', 'val_loss', 'val_mse', 'val_mae'])
    df_results.to_csv('d:/traffic_centralized_guoqing.csv', index=False)

    model.eval()
    val_loss = []
    truth, pred = [], []
    for idx, (xc, xp, y) in enumerate(test_loader):
        xc, xp = xc.float().to(device), xp.float().to(device)
        y = y.float().to(device)

        optimizer.zero_grad()
        y_hat = model(xc, xp)

        loss = criterion(y_hat, y)
        val_loss.append(loss.item())
        truth.append(y.detach().cpu())
        pred.append(y_hat.detach().cpu())

    truth_arr = np.concatenate(truth).reshape((-1, args.test_days*24)).T
    pred_arr = np.concatenate(pred).reshape((-1, args.test_days*24)).T

    print('Test MSE: {:.4f}'.format(metrics.mean_squared_error(truth_arr.ravel(), pred_arr.ravel())))
    print('Test MAE: {:.4f}'.format(metrics.mean_absolute_error(truth_arr.ravel(), pred_arr.ravel())))
