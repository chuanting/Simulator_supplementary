# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: fed_avg_guoqing.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-03-01 (YYYY-MM-DD)
-----------------------------------------------
"""
# -*- coding: utf-8 -*-
"""
-----------------------------------------------
# File: fed_avg_algo.py
# This file is created by Chuanting Zhang
# Email: chuanting.zhang@kaust.edu.sa
# Date: 2020-01-13 (YYYY-MM-DD)
-----------------------------------------------
"""
import numpy as np
import h5py
import tqdm
import copy
import torch
import pandas as pd
import sys
import random

sys.path.append('../../')
from FedMMoE.utils.misc_v2 import args_parser, average_weights
from FedMMoE.utils.misc_v2 import get_data, process_isolated
from FedMMoE.utils.models import LSTM
from FedMMoE.utils.fed_update import LocalUpdate, test_inference
from sklearn import metrics
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data, _, selected_cells, mean, std, _, _ = get_data(args)

    device = 'cuda' if args.gpu else 'cpu'

    parameter_list = 'FedAvg-data-{:}-type-{:}-'.format(args.file, args.type)
    parameter_list += '-frac-{:.2f}-le-{:}-lb-{:}-seed-{:}'.format(args.frac, args.local_epoch,
                                                                   args.local_bs,
                                                                   args.seed)
    log_id = args.directory + parameter_list
    # train = (train_xc, train_xp, train_label), test=(test_xc, test_xp, test_label)
    train, val, test = process_isolated(args, data)

    global_model = LSTM(args).to(device)
    global_model.train()
    # print(global_model)

    global_weights = global_model.state_dict()

    best_val_loss = None
    val_loss = []
    results = []

    m = max(int(args.frac * args.bs), 1)
    cell_idx = random.sample(selected_cells, m)
    print(cell_idx)

    for epoch in range(args.epochs):
        local_weights, local_losses = [], []
        # print(f'\n | Global Training Round: {epoch+1} |\n')
        global_model.train()

        for cell in cell_idx:
            cell_train, cell_test = train[cell], test[cell]
            local_model = LocalUpdate(args, cell_train, cell_test)

            global_model.load_state_dict(global_weights)
            global_model.train()

            w, loss, epoch_loss = local_model.update_weights(model=copy.deepcopy(global_model),
                                                             global_round=epoch)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # Update global model
        avg_loss = sum(local_losses) / len(local_losses)
        global_weights = average_weights(local_weights)

        # validate global model
        val_loss_group, val_loss_avg = [], []
        cell_loss = []
        val_pred, val_truth = {}, {}
        for cell in selected_cells:
            cell_val = test[cell]
            global_model.load_state_dict(global_weights)
            global_model.eval()
            v_loss, _, _pred, _truth = test_inference(args, global_model, cell_val)
            val_loss_group.append(v_loss)
            cell_loss.append(v_loss)
            val_pred[cell] = _pred
            val_truth[cell] = _truth

        avg_val = sum(cell_loss) / len(cell_loss)
        val_loss.append(avg_val)

        val_pred_df = pd.DataFrame.from_dict(val_pred)
        val_truth_df = pd.DataFrame.from_dict(val_truth)
        val_mse = metrics.mean_squared_error(val_pred_df.values.ravel(), val_truth_df.values.ravel())
        val_mae = metrics.mean_absolute_error(val_pred_df.values.ravel(), val_truth_df.values.ravel())
        print('Epoch: {:} val mse: {:.4f} val mae: {:.4f}'.format(epoch, val_mse, val_mae))
        results.append((avg_loss, avg_val, val_mse, val_mae))
        #
        # if not best_val_loss or avg_val < best_val_loss:
        #     torch.save(global_model.state_dict(), log_id + '.pt')
        #     best_val_loss = avg_val
        #
        # # global_model.load_state_dict(global_weights)

    # Test model accuracy
    df_results = pd.DataFrame(results, columns=['train_loss', 'val_loss', 'val_mse', 'val_mae'])
    df_results.to_csv('d:/traffic_fl_guoqing.csv', index=False)
    pred, truth = {}, {}
    test_loss_list = []
    test_mse_list = []
    nrmse = 0.0

    global_model.load_state_dict(global_weights)
    # global_model.load_state_dict(torch.load(log_id + '.pt'))

    for cell in selected_cells:
        cell_test = test[cell]
        test_loss, test_mse, pred[cell], truth[cell] = test_inference(args, global_model, cell_test)

        test_loss_list.append(test_loss)
        test_mse_list.append(test_mse)

    df_pred = pd.DataFrame.from_dict(pred)
    df_truth = pd.DataFrame.from_dict(truth)

    mse = metrics.mean_squared_error(df_pred.values.ravel(), df_truth.values.ravel())
    mae = metrics.mean_absolute_error(df_pred.values.ravel(), df_truth.values.ravel())
    print('FedAvg File: {:} Type: {:} MSE: {:.4f} MAE: {:.4f}'.format(args.file, args.type, mse, mae))
