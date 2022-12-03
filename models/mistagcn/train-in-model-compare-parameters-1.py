import random
import os
from time import time
from unittest import result
import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import autograd, nd
from mxnet.gluon import Trainer, nn
from mxnet.gluon.data.dataloader import DataLoader
from mxnet.gluon.data.dataset import ArrayDataset
from mxnet.metric import MAE, RMSE
from pytictoc import TicToc

from mistagcn import MISTAGCN
from share import batch_size, devices, epochs, threshold, runs, K, Tr, Td, Tw, Tp

scenario = 'YellowTrip'  # Haikou, YellowTrip

# runs = 3
# epochs = 20

data_file = './data/' + scenario + '/data'
acg_file = './data/' + scenario + '/acg'
adg_file = './data/' + scenario + '/adg'
aig_file = './data/' + scenario + '/aig'

data, = nd.load(data_file)
ACG, = nd.load(acg_file)
ADG, = nd.load(adg_file)
AIG, = nd.load(aig_file)

result_dir ='./results-in-model/effect-parameters/' + scenario


def gen_data_samples(data, Tr=Tr, Td=Td, Tw=Tw, Tp=Tp):
    N, F, n_time_slices = data.shape

    n_sample = n_time_slices - Tp - Tw * 7 * 24  #the first Tw * 7 * 24 time slices are reserved for weekly input, the last Tp time slices are reserved for target Yp
    assert n_sample > 0, 'The number of samples shoould be greater than 0, i.e. n_time_slices-Tp-Tw*7*24 > 0.'

    Yp_sample = nd.zeros((n_sample, N, Tp))
    Xr_sample = nd.zeros((n_sample, N, F, Tr))  # the primary Tw*7*24 time slices are reserved for Xw
    Xd_sample = nd.zeros((n_sample, N, F, Td * Tp))
    Xw_sample = nd.zeros((n_sample, N, F, Tw * Tp))
    # month_sample = nd.array(month_sample[Tw*7*24 : n_time_slices-Tp])
    for k in range(n_sample):
        Yp_sample[k] = data[:, 0, k + Tw * 7 * 24: k + Tw * 7 * 24 + Tp]
        Xr_sample[k] = data[:, :, k + Tw * 7 * 24 - Tr: k + Tw * 7 * 24]
        for k1 in range(Td):
            Xd_sample[k, :, :, k1 * Tp:k1 * Tp + Tp] = data[:, :, k + Tw *
                                                            7 * 24 - (Td - k1) * 24: k + Tw * 7 * 24 - (Td - k1) * 24 + Tp]
        for k1 in range(Tw):
            Xw_sample[k, :, :, k1 * Tp:k1 * Tp + Tp] = data[:, :, k + Tw * 7 *
                                                            24 - (Tw - k1) * 7 * 24: k + Tw * 7 * 24 - (Tw - k1) * 7 * 24 + Tp]

    return Xr_sample, Xd_sample, Xw_sample, Yp_sample


def train(run, runs, Xr_sample, Xd_sample, Xw_sample, Yp_sample):
    n_sample, N, F, Tr = Xr_sample.shape
    Tp = Yp_sample.shape[-1]
    Td = Xd_sample.shape[-1] // Tp
    Tw = Xw_sample.shape[-1] // Tp

    # split data index to train, validate, test
    idx = list(range(n_sample))
    idx_train = random.sample(idx, int(0.8 * len(idx)))
    idx = list(set(idx) - set(idx_train))  # exclude idx_train
    idx_validate = random.sample(idx, int(0.5 * len(idx)))
    idx_test = list(set(idx) - set(idx_validate))  # exclude idx_train

    # caculate number of train, validate, test samples
    n_train = len(idx_train)
    n_test = len(idx_test)
    # convert to batch load version
    data_train = ArrayDataset(Xr_sample[idx_train], Xd_sample[idx_train], Xw_sample[idx_train], Yp_sample[idx_train])
    data_validate = ArrayDataset(Xr_sample[idx_validate], Xd_sample[idx_validate],
                                 Xw_sample[idx_validate], Yp_sample[idx_validate])
    data_test = ArrayDataset(Xr_sample[idx_test], Xd_sample[idx_test], Xw_sample[idx_test], Yp_sample[idx_test])
    loader_train = DataLoader(data_train, batch_size=batch_size)
    loader_validate = DataLoader(data_validate, batch_size=batch_size)
    loader_test = DataLoader(data_test, batch_size=batch_size)

    # construct the model, compare

    net = MISTAGCN(ACG, ADG, AIG, K, N, F, Tr, Td, Tw, Tp)

    net.initialize(ctx=devices, force_reinit=True)
    loss_fun = mx.gluon.loss.L2Loss()
    trainer = Trainer(net.collect_params(), 'adam')

    # train and validate the model
    train_records = []
    validate_records = []
    for e in range(epochs):
        start = time()
        train_loss_acc = 0
        for xr, xd, xw, y in loader_train:
            # split batch and load into corresponding devices
            xr_list = mx.gluon.utils.split_and_load(xr, devices, even_split=False)
            xd_list = mx.gluon.utils.split_and_load(xd, devices, even_split=False)
            xw_list = mx.gluon.utils.split_and_load(xw, devices, even_split=False)
            y_list = mx.gluon.utils.split_and_load(y, devices, even_split=False)
            with autograd.record():
                y_hat_list = [net(xr1, xd1, xw1) for xr1, xd1, xw1, in zip(xr_list, xd_list, xw_list, )]
                losses = [loss_fun(y_hat, y1) for y_hat, y1 in zip(y_hat_list, y_list)]
            for loss in losses:
                loss.backward()
            trainer.step(batch_size)
            # sum losses over all devices
            train_loss_acc += sum([loss.sum().asscalar() for loss in losses])
        train_loss_mean = train_loss_acc * 2 / n_train  # mse
        train_rmse = np.sqrt(train_loss_mean)
        train_records.append([e, train_rmse])
        end = time()
        print(
            f'mistagcn, run {run+1}/{runs}, train, epoch {e+1}/{epochs}, duration {end-start}s, average train loss {train_loss_mean}')

        mae = MAE()
        rmse = RMSE()
        for xr, xd, xw, y in loader_validate:
            xr, xd, xw, y = xr.as_in_context(devices[0]), xd.as_in_context(
                devices[0]), xw.as_in_context(devices[0]), y.as_in_context(devices[0])
            label = nd.zeros_like(y)
            y_hat = net(xr, xd, xw)
            pred = nd.ceil(y_hat - y - threshold)
            mae.update(label, pred)
            rmse.update(label, pred)

        validate_records.append([e, mae.get()[1], rmse.get()[1]])
        mae.reset()
        rmse.reset()

    # test the model
    start = time()
    test_loss_acc = 0
    for xr, xd, xw, y in loader_test:
        xr, xd, xw, y = xr.as_in_context(devices[0]), xd.as_in_context(
            devices[0]), xw.as_in_context(devices[0]), y.as_in_context(devices[0])
        y_hat = net(xr, xd, xw)
        loss = loss_fun(y_hat, y)
        test_loss_acc += nd.sum(loss).asscalar()
    test_loss_mean = test_loss_acc * 2 / n_test
    test_rmse = np.sqrt(test_loss_mean)
    end = time()
    print(f'mistagcn, run {run+1}/{runs}, test, duration {end-start}s, average test loss {test_loss_mean}')

    # show gradients of parameters for checking convergence
    # for name, param in net.collect_params().items():
    #     print(name)
    #     print(param.grad())

    train_records = np.array(train_records)
    validate_records = np.array(validate_records)
    test_records = np.array([[test_rmse]])

    return train_records, validate_records, test_records


def main():
    # use stop watch
    t = TicToc()  # create instance of class
    t.tic()  # Start timer

    tr_min, tr_max = 5, 12
    td_min, td_max = 1, 5
    tw_min, tw_max = 1, 4
    tp_min, tp_max = 1, 5

    # recent input change and compare
    # run the trainning process for runs times, calculate the mean
    records_recent = []
    for tr in range(tr_min, tr_max+1):
        Xr_sample, Xd_sample, Xw_sample, Yp_sample = gen_data_samples(data, tr, Td, Tw, Tp)
        for run in range(runs):
            random.seed(run)
            train_r, val_r, test_r = train(run, runs, Xr_sample, Xd_sample, Xw_sample, Yp_sample)
            if run == 0:
                train_records, validate_records, test_records = train_r, val_r, test_r
            else:
                # train_records += train_r
                # validate_records += val_r
                test_records += test_r
        # train_records /= runs
        # validate_records /= runs
        test_records /= runs
        records_recent.append([tr, test_records[0,0]])
    # write out records
    result_file = result_dir + '/recent-input-change-compare-results.xlsx'
    with pd.ExcelWriter(result_file) as writer:
        records_recent = np.array(records_recent)
        df_recent = pd.DataFrame(records_recent, columns=['Tr', scenario])
        df_recent.to_excel(writer, sheet_name='test_records')
    t.toc('mistagcn, recent input change and compare, trainning finished')

    # daily input change and compare
    # run the trainning process for runs times, calculate the mean
    records_daily = []
    for td in range(td_min, td_max+1):
        Xr_sample, Xd_sample, Xw_sample, Yp_sample = gen_data_samples(data, Tr, td, Tw, Tp)
        for run in range(runs):
            random.seed(run)
            train_r, val_r, test_r = train(run, runs, Xr_sample, Xd_sample, Xw_sample, Yp_sample)
            if run == 0:
                train_records, validate_records, test_records = train_r, val_r, test_r
            else:
                # train_records += train_r
                # validate_records += val_r
                test_records += test_r
        # train_records /= runs
        # validate_records /= runs
        test_records /= runs
        records_daily.append([td, test_records[0,0]])
    # write out records
    result_file = result_dir + '/daily-input-change-compare-results.xlsx'
    with pd.ExcelWriter(result_file) as writer:
        records_daily = np.array(records_daily)
        df_daily = pd.DataFrame(records_daily, columns=['Td', scenario])
        df_daily.to_excel(writer, sheet_name='test_records')
    t.toc('mistagcn, daily input change and compare, trainning finished')

    # weekly input change and compare
    # run the trainning process for runs times, calculate the mean
    records_weekly = []
    for tw in range(tw_min, tw_max+1):
        Xr_sample, Xd_sample, Xw_sample, Yp_sample = gen_data_samples(data, Tr, Td, tw, Tp)
        for run in range(runs):
            random.seed(run)
            train_r, val_r, test_r = train(run, runs, Xr_sample, Xd_sample, Xw_sample, Yp_sample)
            if run == 0:
                train_records, validate_records, test_records = train_r, val_r, test_r
            else:
                # train_records += train_r
                # validate_records += val_r
                test_records += test_r
        # train_records /= runs
        # validate_records /= runs
        test_records /= runs
        records_weekly.append([tw, test_records[0,0]])
    # write out records
    result_file = result_dir + '/weekly-input-change-compare-results.xlsx'
    with pd.ExcelWriter(result_file) as writer:
        records_weekly = np.array(records_weekly)
        df_weekly = pd.DataFrame(records_weekly, columns=['Tw', scenario])
        df_weekly.to_excel(writer, sheet_name='test_records')
    t.toc('mistagcn, weekly input change and compare, trainning finished')

    # prediction length change and compare
    # run the trainning process for runs times, calculate the mean
    records_predict_len = []
    for tp in range(tp_min, tp_max+1):
        Xr_sample, Xd_sample, Xw_sample, Yp_sample = gen_data_samples(data, Tr, Td, Tw, tp)
        for run in range(runs):
            random.seed(run)
            train_r, val_r, test_r = train(run, runs, Xr_sample, Xd_sample, Xw_sample, Yp_sample)
            if run == 0:
                train_records, validate_records, test_records = train_r, val_r, test_r
            else:
                # train_records += train_r
                # validate_records += val_r
                test_records += test_r
        # train_records /= runs
        # validate_records /= runs
        test_records /= runs
        records_predict_len.append([tp, test_records[0,0]])
    # write out records
    result_file = result_dir + '/prediction-length-change-compare-results.xlsx'
    with pd.ExcelWriter(result_file) as writer:
        records_predict_len = np.array(records_predict_len)
        df_predict_len = pd.DataFrame(records_predict_len, columns=['Tp', scenario])
        df_predict_len.to_excel(writer, sheet_name='test_records')
    t.toc('mistagcn, prediction length change and compare, trainning finished')


if __name__ == '__main__':
    main()
