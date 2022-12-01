import random
from time import time
import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import autograd, nd
from mxnet.gluon import Trainer, nn
from mxnet.gluon.data.dataloader import DataLoader
from mxnet.gluon.data.dataset import ArrayDataset
from mxnet.metric import MAE, RMSE
from pytictoc import TicToc

from gman import GMAN
from share import Tr, Tw, Tp, batch_size, devices, epochs, runs, threshold, time_slices
batch_size = batch_size // 2

scenario = 'YellowTrip'  # Haikou, YellowTrip
# runs = 1
# epochs = 5

data_sample_file = './data/' + scenario + '/data-samples'
result_file = 'results/' + scenario + '/GMAN-results.xlsx'

Xr_sample, Xd_sample, Xw_sample, Yp_sample, regions = nd.load(data_sample_file)


def train(run, runs):
    X = Xr_sample[:, :, 0, :]  # (n_sample, N, T)
    X = nd.transpose(X, (0, 2, 1))  # (n_sample, T, N) -- (n_sample, P, N)
    Y = Yp_sample  # (n_sample, N, Tp)
    Y = nd.transpose(Y, (0, 2, 1))  # (n_sample, Tp, N) -- (n_sample, Q, N)
    n_sample, P, N = X.shape
    Q = Y.shape[1]

    # spatial embedding
    f = open('data/' + scenario + '/se.txt', mode='r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, D = int(temp[0]), int(temp[1])
    se = nd.zeros(shape=(N, D))
    for line in lines[1:]:
        temp = line.split(' ')
        index = int(temp[0])
        se[index] = temp[1:]

    # temporal embedding
    day_of_week = np.reshape(time_slices.weekday, newshape=(-1, 1))
    time_of_day = np.reshape(time_slices.hour, newshape=(-1, 1))
    time_array = np.concatenate((day_of_week, time_of_day), axis=-1)
    time_array = nd.array(time_array)
    # convert the data to samples (see 'scripts/gen_data_samples.py')
    te_sample = nd.zeros(shape=(n_sample, P + Q, 2))
    for k in range(n_sample):
        te_sample[k] = time_array[k + Tw * 7 * 24 - P: k + Tw * 7 * 24 + Q, :]

    # split data index to train, validate, test
    idx = list(range(n_sample))
    idx_train = random.sample(idx, int(0.8 * len(idx)))
    idx = list(set(idx) - set(idx_train))  # exclude idx_train
    idx_validate = random.sample(idx, int(0.5 * len(idx)))
    idx_test = list(set(idx) - set(idx_validate))  # exclude idx_train
    # caculate number of train, validate, test samples
    n_train = len(idx_train)
    n_test = len(idx_test)

    # prepare train, validate, test data
    te_train, te_validate, te_test = te_sample[idx_train], te_sample[idx_validate], te_sample[idx_test]
    # traffic records, normalize
    X_train, X_validate, X_test = X[idx_train], X[idx_validate], X[idx_test]
    mean = nd.mean(X_train)
    std = nd.sqrt(nd.sum((X_train - mean)**2) / X_train.size)
    X_train = (X_train - mean) / std
    X_validate = (X_validate - mean) / std
    X_test = (X_test - mean) / std

    # convert to batch loader
    data_train = ArrayDataset(X_train, te_train, Y[idx_train])
    data_validate = ArrayDataset(X_validate, te_validate, Y[idx_validate])
    data_test = ArrayDataset(X_test, te_test, Y[idx_test])
    loader_train = DataLoader(data_train, batch_size=batch_size)
    loader_validate = DataLoader(data_validate, batch_size=batch_size)
    loader_test = DataLoader(data_test, batch_size=batch_size)

    # construct the model
    net = GMAN(L=2, K=2, d=2, P=P, T=24)
    net.initialize(ctx=devices, force_reinit=True)
    loss_fun = mx.gluon.loss.L2Loss()
    trainer = Trainer(net.collect_params(), 'adam')

    # train and validate the model
    train_records = []
    validate_records = []
    for e in range(epochs):
        start = time()
        train_loss_acc = 0
        for x, te, y in loader_train:
            # split batch and load into corresponding devices
            x_list = mx.gluon.utils.split_and_load(x, devices, even_split=False)
            te_list = mx.gluon.utils.split_and_load(te, devices, even_split=False)
            y_list = mx.gluon.utils.split_and_load(y, devices, even_split=False)
            with autograd.record():
                y_hat_list = [net(x1, se, te1) for x1, te1 in zip(x_list, te_list)]
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
            f'gman, run {run+1}/{runs}, train, epoch {e+1}/{epochs}, duration {end-start}s, average train loss {train_loss_mean}')

        mae = MAE()
        rmse = RMSE()
        for x, te, y in loader_validate:
            x = x.copyto(devices[0])
            te = te.copyto(devices[0])
            y = y.copyto(devices[0])
            label = nd.zeros_like(y)
            y_hat = net(x, se, te)
            pred = nd.ceil(y_hat - y - threshold)
            mae.update(label, pred)
            rmse.update(label, pred)
        validate_records.append([e, mae.get()[1], rmse.get()[1]])
        mae.reset()
        rmse.reset()

    # test the model
    start = time()
    test_loss_acc = 0
    for x, te, y in loader_test:
        x = x.copyto(devices[0])
        te = te.copyto(devices[0])
        y = y.copyto(devices[0])
        y_hat = net(x, se, te)
        # y_hat = y_hat * max
        loss = loss_fun(y_hat, y)
        test_loss_acc += nd.sum(loss).asscalar()
    test_loss_mean = test_loss_acc * 2 / n_test
    test_rmse = np.sqrt(test_loss_mean)
    end = time()
    print(f'gman, run {run+1}/{runs}, test, duration {end-start}s, average test loss {test_loss_mean}')

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

    # run the trainning process for runs times, calculate the mean
    for run in range(runs):
        random.seed(run)
        train_r, val_r, test_r = train(run, runs)
        if run == 0:
            train_records, validate_records, test_records = train_r, val_r, test_r
        else:
            train_records += train_r
            validate_records += val_r
            test_records += test_r
    train_records /= runs
    validate_records /= runs
    test_records /= runs

    # write out records
    with pd.ExcelWriter(result_file) as writer:
        df_train_records = pd.DataFrame(train_records, columns=['Epoch', 'RMSE'])
        df_train_records.to_excel(writer, sheet_name='train_records')
        df_validate_records = pd.DataFrame(validate_records, columns=['Epoch', 'MAE', 'RMSE'])
        df_validate_records.to_excel(writer, sheet_name='validate_records')
        df_test_records = pd.DataFrame(test_records, columns=['RMSE'])
        df_test_records.to_excel(writer, sheet_name='test_records')

    t.toc('gman, trainning finished')


if __name__ == '__main__':
    main()
