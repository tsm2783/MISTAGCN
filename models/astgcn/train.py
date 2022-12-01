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

from astgcn import ASTGCN
from share import Tr, Td, Tw, Tp, K, batch_size, epochs, devices, threshold, get_max_eigenvalue, runs

scenario = 'YellowTrip'  # Haikou, YellowTrip
# runs = 1
# epochs = 5

data_sample_file = './data/' + scenario + '/data-samples'
adg_file = './data/' + scenario + '/adg'
result_file = 'results/' + scenario + '/astgcn-results.xlsx'

Xr_sample, Xd_sample, Xw_sample, Yp_sample, regions = nd.load(data_sample_file)
ADG, = nd.load(adg_file)


def get_backbones():
    r'''
    Prepare parameters of the parallel models, where
        adj_filename indicates the file where adjacency matrix is stored,\
        K is the order of polynomials,\
        Tr is the number of time slices in recent input,\
        Td is the number of days in daily input,\
        Tw is the number of weeks in weekly input.
    '''
    # load ajacency matrix
    A = ADG
    # calculate Chebyshev polynomial
    N = len(A)
    D = nd.sum(A, axis=1)
    eps = 0.0001
    D_12 = 1 / (nd.sqrt(D) + eps)
    L = nd.eye(N, ctx=A.ctx) - (D_12 * A) * D_12  # element wise multiply
    lambda_max = get_max_eigenvalue(L)
    L_tilde = 2 / lambda_max * L - nd.eye(N, ctx=A.ctx)
    cheb_polynomial = [nd.eye(N, ctx=A.ctx), L_tilde]
    for k in range(2, K):
        cheb_polynomial.append(
            2 * L_tilde * cheb_polynomial[-1] - cheb_polynomial[-2])
    # convert list to array
    cheb_polynomial = [nd.expand_dims(_, axis=0) for _ in cheb_polynomial]
    temp = cheb_polynomial[0]
    for i in range(1, K):
        temp = nd.concat(temp, cheb_polynomial[i], dim=0)
    cheb_polynomial = temp

    backbones1 = [
        {
            "K": K,
            "n_chev_filters": 32,
            "n_time_filters": 32,
            "time_conv_strides": Tr,
            "cheb_polynomial": cheb_polynomial
        },
        {
            "K": K,
            "n_chev_filters": 32,
            "n_time_filters": 32,
            "time_conv_strides": 1,
            "cheb_polynomial": cheb_polynomial
        }
    ]

    backbones2 = [
        {
            "K": K,
            "n_chev_filters": 32,
            "n_time_filters": 32,
            "time_conv_strides": Td,
            "cheb_polynomial": cheb_polynomial
        },
        {
            "K": K,
            "n_chev_filters": 32,
            "n_time_filters": 32,
            "time_conv_strides": 1,
            "cheb_polynomial": cheb_polynomial
        }
    ]

    backbones3 = [
        {
            "K": K,
            "n_chev_filters": 32,
            "n_time_filters": 32,
            "time_conv_strides": Tw,
            "cheb_polynomial": cheb_polynomial
        },
        {
            "K": K,
            "n_chev_filters": 32,
            "n_time_filters": 32,
            "time_conv_strides": 1,
            "cheb_polynomial": cheb_polynomial
        }
    ]

    all_backbones = [
        backbones1,
        backbones2,
        backbones3
    ]

    return all_backbones


def train(run, runs):
    n_sample = Xr_sample.shape[0]
    # split data index to train, validate, test
    idx = list(range(n_sample))
    idx_train = random.sample(idx, int(0.8 * len(idx)))
    idx = list(set(idx) - set(idx_train))  # exclude idx_train
    idx_validate = random.sample(idx, int(0.5 * len(idx)))
    idx_test = list(set(idx) - set(idx_validate))  # exclude idx_train
    # caculate number of train, validate, test samples
    n_train = len(idx_train)
    n_test = len(idx_test)
    # comvert to batch load version
    data_train = ArrayDataset(
        Xr_sample[idx_train], Xd_sample[idx_train], Xw_sample[idx_train], Yp_sample[idx_train])
    data_validate = ArrayDataset(
        Xr_sample[idx_validate], Xd_sample[idx_validate], Xw_sample[idx_validate], Yp_sample[idx_validate])
    data_test = ArrayDataset(
        Xr_sample[idx_test], Xd_sample[idx_test], Xw_sample[idx_test], Yp_sample[idx_test])
    loader_train = DataLoader(data_train, batch_size=batch_size)
    loader_validate = DataLoader(data_validate, batch_size=batch_size)
    loader_test = DataLoader(data_test, batch_size=batch_size)

    # get model's structure and prepare the net
    all_backbones = get_backbones()
    net = ASTGCN(Tp, all_backbones)
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
            xr_list = mx.gluon.utils.split_and_load(
                xr, devices, even_split=False)
            xd_list = mx.gluon.utils.split_and_load(
                xd, devices, even_split=False)
            xw_list = mx.gluon.utils.split_and_load(
                xw, devices, even_split=False)
            y_list = mx.gluon.utils.split_and_load(
                y, devices, even_split=False)
            # run forward and backward on each device, mxnet will automatically run them in parallel
            with autograd.record():
                y_hat_list = [net([xr1, xd1, xw1])
                              for xr1, xd1, xw1 in zip(xr_list, xd_list, xw_list)]
                losses = [loss_fun(y_hat, y1)
                          for y_hat, y1 in zip(y_hat_list, y_list)]
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
            f'astgcn, run {run+1}/{runs}, train, epoch {e+1}/{epochs}, duration {end-start}s, average train loss {train_loss_mean}')

        mae = MAE()
        rmse = RMSE()
        for xr, xd, xw, y in loader_validate:
            xr = xr.copyto(devices[0])
            xd = xd.copyto(devices[0])
            xw = xw.copyto(devices[0])
            y = y.copyto(devices[0])
            label = nd.zeros_like(y)
            y_hat = net([xr, xd, xw])
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
        xr = xr.copyto(devices[0])
        xd = xd.copyto(devices[0])
        xw = xw.copyto(devices[0])
        y = y.copyto(devices[0])
        y_hat = net([xr, xd, xw])
        loss = loss_fun(y_hat, y)
        test_loss_acc += nd.sum(loss).asscalar()
    test_loss_mean = test_loss_acc * 2 / n_test
    test_rmse = np.sqrt(test_loss_mean)
    end = time()
    print(f'astgcn, run {run+1}/{runs}, test, duration {end-start}s, average test loss {test_loss_mean}')

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

    t.toc('astgcn, trainning finished')


if __name__ == '__main__':
    main()
