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

from vae import VAE
from mstif_net import MSTIF_NET
from share import batch_size, devices, epochs, epochs1, Tp, threshold, runs

scenario = 'YellowTrip'  # Haikou, YellowTrip
# runs = 1
# epochs = 5

data_sample_file = './data/' + scenario + '/data-samples'
acg_file = './data/' + scenario + '/acg'
adg_file = './data/' + scenario + '/adg'
aig_file = './data/' + scenario + '/aig'
result_file = 'results/' + scenario + '/MSTIF-Net-results.xlsx'

Xr_sample, Xd_sample, Xw_sample, Yp_sample, regions = nd.load(data_sample_file)
ACG, = nd.load(acg_file)
ADG, = nd.load(adg_file)
AIG, = nd.load(aig_file)


def train(run, runs):
    X = Xr_sample[:, :, 0, :]
    Y = Yp_sample
    n_sample, N, T = X.shape
    # X_weather_air = Xr_sample[:,:,1:,:]

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
    data_train = ArrayDataset(X[idx_train], Y[idx_train])
    data_validate = ArrayDataset(X[idx_validate], Y[idx_validate])
    data_test = ArrayDataset(X[idx_test], Y[idx_test])
    loader_train = DataLoader(data_train, batch_size=batch_size)
    loader_validate = DataLoader(data_validate, batch_size=batch_size)
    loader_test = DataLoader(data_test, batch_size=batch_size)

    # VAE model (stage 1)

    vae = VAE(n_encoder_filter_list=[100, 50], n_latent=20, n_decoder_filter_list=[50, 100])
    vae.initialize(ctx=devices, force_reinit=True)
    loss_fun = mx.gluon.loss.L2Loss()
    trainer = mx.gluon.Trainer(vae.collect_params(), optimizer='adam')
    for e in range(epochs1):
        start = time()
        train_loss_acc = 0
        for x, y in loader_train:
            # split batch and load into corresponding devices
            x_list = mx.gluon.utils.split_and_load(x, devices, even_split=False)
            with autograd.record():
                z_out_list = [vae(x1) for x1 in x_list]  # a list of tuple (z, out)
                losses = [loss_fun(z_out[1], x1) for z_out, x1 in zip(z_out_list, x_list)]
            for loss in losses:
                loss.backward()
            trainer.step(batch_size)
            # sum losses over all devices
            train_loss_acc += sum([loss.sum().asscalar() for loss in losses])
        train_loss_mean = train_loss_acc * 2 / n_train  # mse
        train_rmse = np.sqrt(train_loss_mean)
        end = time()
        print(
            f'mstif-net, train vae, run {run+1}/{runs}, train, epoch {e+1}/{epochs1}, duration {end-start}s, average train loss {train_loss_mean}')

    # MISTF-NET model (stage 2)

    net = MSTIF_NET(ACG, ADG, AIG, n_filter1=50, n_filter2=50, n_filter=20, n_fusion_out=50, out_dim=20,
                    n_hidden_size_encoder=64, n_layers_enconder=2, n_hidden_size_decoder=64, n_layers_deconder=2, Tp=Tp)
    net.initialize(ctx=devices, force_reinit=True)
    loss_fun = mx.gluon.loss.L2Loss()
    trainer = mx.gluon.Trainer(net.collect_params(), optimizer='adam')

    # train and validate the model
    train_records = []
    validate_records = []
    for e in range(epochs):
        start = time()
        train_loss_acc = 0
        for x, y in loader_train:
            # split batch and load into corresponding devices
            x_list = mx.gluon.utils.split_and_load(x, devices, even_split=False)
            y_list = mx.gluon.utils.split_and_load(y, devices, even_split=False)
            z_out_list = [vae(x1) for x1 in x_list]  # a list of tuple (z, out)
            with autograd.record():
                y_hat_list = [net(x1, z_out[0]) for x1, z_out in zip(x_list, z_out_list)]
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
            f'mstif-net, run {run+1}/{runs}, train, epoch {e+1}/{epochs}, duration {end-start}s, average train loss {train_loss_mean}')

        mae = MAE()
        rmse = RMSE()
        for x, y in loader_validate:
            x = x.copyto(devices[0])
            y = y.copyto(devices[0])
            z, out = vae(x)
            label = nd.zeros_like(y)
            y_hat = net(x, z)
            pred = nd.ceil(y_hat - y - threshold)
            mae.update(label, pred)
            rmse.update(label, pred)

        validate_records.append([e, mae.get()[1], rmse.get()[1]])
        mae.reset()
        rmse.reset()

    # test the model
    start = time()
    test_loss_acc = 0
    for x, y in loader_test:
        x = x.copyto(devices[0])
        y = y.copyto(devices[0])
        z, out = vae(x)
        y_hat = net(x, z)
        loss = loss_fun(y_hat, y)
        test_loss_acc += nd.sum(loss).asscalar()
    test_loss_mean = test_loss_acc * 2 / n_test
    test_rmse = np.sqrt(test_loss_mean)
    end = time()
    print(f'mstif-net, run {run+1}/{runs}, test, duration {end-start}s, average test loss {test_loss_mean}')

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

    t.toc('mstif-net, trainning finished')


if __name__ == '__main__':
    main()
