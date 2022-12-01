from calendar import month
import mxnet as mx
import mxnet.ndarray as nd
import os
import numpy as np
from numpy import ndarray
from pytictoc import TicToc
import pandas as pd

# prefer gpu(s) to cpu(s), model is dispatched to all gpus (or cpus)
# data is stored in gpu(0) (or cpu(0)), when needed (eg. training), it should be splited and sent (or copied) to other devices.
if mx.context.num_gpus() > 0:
    devices = [mx.gpu(i) for i in range(mx.context.num_gpus())]  # mxnet
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    os.environ["MXNET_CUDNN_LIB_CHECKING"] = "0"
else:
    devices = [mx.cpu()]  # mxnet

# constants
# Haikou city is in area of latitude 19.5167~20.0667 degree and longitude 110.1167~110.7000 degree. The city area is divided into M_lat x N_lng regions, of equal size.
# The coordinates of region i is (the point of the left corner of region i)
# (lng_min_i, lat_min_i) = (longitude_min + (i // N) * longitude_span / N, latitude_min + (i % N) * latitude_span / M)
# The longitude_span and latitude_span of region i is
# (lng_span_i, lat_span_i) = (longitude_span/N, latitude_span/M)
longitude_min = 110.1167
longitude_span = 110.7000 - 110.1167
latitude_min = 19.5167
latitude_span = 20.0667 - 19.5167
M_lat = 10
N_lng = 20
R = 6371  # radius of the earth (km)

# key parameters for trainning
Tp = 3  # Tp < Tr
Tr = 7
Td = 3
Tw = 2
K = 3
percent_train = 0.8
batch_size = 200
threshold = 0.5
eps = 0.0001
epochs = 200
epochs1 = 40
runs = 10

# date-related data
dates = ["2017-08-01 00:00:00", "2017-10-31 23:00:00"]
time_slices = pd.date_range(start=dates[0], end=dates[1], freq='1H')
months = pd.date_range(start=dates[0], end=dates[1], freq='1M')
months = [_.month for _ in months]


def get_max_eigenvalue(A: ndarray) -> float:
    '''
    Get the maximum eigenvalue of a 2d square matrix (A), using power method.
    '''
    assert A.ndim == 2 and A.shape[0] == A.shape[1], 'A should be a 2d squary matrix.'
    n = A.shape[0]
    x = nd.ones(n)
    for i in range(100):
        x = nd.dot(A, x)
        lambda_max = nd.max(nd.abs(x))
        x = x / lambda_max
    return lambda_max


def dot(a: ndarray, b: ndarray) -> ndarray:
    '''
    Numpy style dot operation on two mx.ndarray a and b. eg. np.dot(a, b)[2,3,2,1,2,2] = sum(a[2,3,2,:] * b[1,2,:,2])
    '''
    dim_b = len(b.shape)
    if dim_b > 2:
        b = nd.swapaxes(b, 0, -2)
        out = nd.dot(a, b)
        out = nd.swapaxes(out, -dim_b + 1, -2)
    else:
        out = nd.dot(a, b)
    return out


def merge_list_mx_ndarray(x):
    '''
    Merge a list of arrays to a single array, where
        x: [arr1, ..., arrN]
        out: out[0]=arr1, ..., out[N-1]=arrN
    '''
    x = [nd.expand_dims(_, axis=0) for _ in x]
    out = x[0]
    for i in range(1, len(x)):
        out = nd.concat(out, x[i], dim=0)
    return out


def merge_list_np_ndarray(x):
    '''
    Merge a list of arrays to a single array, where
        x: [arr1, ..., arrN]
        out: out[0]=arr1, ..., out[N-1]=arrN
    '''
    x = [np.expand_dims(_, axis=0) for _ in x]
    out = x[0]
    for i in range(1, len(x)):
        out = np.concatenate((out, x[i]), axis=0)
    return out


def max(x, y, axis=-1):
    assert x.shape == y.shape
    xy = nd.stack(x, y, axis=-1)
    m = nd.max(xy, axis=axis)
    return m


if __name__ == '__main__':
    x = nd.random.uniform(shape=(3, 5))
    y = nd.ones_like(x) / 2
    print(x, y, max(x, y))
