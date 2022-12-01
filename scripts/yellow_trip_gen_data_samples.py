from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from glob import glob
import os

import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import ndarray as nd
from pytictoc import TicToc

from share import Td, Tp, Tr, Tw

# use stop watch
t = TicToc()  # create instance of class
t.tic()  # Start timer

t.toc('flow aggregation, calculate hourly traffic flows')

dates = ["2022-01-01 00:00:00", "2022-03-01 01:00:00"]
time_slices = pd.date_range(start=dates[0], end=dates[1], freq='1H')
n_time_slices = time_slices.size - 1


def read_file(file):
    # return pd.read_parquet(file)
    return pd.read_parquet(file, columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID'])


files = glob("data/YellowTrip/*.parquet")
n_cpu = os.cpu_count()
with ThreadPoolExecutor(n_cpu) as pool:
    df = pd.concat(pool.map(read_file, files))

# print(df.describe())
# print(df.head())
# print(df.columns)
# print(df.info())
# print(df.nunique())  #number of values for each column
# print(df['passenger_count'])

regions = pd.concat([df['PULocationID'], df['DOLocationID']])
regions = regions.unique()
regions = np.sort(regions)
regions = [_ for _ in regions]  # convert to list
n_regions = len(regions)

# aggregate records at the interval of hours
flow_out = nd.zeros((n_time_slices, n_regions))
flow_in = nd.zeros((n_time_slices, n_regions))

fmt = '%Y-%m-%d %H:%M:%S'
t0 = time_slices[0]  # global start time
t0 = int(t0.timestamp())
t1 = time_slices[-1]  # global end time
t1 = int(t1.timestamp())
for index, row in df.iterrows():
    # pickup_time = datetime.strptime(row['tpep_pickup_datetime'], fmt)
    pickup_time = row['tpep_pickup_datetime']
    pickup_time = int(pickup_time.timestamp())  # seconds
    if pickup_time < t0 or pickup_time >= t1:
        continue
    idx_pickup_time = (pickup_time - t0) // 3600
    idx_pickup_region = regions.index(row['PULocationID'])
    flow_out[idx_pickup_time, idx_pickup_region] += 1

    # dropoff_time = datetime.strptime(row['tpep_dropoff_datetime'], fmt)
    dropoff_time = row['tpep_dropoff_datetime']
    dropoff_time = int(dropoff_time.timestamp())  # seconds
    if dropoff_time < t0 or dropoff_time >= t1:
        continue
    idx_dropoff_time = (dropoff_time - t0) // 3600
    idx_dropoff_region = regions.index(row['DOLocationID'])
    flow_in[idx_dropoff_time, idx_dropoff_region] += 1

    index1 = index + 1
    if index1 % 10000 == 0:
        t.toc('flow aggregation, processed 10000 records')
        # break

data = nd.stack(flow_out, flow_in, axis=2)  # n_time_slices, n_regions, n_featurs
n_time_slices, N, F = data.shape

t.toc('gen data samples')

# prepare inputs Xs and the target flow_out Ys
n_sample = n_time_slices - Tp - Tw * 7 * 24  # the last Tp time slices are reserved for Yp
assert n_sample > 0, 'The number of samples shoould be greater than 0, i.e. n_time_slices-Tp-Tw*7*24 > 0.'
data = nd.array(data)
data = nd.transpose(data, axes=(1, 2, 0))  # N, F, n_time_slices
# data = data / 1000  #convert unit to k

Yp_sample = nd.zeros((n_sample, N, Tp))
Xr_sample = nd.zeros((n_sample, N, F, Tr))  # the primary Tw*7*24 time slices are reserved for Xw
Xd_sample = nd.zeros((n_sample, N, F, Td * Tp))
Xw_sample = nd.zeros((n_sample, N, F, Tw * Tp))
for k in range(n_sample):
    Yp_sample[k] = data[:, 0, k + Tw * 7 * 24: k + Tw * 7 * 24 + Tp]
    Xr_sample[k] = data[:, :, k + Tw * 7 * 24 - Tr: k + Tw * 7 * 24]
    for k1 in range(Td):
        Xd_sample[k, :, :, k1 * Tp: k1 * Tp + Tp] = data[:, :, k + Tw *
                                                         7 * 24 - (Td - k1) * 24: k + Tw * 7 * 24 - (Td - k1) * 24 + Tp]
    for k1 in range(Tw):
        Xw_sample[k, :, :, k1 * Tp: k1 * Tp + Tp] = data[:, :, k + Tw * 7 *
                                                         24 - (Tw - k1) * 7 * 24: k + Tw * 7 * 24 - (Tw - k1) * 7 * 24 + Tp]

t.toc('write data samples')

regions = nd.array(regions)
nd.save('data/YellowTrip/data-samples', [Xr_sample, Xd_sample, Xw_sample, Yp_sample, regions])
