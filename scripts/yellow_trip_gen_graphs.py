from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from glob import glob

import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import ndarray as nd
from pytictoc import TicToc

from share import Td, Tp, Tr, Tw
from share import max

# use stop watch
t = TicToc()  # create instance of class
t.tic()  # Start timer

dates = ["2022-01-01 00:00:00", "2022-03-01 01:00:00"]
time_slices = pd.date_range(start=dates[0], end=dates[1], freq='1H')
n_time_slices = time_slices.size - 1


def read_file(file):
    # return pd.read_parquet(file)
    return pd.read_parquet(file, columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance', 'PULocationID', 'DOLocationID'])


files = glob("data/YellowTrip/*.parquet")
with ThreadPoolExecutor(8) as pool:
    df = pd.concat(pool.map(read_file, files))

# print(df.describe())
# print(df.head())
# print(df.columns)
# print(df.info())
# print(df.nunique())  #number of values for each column

regions = pd.concat([df['PULocationID'], df['DOLocationID']])
regions = regions.unique()
regions = np.sort(regions)
regions = [_ for _ in regions]  # convert to list
n_regions = len(regions)

# aggregate records at the interval of hours
flow_out = nd.zeros((n_time_slices, n_regions))
flow_in = nd.zeros((n_time_slices, n_regions))

ADG = nd.zeros((n_regions, n_regions))
ADG_count = nd.zeros((n_regions, n_regions))
AIG = nd.zeros((n_regions, n_regions))

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

    ADG[idx_pickup_region, idx_dropoff_region] += row['trip_distance']
    ADG_count[idx_pickup_region, idx_dropoff_region] += 1
    AIG[idx_pickup_region, idx_dropoff_region] += 1

    index1 = index + 1
    if index1 % 10000 == 0:
        t.toc('aggregation, processed 10000 records')
        # break

data = nd.stack(flow_out, flow_in, axis=2)  # n_time_slices, n_regions, n_featurs
n_time_slices, N, F = data.shape

ADG = ADG + nd.transpose(ADG)
ADG_count = max(ADG_count, nd.ones((n_regions, n_regions)))
ADG_count = ADG_count + nd.transpose(ADG_count)
ADG = ADG / ADG_count
ADG = nd.softmax(ADG, axis=1)

AIG = AIG + nd.transpose(AIG)
AIG = nd.softmax(AIG, axis=1)

ACG = nd.sum(data, axis=2)  # np ndarray
ACG = pd.DataFrame(ACG.asnumpy())  # data frame
ACG = ACG.corr(method='pearson')  # np ndarray
ACG = ACG.fillna(value=-1)  # substitue 'nan' to '-1'
ACG = nd.array(ACG)
ACG = nd.softmax(ACG, axis=1)

nd.save('data/YellowTrip/acg', ACG)
nd.save('data/YellowTrip/adg', ADG)
nd.save('data/YellowTrip/aig', AIG)

t.toc('finish')
