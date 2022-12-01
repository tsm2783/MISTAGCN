from pytictoc import TicToc
from datetime import datetime
import pandas as pd
import numpy as np
from glob import glob
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from mxnet import ndarray as nd

from share import dates, eps

# this script is created for model STFGNN

def normalize(a):
    mu = np.mean(a, axis=1, keepdims=True)
    std = np.std(a, axis=1, keepdims=True)
    return (a-mu) / (std + eps)

def compute_dtw(a, b, order=1, Ts=12, normal=True):
    if normal:
        a=normalize(a)
        b=normalize(b)
    T0=a.shape[1]
    d=np.reshape(a,[-1,1,T0])-np.reshape(b,[-1,T0,1])
    d=np.linalg.norm(d,axis=0,ord=order)
    D=np.zeros([T0,T0])
    for i in range(T0):
        for j in range(max(0,i-Ts),min(T0,i+Ts+1)):
            if (i==0) and (j==0):
                D[i,j]=d[i,j]**order
                continue
            if (i==0):
                D[i,j]=d[i,j]**order+D[i,j-1]
                continue
            if (j==0):
                D[i,j]=d[i,j]**order+D[i-1,j]
                continue
            if (j==i-Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j])
                continue
            if (j==i+Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i,j-1])
                continue
            D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j],D[i,j-1])
    return D[-1,-1]**(1.0/order)


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
flow_out = np.zeros((n_time_slices, n_regions))
flow_in = np.zeros((n_time_slices, n_regions))

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

data = np.stack([flow_out, flow_in], axis=2)  # n_time_slices, n_regions, n_featurs
n_time_slices, N, F = data.shape

t.toc('gen temporal graph, traffic data generated')

n_time_slices, N, F = data.shape  # n_time_slices - number of total time points, N - number of areas (nodes in graph), F - number of features
period = 24  #there are 24 hours in a day
num_dtw = int(n_time_slices/period) * period
data = data[:num_dtw,:,:1].reshape([-1, period, N])
adj = np.zeros(shape=(N, N))
for i in range(N):
    for j in range(i+1,N):
        adj[i,j] = compute_dtw(data[:,:,i], data[:,:,j])

adj = adj + adj.T
adj1 = nd.array(adj)
nd.save('data/YellowTrip/adj-temporal', [adj1])

w_adj = np.zeros([N,N])
# adj = adj+ adj.T
adj_percent = 0.01
top = int(N * adj_percent)
for i in range(N):
    idx = adj[i,:].argsort()[0:top]
    for j in range(top):
        w_adj[i, idx[j]] = 1
for i in range(N):
    for j in range(N):
        if(w_adj[i][j] != w_adj[j][i] and w_adj[i][j] ==0):
            w_adj[i][j] = 1
        if(i==j):
            w_adj[i][j] = 1

w_adj = nd.array(w_adj)
nd.save('data/YellowTrip/adj-temporal-sparse', [w_adj])

t.toc('gen temporal graph, adjacency matrix of temporal graph calculated')