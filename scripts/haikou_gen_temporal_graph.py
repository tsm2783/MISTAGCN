from pytictoc import TicToc
from datetime import datetime
import pandas as pd
import numpy as np
import sqlite3
import mxnet as mx
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

# read trafffic flow_out and flow_in
con = sqlite3.connect('data/Haikou/traffic-flows.db')
query = f'''select * from Flow_out
            where time_slices between '{dates[0]}' and '{dates[1]}'
            order by time_slices
        '''
df_flow_out = pd.read_sql(query, con)
query = f'''select * from Flow_in
            where time_slices between '{dates[0]}' and '{dates[1]}'
            order by time_slices
        '''
df_flow_in = pd.read_sql(query, con)
# close the connection
con.close()

t.toc('gen temporal graph, read database successfully')

# caluculate the corresponding 'month' of each record, which will be used for ajacency matrix selection
fmt = '%Y-%m-%d %H:%M:%S'
# s_month_sample = pd.to_datetime(df_flow_out['time_slices'], format=fmt).dt.month #a Series
# month_sample = s_month_sample.values
# exclude the first column and the last column
df_flow_out.pop('index')
df_flow_out.pop('time_slices')
df_flow_in.pop('index')
df_flow_in.pop('time_slices')

# calculate the regions where flows are recorded
regions = df_flow_out.columns
regions = [_[6:] for _ in regions]
regions = pd.to_numeric(regions)
regions = regions.tolist()
regions = nd.array(regions)

# convert flows to ndarray
flow_out = df_flow_out.to_numpy()  # flow_out has more than one clumn than flow_in, the 'month'
flow_in = df_flow_in.to_numpy()

# merge information
data = np.stack([flow_out, flow_in], axis=-1)

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
nd.save('data/Haikou/adj-temporal', [adj1])

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
nd.save('data/Haikou/adj-temporal-sparse', [w_adj])

t.toc('gen temporal graph, adjacency matrix of temporal graph calculated')