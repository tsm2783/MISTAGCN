from pytictoc import TicToc
from datetime import datetime
import pandas as pd
import numpy as np
import sqlite3
import mxnet as mx
from mxnet import ndarray as nd

from share import dates, Tr, Td, Tw, Tp

# use stop watch
t = TicToc()  # create instance of class
t.tic()  # Start timer

# merge flow and weather data to get data samples
path0 = 'data/Haikou/traffic-flows.db'
path1 = 'data/Haikou/weather-air-condition-Haikou-2017.xlsx'
path = 'data/Haikou/data-samples'
t.toc('gen data samples, merge traffic flows and weather conditions')

# read trafffic flow_out and flow_in
con = sqlite3.connect(path0)
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

t.toc('gen data samples, read database successfully')

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

# read weather and air condition
df_weather_air = pd.read_excel(path1, sheet_name='Sheet1')
df_weather_air = df_weather_air.iloc[:, 1:]
weather_air = df_weather_air.to_numpy()

# merge information
n_time_slices, N = flow_out.shape  # n_time_slices - number of total time points, N - number of areas (nodes in graph)
F = df_weather_air.shape[1] + 2  # number of attributes for each region, flow_out, flow_in, weather and air conditions
data = np.zeros((N, F, n_time_slices))
data[:, 0, :] = flow_out.transpose()
data[:, 1, :] = flow_in.transpose()
for i in range(N):
    for j in range(2, F):
        for k in range(n_time_slices):
            data[i, j, k] = weather_air[k // 24, j - 2]

t.toc('gen data samples, merging weather air information successfully')

# prepare inputs Xs and the target flow_out Ys
n_sample = n_time_slices - Tp - Tw * 7 * 24  # the last Tp time slices are reserved for Yp
assert n_sample > 0, 'The number of samples shoould be greater than 0, i.e. n_time_slices-Tp-Tw*7*24 > 0.'
data = nd.array(data)

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

# write the data to file
nd.save(path, [Xr_sample, Xd_sample, Xw_sample, Yp_sample, regions])
t.toc('gen data samples, saving data sucessfully')
