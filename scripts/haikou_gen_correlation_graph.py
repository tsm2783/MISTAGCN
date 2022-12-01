from datetime import date
from itertools import count
from share import *
from pytictoc import TicToc
import pandas as pd
from mxnet import ndarray as nd
import numpy as np
import sqlite3

from share import dates, months

# use stop watch
t = TicToc()  # create instance of class
t.tic()  # Start timer

t.toc('gen correlation graph, calculate ACGs')

Xr_sample, Xd_sample, Xw_sample, Yp_sample, regions = nd.load('./data/Haikou/data-samples')
regions = [_ for _ in regions]

# connect database and compute total trafffic flows
con = sqlite3.connect('data/Haikou/traffic-flows.db')
query = f'''select * from Flow_out
            where time_slices between '{dates[0]}' and '{dates[1]}'
            order by time_slices
        '''
df_flow_out = pd.read_sql(query, con)
df_flow_out.pop('index')
query = f'''select * from Flow_in
            where time_slices between '{dates[0]}' and '{dates[1]}'
            order by time_slices
        '''
df_flow_in = pd.read_sql(query, con)
df_flow_in.pop('index')
# close the connection
con.close()

# # process flow out data
# fmt = '%Y-%m-%d %H:%M:%S'
# df_flow_out['time_slices'] = pd.to_datetime(df_flow_out['time_slices'], format=fmt)  # dtype: object->datetime
# df_flow_out['month'] = df_flow_out['time_slices'].map(lambda x: x.month)
# df_flow_in['time_slices'] = pd.to_datetime(df_flow_in['time_slices'], format=fmt)  # dtype: object->datetime
# df_flow_in['month'] = df_flow_in['time_slices'].map(lambda x: x.month)

# compute monthly correlation graph (ajacency matrices)
# n_months = len(months)
# n_regions = len(regions)
# ACGs = np.zeros((n_months, n_regions, n_regions))
# ACG = np.zeros((n_regions, n_regions))
# fmt = '%Y-%m-%d %H:%M:%S'
# for m in months:
#     df_flow_out_m = df_flow_out[df_flow_out['month'] == m]
#     df_flow_in_m = df_flow_in[df_flow_in['month'] == m]
#     df_flow_out_m.pop('month')
#     df_flow_in_m.pop('month')
#     df_flow_out_m.pop('time_slices')
#     df_flow_in_m.pop('time_slices')
#     df_counts = df_flow_out_m + df_flow_in_m
#     ACG = df_counts.corr(method='pearson')
#     ACG = np.nan_to_num(ACG)  # if some column is zero, the denominator is zero, correlation is nan
#     m_index = months.index(m)
#     ACGs[m_index] = ACG
df_flow_out.pop('time_slices')
df_flow_in.pop('time_slices')
df_counts = df_flow_out + df_flow_in
ACG = df_counts.corr(method='pearson')
ACG = ACG.fillna(value=-1)  # substitue 'nan' to '-1'
ACG = nd.array(ACG)
ACG = nd.softmax(ACG, axis=1)

# store the ajacency matrix
# ACGs = nd.array(ACGs)
# ACGs = nd.relu(ACGs)
# nd.save(path, ACGs)
nd.save('data/Haikou/acg', ACG)

# t.toc('gen correlation graph, finish ACGs aggregation')
t.toc('gen correlation graph, finish ACG aggregation')
