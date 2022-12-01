import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import os
from pytictoc import TicToc
import multiprocessing as mp

from share import M_lat, N_lng, latitude_min, latitude_span, longitude_min, longitude_span, dates

# use stop watch
t = TicToc()  # create instance of class
t.tic()  # Start timer

# aggregate travel orders to calculate flows between regions
path0 = 'data/Haikou/trip-2017.db'  # original data path
path = 'data/Haikou/traffic-flows.db'  # destination data path

t.toc('flow aggregation, calculate hourly traffic flows')

# compute flow_out and flow_in of each region in every hour, t denotes the start time of the t'th time slice
# (region0_flow_in, ..., regionMN_flow_in, t) and (region0_flow_out, ..., regionMN_flow_out, t)
n_regions = M_lat * N_lng
regions = np.arange(n_regions)
time_slices = pd.date_range(start=dates[0], end=dates[1], freq='H')
n_time_slices = time_slices.size

# preparre query and data reader
con = sqlite3.connect(path0)
query = f'''
    select order_id, departure_time, normal_time, starting_lng, starting_lat, dest_lng, dest_lat
    from Raw
    where departure_time >= '{dates[0]}'
'''
reader = pd.read_sql_query(query, con, chunksize=100000)

# define a function to be executed parallely


def process_chunk(chunk):
    '''calculate the corresponding traffice flows (flow_out, flow_in) from a pandas data chunk.'''
    flow_out = np.zeros((n_time_slices, n_regions))
    flow_in = np.zeros((n_time_slices, n_regions))
    for index, row in chunk.iterrows():
        # start region in grid, real region number is i = i1*N_lng + j1
        i1 = int((row['starting_lat'] - latitude_min) / (latitude_span / M_lat))
        j1 = int((row['starting_lng'] - longitude_min) / (longitude_span / N_lng))
        i = i1 * N_lng + j1
        # destination region in grid, reall region number is j = i2*N_lng + j2
        i2 = int((row['dest_lat'] - latitude_min) / (latitude_span / M_lat))
        j2 = int((row['dest_lng'] - longitude_min) / (longitude_span / N_lng))
        j = i2 * N_lng + j2
        # if not moving out a region, ignore
        if i == j:
            continue
        # if the driving time (from region i to j) is not recorded, ignore
        if np.isnan(row['normal_time']):
            continue
        # calculate the time of this order, in the start region and the destination region
        fmt = '%Y-%m-%d %H:%M:%S'
        # t0 = datetime.strptime((datetime.strftime(time_slices[0], fmt)), fmt)
        t0 = time_slices[0]  # global start time
        t0 = int(t0.timestamp())
        t1 = time_slices[-1]  # global end time
        t1 = int(t1.timestamp())
        ti = datetime.strptime(row['departure_time'], fmt)  # the time when the vehicle leaves region i
        ti = int(ti.timestamp())  # seconds
        tj = ti + int(row['normal_time']) * 60
        # compute statistics of the vehicle flow
        if (i >= 0) and (i < n_regions) and (ti >= t0) and (ti <= t1):
            ti_index = (ti - t0) // 3600
            flow_out[ti_index, i] += 1
        if (j >= 0) and (j < n_regions) and (tj >= t0) and (tj <= t1):
            tj_index = (tj - t0) // 3600
            flow_in[tj_index, j] += 1
    t.toc('flow aggregation, calculate hourly flow records, process a chunk of raw data')
    return flow_out, flow_in


# load data from database to calculate flows
n_proc = os.cpu_count()
pool = mp.Pool(processes=n_proc)
ret = pool.map(process_chunk, [chunk for chunk in reader])  # 'apply' is not used here, because it mask console outputs
flow_out_list = [_[0] for _ in ret]  # ret is a list of (flow_out, flow_in) tuples, len(ret)==num_chunks_in_reader
flow_in_list = [_[1] for _ in ret]
flow_out = sum(flow_out_list)
flow_in = sum(flow_in_list)
pool.close()

# close the connection
con.close()

# filter out ineffective regions
total_flow_out = np.sum(flow_out, axis=0)
total_flow_in = np.sum(flow_in, axis=0)
indices = (total_flow_out + total_flow_in) >= 10  # remove regions where traffic is very low
indices = np.nonzero(indices)  # (array([...]),)
flow_out = flow_out[:, indices[0]]
flow_in = flow_in[:, indices[0]]
regions = regions[indices]

# prepare connection to write the regions flow data to database
con = sqlite3.connect(path)
# query = 'drop table if exists Flow_out;'
# con.execute(query)
# query = 'drop table if exists Flow_in;'
# con.execute(query)

# write flow data to database
columns = [f'region{_}' for _ in regions]
df_flow_out = pd.DataFrame(flow_out, columns=columns)
df_flow_in = pd.DataFrame(flow_in, columns=columns)
df_flow_out['time_slices'] = time_slices
df_flow_in['time_slices'] = time_slices
df_flow_out.to_sql('Flow_out', con=con, if_exists='replace')
df_flow_in.to_sql('Flow_in', con=con, if_exists='replace')

# close the connection
con.close()

t.toc('finish traffic flow aggregation')
