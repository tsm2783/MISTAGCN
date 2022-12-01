from calendar import month
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
from mxnet import ndarray as nd
import os
from pytictoc import TicToc
import multiprocessing as mp

from share import latitude_min, latitude_span, M_lat, longitude_min, longitude_span, N_lng, dates, months

# use stop watch
t = TicToc()  # create instance of class
t.tic()  # Start timer

# calculate AIG
path0 = './data/Haikou/trip-2017.db'
path = './data/Haikou/aig'
t.toc('gen interaction graph, calculate AIG')

# load data (here, only 'regions' needed)
Xr_sample, Xd_sample, Xw_sample, Yp_sample, regions = nd.load('./data/Haikou/data-samples')
regions = [_ for _ in regions]  # convert to a list

# initialize monthly interaction graph (ajacency matrices)
# n_months = len(months)
n_regions = len(regions)

# read traffic records
con = sqlite3.connect(path0)
query = f'''
    select order_id, departure_time, normal_time, starting_lng, starting_lat, dest_lng, dest_lat
    from Raw
    where departure_time between '{dates[0]}' and '{dates[1]}'
'''
reader = pd.read_sql_query(query, con, chunksize=1000)

# define a function to be executed parallely


def process_chunk(chunk):
    # aigs = nd.zeros(shape=(n_months, n_regions, n_regions))
    aig = nd.zeros(shape=(n_regions, n_regions))
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
        fmt = '%Y-%m-%d %H:%M:%S'
        ti = datetime.strptime(row['departure_time'], fmt)  # the time when the vehicle leaves region i
        tj = ti + timedelta(minutes=int(row['normal_time']))  # the time when vehicle arrives region j
        mi = ti.month
        mj = tj.month
        if mi != mj:  # only calculate flows in the same month
            continue
        # mi_idx = months.index(mi)
        # aggregate
        if (i in regions) and (j in regions):
            i_idx = regions.index(i)
            j_idx = regions.index(j)
            # aigs[mi_idx, i_idx, j_idx] += 1
            # aigs[mi_idx, j_idx, i_idx] += 1
            aig[i_idx, j_idx] += 1
            aig[j_idx, i_idx] += 1
    t.toc('gen interaction graph, process a chunk')
    return aig


# load data from database to calculate interaction matrix
n_proc = os.cpu_count()
pool = mp.Pool(processes=n_proc)
ret = pool.map(process_chunk, [chunk for chunk in reader])  # 'apply' is not used here, because it mask console outputs
# AIGs = sum(ret)
AIG = sum(ret)
pool.close()

# close database connection
con.close()

# store the ajacency matrix
# AIGs = nd.array(AIGs)
# nd.save(path, AIGs)
AIG = nd.array(AIG)
AIG = nd.softmax(AIG, axis=1)
nd.save(path, AIG)

# t.toc('gen interaction graph, finish AIGs aggregation')
t.toc('gen interaction graph, finish AIG aggregation')
