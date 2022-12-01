import sqlite3
import zipfile
import fnmatch
import time
from colorama import Cursor
import pandas as pd
import multiprocessing as mp
from pytictoc import TicToc
import os


# use stop watch
t = TicToc()  # create instance of class
t.tic()  # Start timer

# create database, connect it and cursor
con = sqlite3.connect('./data/Haikou/trip-2017.db')
cur = con.cursor()

# add a table
query = '''
drop table if exists Raw;
create table if not exists Raw (
    order_id char(16),
    departure_time datetime,
    normal_time int,
    starting_lat float,
    starting_lng float,
    dest_lat float,
    dest_lng float,
    primary key (order_id));
'''
cur.executescript(query)
con.commit()

# define a method to be execute parallely, which process a chunk of data


def process_chunk(df_chunk):
    list1 = []
    query = '''insert or ignore into Raw values (?, ?, ?, ?, ?, ?, ?) '''
    for index, row in df_chunk.iterrows():
        temp = [row['order_id'], row['departure_time'], row['normal_time'],
                row['starting_lat'], row['starting_lng'], row['dest_lat'], row['dest_lng']]
        temp = tuple(temp)
        list1.append(temp)
    cur.executemany(query, list1)
    t.toc('data extraction, write to database, process a chunk')


# read data from zip file and write into database
with zipfile.ZipFile('./data/Haikou/trip-2017.zip') as zf:
    flist = zf.namelist()[0:]
    flist = fnmatch.filter(flist, "*.txt")
    reader = [pd.read_csv(zf.open(fname),
                          sep='\t',
                          iterator=True,
                          chunksize=100000,
                          usecols=['order_id', 'departure_time', 'normal_time', 'starting_lng', 'starting_lat', 'dest_lng', 'dest_lat'])
              for fname in flist]

    # parallel process
    # sqlite3 only support 'one write many read', the code below leads to errors
    # n_proc = os.cpu_count()
    # pool = mp.Pool(processes=n_proc)
    # for rd in reader:
    #     pool.map(process_chunk, [chunk for chunk in rd])
    # pool.close()

    # seriel process
    for rd in reader:
        for chunk in rd:
            process_chunk(chunk)
        con.commit()

    # data = [rd.get_chunk() for rd in reader]
    # df = pd.concat(data)
    # print(df.columns)
    # print(df.head())
    # df[['order_id',
    #        'district',
    #        'county',
    #        'passenger_count',
    #        'start_dest_distance',
    #        'arrive_time',
    #        'departure_time',
    #        'normal_time',
    #        'dest_lng',
    #        'dest_lat',
    #        'starting_lng',
    #        'starting_lat',
    #        'year',
    #        'month',
    #        'day']]

# close database connection
con.close()

t.toc('data extraction, finished')
