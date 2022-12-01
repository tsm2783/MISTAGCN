import sqlite3
from mxnet import ndarray as nd
import pandas as pd
from pytictoc import TicToc
import numpy as np

from share import latitude_min, latitude_span, M_lat, longitude_min, longitude_span, N_lng, R

# use stop watch
t = TicToc()  # create instance of class
t.tic()  # Start timer

t.toc('gen distance graph, calculate ADG')

# calculate ADG

Xr_sample, Xd_sample, Xw_sample, Yp_sample, regions = nd.load('./data/Haikou/data-samples')
regions = regions.asnumpy()
regions = [_ for _ in regions]
n_regions = len(regions)

# compute distance graph
ADG = nd.zeros(shape=(n_regions, n_regions))
for i_idx in range(n_regions):
    i = regions[i_idx]
    center_lat_i = latitude_min + ((i // N_lng) + 0.5) * latitude_span / M_lat
    center_lng_i = longitude_min + ((i % N_lng) + 0.5) * longitude_span / N_lng
    center_lat_i *= np.pi / 180
    center_lng_i *= np.pi / 180
    for j_idx in range(i_idx + 1, n_regions):
        j = regions[j_idx]
        center_lat_j = latitude_min + ((j // N_lng) + 0.5) * latitude_span / M_lat
        center_lng_j = longitude_min + ((j % N_lng) + 0.5) * longitude_span / N_lng
        center_lat_j *= np.pi / 180
        center_lng_j *= np.pi / 180
        # dij = np.sqrt((center_lng_j - center_lng_i)**2 + (center_lat_j - center_lat_i)**2)
        # Cartesian coordinates
        # A (Rcos(lat1)cos(lng1),  Rcos(lat1)sin(lng1), Rsin(lat1))
        # B（Rcos(lat2)cos(lng2),  Rcos(lat2)sin(lng2), Rsin(lat2)
        # Cosine of the Angle between vectors OA and OB
        # COS = cos(lat2) * cos(lat1) * cos(lng1-lng2) + sin(lat2)*sin(lat1)
        d = R * np.arccos(np.cos(center_lat_j) * np.cos(center_lat_i) * np.cos(center_lng_j -
                          center_lng_i) + np.sin(center_lat_j) * np.sin(center_lat_i))
        a = 1.0 / d
        # if a >= 0.05:
        #     ADG[i_idx,j_idx] = a
        #     ADG[j_idx,i_idx] = a

# store the ajacency matrix
ADG = nd.array(ADG)
ADG = nd.softmax(ADG, axis=1)
nd.save('data/Haikou/adg', ADG)

t.toc('gen distance graph, finish ADG aggregation')
