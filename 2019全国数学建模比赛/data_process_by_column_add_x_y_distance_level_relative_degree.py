# --*-- coding:gb18030 --*--

import numpy as np
import pandas as pd
import math
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os
import gc


def get_distance(data):
    data['distance']  = np.sqrt(pow(data['Cell X'] - data['X'], 2) + pow(data["Cell Y"] - data['Y'], 2)) * 5
    #data['distance']  = pow(data['Cell X'] - data['X'], 2) + pow(data["Cell Y"] - data['Y'], 2)
    return data

def get_relative_height(data):
    #data['distance']  = math.sqrt(pow(data['Cell X'] - data['X'], 2) + pow(data["Cell Y"] - data['Y'], 2)) * 5
    data['relative_height']  = np.fabs(data['Cell Altitude'] + data["Height"] - (math.tan(math.radians(data['Electrical Downtilt'] + data['Mechanical Downtilt'])) * data['distance'] + data['Altitude']) )
    return data

def get_x_distance(data):
    data['x_distance']  = np.sqrt(pow(data['Cell X'] - data['X'], 2)) * 5
    return data

def get_y_distance(data):
    data['y_distance']  = np.sqrt(pow(data["Cell Y"] - data['Y'], 2)) * 5
    return data

def count_level_relative_degree(cellx,celly,x,y):
    degree = math.degrees(math.atan( (celly - y)/(cellx - x) ))
    if degree < 0:
        degree = 180- math.fabs(degree)
    return degree

def get_level_relative_degree(data):
    # 计算水平相对角度
    # 先判断是否为特殊的点  在x轴上或者在y轴上
    cellx = data['Cell X']
    celly = data['Cell Y']
    x = data['X']
    y = data['Y']
    Azimuth = data['Azimuth']
    jiaodu = 0
    if cellx == x:
        if celly > 0:
            jiaodu = math.fabs(90 - Azimuth)
        else:
            jiaodu = math.fabs(Azimuth - 270)
    if celly == y:
        if x > 0:
            jiaodu = math.fabs(90 - Azimuth)
        else:
            jiaodu = math.fabs(Azimuth - 270)
    # 在象限内
    if cellx < x and y > celly:  # 1
        jiaodu = 90 - count_level_relative_degree(cellx, celly, x, y) - Azimuth
    elif cellx > x and y > celly:  # 2
        jiaodu = 360 - count_level_relative_degree(cellx, celly, x, y) + 90 - Azimuth
    elif cellx > x and y < celly:  # 3
        jiaodu = 90 - count_level_relative_degree(cellx, celly, x, y) + 180 - Azimuth
    elif cellx < x and y < celly:  # 4
        jiaodu = 180 - count_level_relative_degree(cellx, celly, x, y) + 90 - Azimuth
    jiaodu = math.fabs(jiaodu)
    if jiaodu <= 180:
        data['level_relative_degree'] =  math.fabs(jiaodu)
    else:
        data['level_relative_degree'] = math.fabs(jiaodu - 180)
    return data



# 读取文件
os.chdir('D:/data/train_set_process_by_column_add_xy_distance/')
file_chdir = os.getcwd()
filecsv_list = []
for root,dirs,files in os.walk(file_chdir):
    for file in files:
        if os.path.splitext(file)[1] == '.csv':
            filecsv_list.append(file)

count = 0
for csv in filecsv_list:
    if os.path.exists('../train_set_process_by_column_add_xy_distance_level_relative_degree/'+csv):
        continue
    data = pd.DataFrame()
    data = pd.read_csv(csv,header = 0,sep=None,encoding='gb18030')
    one_data = data['Altitude']
    #data.insert(17,'fa_height',one_data)
    #data.insert(17, 'distance', one_data)
    #data.insert(17, 'relative_height', one_data)
    #data.insert(17, 'downtilt', one_data)
    #data.insert(17, 'x_distance', one_data)
    #data.insert(17, 'y_distance', one_data)
    data.insert(17, 'level_relative_degree', one_data)
    # 多列一同计算
    # 注释一波，看看时间会不会减少
    # data['downtilt'] = data.apply(lambda x:x['Electrical Downtilt'] + x['Mechanical Downtilt'], axis=1)
    # data['fa_height'] = data.apply(lambda x:x['Height'] + x['Cell Building Height'] + x['Cell Altitude'] - x['Altitude'], axis=1)
    #
    #
    # data = data.apply(get_distance,axis=1) #
    # data = data.apply(get_relative_height, axis=1)  #
    #
    # # x y distance
    # data = data.apply(get_x_distance, axis=1)
    # data = data.apply(get_y_distance, axis=1)

    data = data.apply(get_level_relative_degree, axis=1)
    # 将数据写入新的文件夹中
    count = count+1
    print("current is ", count, "current file is ", csv)
    if count > 310:
        break
    data.to_csv('../train_set_process_by_column_add_xy_distance_level_relative_degree/'+csv, index=False)
    del data
    gc.collect()

