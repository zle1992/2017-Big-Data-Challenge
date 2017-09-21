#encoding=utf-8
import sys
from pyspark import SparkContext
import numpy as np
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.mllib.linalg import Vectors, VectorUDT
import pandas as pd

output = sys.argv[1]
input = sys.argv[2]
lastnumber = 9

def inf_to_nan(x):
    for i in range(len(x)):
        if x[i] == float("inf") or x[i] == float("-inf"):
            x[i] = float('nan') # or x or return whatever makes sense
    return x

def feature(path,str):
    sc = SparkContext(appName="test")
    rdd = sc.textFile(path)
    result = rdd.map(lambda line:line.split(" "))
    
    sqlContext = SQLContext(sc)
    column_count = len(result.first())
    print(column_count)
    if column_count==4:
        train = sqlContext.createDataFrame(result, ["id", "data","target","label"])
        train = train.toPandas()
        train["label"] = train["label"].astype(int)
    else:
        train = sqlContext.createDataFrame(result, ["id", "data","target"])
        train = train.toPandas()
        train["label"] = -1
        
    train["id"] = train["id"].astype(int)
    
    train['data'] = train['data'].apply(lambda x: [list(map(float, point.split(','))) for point in x.split(';')[:-1]])
    train['target'] = train['target'].apply(lambda x: list(map(float, x.split(","))))
    
    train['data'] = train['data'].apply(lambda x: sorted(x, key=lambda k: k[2]))

    train['target_x'] = train['target'].apply(lambda x: x[0])
    train['target_y'] = train['target'].apply(lambda x: x[1])
    train['data_x'] = train['data'].apply(lambda x: [i[0] for i in x])
    train['data_y'] = train['data'].apply(lambda x: [i[1] for i in x])
    train['data_t'] = train['data'].apply(lambda x: [i[2] for i in x])
    train['data_x'] = train['data_x'].apply(lambda x: np.array(x))
    train['data_y'] = train['data_y'].apply(lambda x: np.array(x))
    train['data_t'] = train['data_t'].apply(lambda x: np.array(x))

    # delt
    train['delt_x'] = train['data_x'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
    train['delt_y'] = train['data_y'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
    train['delt_t'] = train['data_t'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
    train['delt_xy'] = train.delt_x ** 2 + train.delt_y ** 2
    train['delt_xy'] = train['delt_xy'].apply(lambda x: np.sqrt(x))
    # speed
    train['speed_x'] = (train.delt_x / train.delt_t).apply(lambda x: inf_to_nan(x)).apply(
        lambda x: np.nan_to_num(np.array(x)))
    train['speed_y'] = (train.delt_y / train.delt_t).apply(lambda x: inf_to_nan(x)).apply(
        lambda x: np.nan_to_num(np.array(x)))
    train['speed_xy'] = (train.delt_xy / train.delt_t).apply(lambda x: inf_to_nan(x)).apply(
        lambda x: np.nan_to_num(np.array(x)))

    # delt_speed
    train['delt_speed_x'] = train['speed_x'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
    train['delt_speed_y'] = train['speed_y'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
    train['delt_speed_xy'] = train['speed_xy'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
    train['delt_speed_t'] = train['data_t'].apply(lambda x: np.array(x)[2:] - np.array(x)[:-2])

    # acc
    train['acc_speed_x'] = (train.delt_speed_x / train.delt_speed_t).apply(lambda x: inf_to_nan(x)).apply(
        lambda x: np.nan_to_num(np.array(x)))
    train['acc_speed_y'] = (train.delt_speed_y / train.delt_speed_t).apply(lambda x: inf_to_nan(x)).apply(
        lambda x: np.nan_to_num(np.array(x)))
    train['acc_speed_xy'] = (train.delt_speed_xy / train.delt_speed_t).apply(lambda x: inf_to_nan(x)).apply(
        lambda x: np.nan_to_num(np.array(x)))

    train = train[[ 'label','id','data_x','data_y','data_t','target_x','target_y',
                    'delt_t','speed_x','speed_y','speed_xy','acc_speed_x']]

    return train
'''--------------------------------------------------------------------------------'''
def get_mean_std(train,key):
    train[key+'_mean'] = train[key].map(lambda x: x.mean() if x.shape[0] > 0 else 0)
    train[key + '_std'] = train[key].map(lambda x: x.std() if x.shape[0] > 0 else 0)

    train = train.fillna(value=0)
    return train

def x_acceleration_lastmean(x):
    leng = x.shape[0]
    if leng>lastnumber:
        x = x[leng-lastnumber:]
    mean = x.mean()
    return mean
    
def x_acceleration_laststd(x):
    leng = x.shape[0]
    if leng>lastnumber:
        x = x[leng-lastnumber:]
    std = x.std()
    return std

def lastmeanstd(train,key):
    train[key + '_lastmean'] = train[key].map(lambda x: x_acceleration_lastmean(x) if x.shape[0] > 0 else 0)
    train[key + '_laststd'] = train[key].map(lambda x: x_acceleration_laststd(x) if x.shape[0] > 0 else 0)

    train[key + '_laststdmean'] = train[key + '_laststd'] / train[key + '_lastmean']
    train = train.fillna(value=0)
    return train
'''--------------------------------------------------------------------------------'''

def get_speed(readpath,model):
    train = feature(readpath,model)
    
    train = get_mean_std(train, 'speed_x')
    
    
    # 最后几个点提取的特征
    train = lastmeanstd(train, 'speed_x')
    train['speed_xstd_laststd'] = train['speed_x_std'] / train['speed_x_laststd']
    

    # X_max与X_target关系
    train['X_max'] = train['data_x'].map(lambda x: x.max() if x.shape[0] > 0 else 0)
    train['X_target'] = train['target_x']
    train['get_target'] = train['X_max'] - train['X_target']
    train = train.replace(float("inf"), 0)
    
    
    # 差分
    train['delt_speed_x'] = train['speed_x'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
    train = get_mean_std(train, 'delt_speed_x')
    train['delt_speed_x_median'] = train['delt_speed_x'].map(lambda x: np.median(x) if x.shape[0] > 0 else 0)
    del train['delt_speed_x']
    
    #第一个
    train['first_data_x'] = train['data_x'].map(lambda x: x[0])
    train['first_speed_x'] = train['speed_x'].map(lambda x: x[0] if x.shape[0] > 0 else 0)
    train['first_data_y'] = train['data_y'].map(lambda x: x[0])
    train['first_delt_t'] = train['delt_t'].map(lambda x: x[0] if x.shape[0] > 0 else 0)
    
    train['delt_x'] = train['data_x'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
    # 是否有回转
    train['data_x_return'] = train['delt_x'].apply(
        lambda x: np.where(x<0)[0].shape[0] if x.shape[0]>0 else 0)
    train['data_x_return'] = train['data_x_return'].apply(
        lambda x: 0 if x>0 else 1)
    # train['data_x_return'] = train['delt_x'].apply(
    #     lambda x: np.where(x<0)[0].shape[0]>0 if x.shape[0]>0 else False)
    # train.loc[(train['data_x_return']==True),"data_x_return"] = 0
    # train.loc[(train['data_x_return']==False),"data_x_return"] = 1
    del train['delt_x']
    
    # 速度 平均值+（-）2* 标准差
    train['speed_x_beyond_index'] = train['speed_x'].map(
        lambda x: np.where(np.logical_or.reduce(((x > x.mean() + 2*x.std()),(x < x.mean() - 2*x.std())))
                          )[0] if x.shape[0] > 1 else np.array([]))
    train['speed_x_beyond_index_delt'] = train['speed_x_beyond_index'].map(
        lambda x: np.array(x)[1:] - np.array(x)[:-1] if x.shape[0]>1 else np.array([]))
    train['speed_x_beyond_index_delt_maxmean'] = train['speed_x_beyond_index_delt'].map(
        lambda x: x.max()/x.mean() if x.shape[0] > 1 else 0)
    train['speed_x_beyond_index'] = train['speed_x'].map(
        lambda x: np.where(np.logical_and.reduce((((x[1:-1] - x[:-2]) > 0), ((x[1:-1] - x[2:]) > 0)))
                          )[0] if x.shape[0] > 2 else np.array([]))
    del train['speed_x_beyond_index']
    del train['speed_x_beyond_index_delt']
    
    # #y坐标唯一值个数的标准差比平均值
    train['data_y_unique_value'] = train['data_y'].map(
            lambda x: np.unique(x,return_counts=True) if x.shape[0] > 0 else 0)
    train['data_y_unique_value_stdmean'] = train['data_y_unique_value'].map(
        lambda x: x[1].std()/x[1].mean() if x != 0 else 0)
    del train['data_y_unique_value']
    
    #开源特征
    train['time_delta_min'] = train['delt_t'].map(lambda x: x.min() if x.shape[0] > 0 else 0)
    train['delt_x'] = train['data_x'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
    train['delt_y'] = train['data_y'].apply(lambda x: np.array(x)[1:] - np.array(x)[:-1])
    train['delt_xy'] = train.delt_x ** 2 + train.delt_y ** 2
    train['delt_xy'] = train['delt_xy'].map(lambda x: np.sqrt(x))
    
    train['distance_deltas_max'] = train['delt_xy'].map(lambda x: x.max() if x.shape[0] > 0 else 0)
    del train['delt_xy']

    train['median_speed'] = train['speed_x'].map(lambda x: np.median(x) if x.shape[0] > 0 else 0)
    train['xs_delta_var'] = train['delt_x'].map(lambda x: x.var() if x.shape[0] > 0 else 0)
    train['xs_delta_max'] = train['delt_x'].map(lambda x: x.max() if x.shape[0] > 0 else 0)
    del train['delt_x']
    del train['delt_y']
    train['y_min'] = train['data_y'].map(lambda x: x.min() )
    train['x_min'] = train['data_x'].map(lambda x: x.min())
    
    # ********************** #
    

    del train['speed_x_mean']
    del train['speed_x_std']
    del train['speed_x_laststdmean']
    
    # del train['delt_speed_x_mean']
    # del train['delt_speed_x_std']

    del train['speed_x_lastmean']
    del train['speed_x_laststd']
    # del train['speed_xstd_laststd']
    # del train['X_max']
    del train['X_target']
    # del train['get_target']

    # del train['first_data_x']
    # del train['first_speed_x']
    # del train['first_data_y']
    # del train['first_delt_t']
    #**********************#
    
    train = train.drop(train.columns[1:12], axis=1)
    
    return train
'''--------------------------------------------------------------------------------'''
def fea_extra(fea_list):
    fea_str_list = [str(item) for item in fea_list]
    fea_str = " ".join(fea_str_list)
    return fea_str

if __name__ == '__main__':
    train = get_speed(input,'train')
    # print(train)
    train = train.astype(str)
    # train['result'] = train.apply( lambda x: " ".join([item for item in x]))
    train['result'] = train.apply( lambda x: " ".join(x),axis=1)
    
    sc = SparkContext.getOrCreate()
    # sqlContext = SQLContext(sc)
    # spark_df = sqlContext.createDataFrame(train,schema)
    # spark_df.show()
    # result = spark_df.rdd.map(fea_extra)
    
    result = sc.parallelize(train['result'].tolist())
    print(result.first())
    print(result.collect())
    result.saveAsTextFile(output)