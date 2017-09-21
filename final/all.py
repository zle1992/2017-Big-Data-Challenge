#encoding=utf-8
import os,re,sys,math
import pandas as pd
import numpy as np
from scipy import sparse
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.mllib.linalg import Vectors, VectorUDT



output = sys.argv[1]
input1 = sys.argv[2]
input2 = sys.argv[3]
sc = SparkContext(appName="test")
def fun(X,Y,LENGTH):
    from scipy.optimize import leastsq
    if LENGTH == 2:
        def error(p, x, y):
            a0, a1 = p
            return (a0 + a1 * x) - y
    elif LENGTH == 3:
        def error(p, x, y):
            a0, a1, a2 = p
            return (a0 + a1 * x +a2*x**2) - y

    tempx = list(X)[:min(X.shape[0],Y.shape[0])]
    tempy = list(Y)[:min(X.shape[0],Y.shape[0])]
    
    if len(tempx)<LENGTH:
        return np.array([0] * LENGTH)
        tempx+= [tempx[-1]] * (LENGTH-len(tempx))
        tempy+= [tempy[-1]] * (LENGTH-len(tempy))
    para = leastsq(error, np.array([1] * LENGTH), args=(np.array(tempx), np.array(tempy)))
        
    return para[0]
    
def inf_to_nan(x):
    for i in range(len(x)):
        if x[i] == float("inf") or x[i] == float("-inf"):
            x[i] = float('nan') # or x or return whatever makes sense
    return x



def sample_data(a,size = 0.7):
    np.random.seed(1)
    len_data = len(a)
    #size = np.random.random_sample()  #抽样比例随机
    size = int(size *len_data)
    if size < 1:
        size = 1
    index = np.sort((np.random.random(size)*len_data).astype(int))
    return np.array(a)[index]
def feature(path,extend=True):
    sc = SparkContext.getOrCreate()
    rdd = sc.textFile(path)
    result = rdd.map(lambda line:line.split(" "))
    
    sqlContext = SQLContext(sc)
    column_count = len(result.first())
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
    train['data'] = train['data'].apply(lambda x: sorted(x, key=lambda k: k[2]))

    
    if extend:
        #######扩充数据集#####


        train_new_1 = train.copy()
        train_new_1['data'] = train['data'].apply(lambda x :sample_data(x,size = 0.3))
        train_new_1['id'] = train['id'] + 3000 

        train=train.append(train_new_1)


        train_new_1 = train.copy()
        train_new_1['data'] = train['data'].apply(lambda x :sample_data(x,size = 0.7))
        train_new_1['id'] = train['id'] + 3000 
        train=train.append(train_new_1)
        
        train['index'] = range(len(train))
        train = train.set_index('index')


    

    train['target'] = train['target'].apply(lambda x: list(map(float,x.split(","))))
    train['data_x'] = train['data'].apply(lambda x: [i[0] for i in x ])
    train['data_y'] = train['data'].apply(lambda x: [i[1] for i in x ])
    train['data_t'] = train['data'].apply(lambda x: [i[2] for i in x ])
    train['data_x'] = train['data_x'].apply(lambda x: np.array(x))
    train['data_y'] = train['data_y'].apply(lambda x: np.array(x))
    train['data_t'] = train['data_t'].apply(lambda x: np.array(x))
    

    del train['data']

    
    train['target_x'] = train['target'].apply(lambda x: x[0])
    train['target_y'] = train['target'].apply(lambda x: x[1])
    #delt 
    train['delt_x'] = train['data_x'].apply(lambda x: np.array(x)[1:]-np.array(x)[:-1])
    train['delt_y'] = train['data_y'].apply(lambda x: np.array(x)[1:]-np.array(x)[:-1])
    train['delt_t'] = train['data_t'].apply(lambda x: np.array(x)[1:]-np.array(x)[:-1])
    train['delt_xy'] = train.delt_x**2 + train.delt_y**2
    train['delt_xy'] = train['delt_xy'].apply(lambda x: np.sqrt(x))
    #speed
    train['speed_x'] = (train.delt_x/train.delt_t).apply(lambda x :inf_to_nan(x)).apply(lambda x :np.nan_to_num(np.array(x)))
    train['speed_y'] = (train.delt_y/train.delt_t).apply(lambda x :inf_to_nan(x)).apply(lambda x :np.nan_to_num(np.array(x)))
    train['speed_xy'] = (train.delt_xy/train.delt_t).apply(lambda x :inf_to_nan(x)).apply(lambda x :np.nan_to_num(np.array(x)))

    #delt_speed
    train['delt_speed_x'] = train['speed_x'].apply(lambda x: np.array(x)[1:]-np.array(x)[:-1])
    train['delt_speed_y'] = train['speed_y'].apply(lambda x: np.array(x)[1:]-np.array(x)[:-1])
    train['delt_speed_xy'] = train['speed_xy'].apply(lambda x: np.array(x)[1:]-np.array(x)[:-1])
    train['delt_speed_t'] = train['delt_t'].apply(lambda x: np.array(x)[1:]-np.array(x)[:-1])

    #acc
    train['acc_speed_x'] = (train.delt_speed_x/train.delt_speed_t).apply(lambda x :inf_to_nan(x)).apply(lambda x :np.nan_to_num(np.array(x)))
    train['acc_speed_y'] = (train.delt_speed_y/train.delt_speed_t).apply(lambda x :inf_to_nan(x)).apply(lambda x :np.nan_to_num(np.array(x)))
    train['acc_speed_xy'] = (train.delt_speed_xy/train.delt_speed_t).apply(lambda x :inf_to_nan(x)).apply(lambda x :np.nan_to_num(np.array(x)))

    #轨迹长度
    train['len_x'] =  train['data_x'].apply(lambda x: len(x))



    #first
    train['first_data_x'] = train['data_x'].apply(lambda x : x[0])
    train['first_speed_x'] = train['speed_x'].apply(lambda x: x[0] if x.shape[0] > 0 else 0)
    train['first_data_y'] = train['data_y'].apply(lambda x : x[0])
    train['first_delt_t'] = train['delt_t'].apply(lambda x: x[0] if len(x) > 0 else 0)
    
    
    
    #
     
    train['time_delta_min'] = train['delt_t'].map(lambda x: x.min() if x.shape[0] > 0 else 0)
    train['distance_deltas_max'] = train['delt_xy'].map(lambda x: x.max() if x.shape[0] > 0 else 0)
    train['median_speed'] = train['speed_x'].map(lambda x: np.median(x) if x.shape[0] > 0 else 0)
    train['xs_delta_var'] = train['delt_x'].map(lambda x: x.var() if x.shape[0] > 0 else 0)
    train['xs_delta_max'] = train['delt_x'].map(lambda x: x.max() if x.shape[0] > 0 else 0)
    train['x_min'] = train['data_x'].map(lambda x: x.min())
    train['y_min'] = train['data_y'].map(lambda x: x.min())
    
    
    #####################
       # X_max与X_target关系
    train['X_max'] = train['data_x'].map(lambda x: x.max() if x.shape[0] > 0 else 0)
    train['get_target'] = train['X_max'] - train['target_x']
    
    #特征75 76
    train['data_x_return'] = train['delt_x'].apply(lambda x: np.where(x<0)[0].shape[0]>0 if x.shape[0]>0 else False)
    # #y坐标唯一值个数的标准差比平均值
    train['data_y_unique_value'] = train['data_y'].map(
            lambda x: np.unique(x,return_counts=True) if x.shape[0] > 0 else 0)
    train['data_y_unique_value_stdmean'] = train['data_y_unique_value'].map(
        lambda x: x[1].std()/x[1].mean() if x != 0 else 0)
    del train['data_y_unique_value']
    
    
    train['dxspeed_mean'] = train["delt_speed_x"].apply(lambda x :np.mean(x) if x.shape[0] > 0 else 0)  #x方向速度差分的平均值
    train['dxspeed_std']  = train["delt_speed_x"].apply(lambda x :np.std(x) if x.shape[0] > 0 else 0)    #x方向速度差分的标准差
    train['dxspeed_median']  = train["delt_speed_x"].apply(lambda x :np.median(x) if x.shape[0] > 0 else 0)    
    #x方向速度全局标准差与最后N个标准差比值
    train['speed_xstd']  = train["speed_x"].apply(lambda x :np.std(x) if x.shape[0] > 0 else 0)    #x方向速度差分的标准差
    train['speed_x_laststd']  = train["speed_x"].apply(lambda x :np.std(x[-9:]) if x.shape[0] > 0 else 0)    #x方向速度差分的标准差
    train['speed_xstd_laststd'] = train['speed_xstd'] / train['speed_x_laststd']
    train['speed_xstd_laststd'] = train['speed_xstd_laststd'].apply(lambda x : float('nan') if(x==float('-inf') or x==float('inf')) else x)
    
    ###0717
    
    #delt_speed
    train['delt_acc_x'] = train['delt_speed_x'].apply(lambda x: np.array(x)[1:]-np.array(x)[:-1])
    #train['delt_acc_y'] = train['delt_speed_y'].apply(lambda x: np.array(x)[1:]-np.array(x)[:-1])
    train['delt_acc_t'] = train['delt_speed_t'].apply(lambda x: np.array(x)[1:]-np.array(x)[:-1])
    train['delt_acc_speed_x'] = (train.delt_acc_x/train.delt_acc_t).apply(lambda x :inf_to_nan(x)).apply(lambda x :np.nan_to_num(np.array(x)))
    train['dxacc_mean'] = train["delt_acc_speed_x"].apply(lambda x :np.mean(x) if x.shape[0] > 0 else 0)  #x方向速度差分的平均值
    train['dxacc_std']  = train["delt_acc_speed_x"].apply(lambda x :np.std(x) if x.shape[0] > 0 else 0)    #x方向速度差分的标准差
    train['dxacc_median']  = train["delt_acc_speed_x"].apply(lambda x :np.median(x) if x.shape[0] > 0 else 0)    
    train['dxacc_range']  = train["delt_acc_speed_x"].apply(lambda x :np.max(x)-np.min(x) if x.shape[0] > 0 else 0)    #x方向速度差分的标准差
    #0719
    from scipy import stats
    train['log_delt_x'] = train['data_x'].apply(lambda x: np.log1p(np.array(x)[1:]-np.array(x)[:-1]))
    train['log_delt_y'] = train['data_y'].apply(lambda x: np.log1p(np.array(x)[1:]-np.array(x)[:-1]))
    train['log_delt_t'] = train['data_t'].apply(lambda x: np.log1p(np.array(x)[1:]-np.array(x)[:-1]))
    
    train['xy_angles'] = train['log_delt_x'] - train['log_delt_y']
    train['xt_angles'] = train['log_delt_x'] - train['log_delt_t']
    train['xy_angles_kurtosis']= train['xy_angles'].apply(lambda x: stats.kurtosis(x))
    train['xt_angles_max'] =train['xt_angles'].apply(lambda x :np.max(x) if x.shape[0] > 0 else 0)  #x方向速度差分的平均值
    
    
    train['dxyspeed_range']  = train["delt_speed_xy"].apply(lambda x :np.max(x)-np.min(x) if x.shape[0] > 0 else 0)    #x方向速度差分的标准差
    
     
    #0731
    train['slope'] = train['data_x']/ train['data_y']
    train['slope_speed'] = train['slope']/ train['data_t']
    train['slope_acc'] = train['slope_speed']/ train['data_t']
    
    train['delt_y_mean'] = train['delt_y'].apply(lambda x : np.mean(x))
    train[['27','27_2']]=train[['delt_t','speed_x']].apply(lambda x: fun(x['delt_t'],x['speed_x'],2) ,axis=1 )
    train[['29','30']]=train[['delt_speed_t','acc_speed_y']].apply(lambda x: fun(x['delt_speed_t'],x['acc_speed_y'],2) ,axis=1 )
    train[['31','32']]=train[['delt_t','slope_speed']].apply(lambda x: fun(x['delt_t'],x['slope_speed'],2) ,axis=1 )
    train[['37','38']]=train[['acc_speed_y','slope_acc']].apply(lambda x: fun(x['acc_speed_y'],x['slope_acc'],2) ,axis=1 )

    
    train['delt_x_max_index'] = train['delt_x'].apply(lambda x : np.argmax(x) if x.shape[0]>0 else 0 )
    
    
    
    train['data_x_max_last'] = train['data_x'].apply(lambda x: np.max(x)-x[-1])
    train['data_x_return'] =(train['data_x_max_last']>0).astype(int)

    feat = ['id','label',

    'speed_xstd_laststd', 'X_max', 'get_target', #20+first

    'dxspeed_mean', 'dxspeed_std', 'data_x_return', 'data_y_unique_value_stdmean',#特征75 76

     'time_delta_min', 'distance_deltas_max',  'median_speed','xs_delta_var', 'xs_delta_max',#开源部分特征
     'dxacc_mean',    #715 feature
              'dxacc_std', #gooooood
    'xt_angles_max', #0718
         'xy_angles_kurtosis',#0719
    'dxacc_range',#0720


     'delt_y_mean','27', '29', '30',  '31',  '32',  '38',
      'delt_x_max_index',  
    
        ]
    return train[feat][train['data_x_return']==0]

def lr_pre(params,train_data1,train_label,test_data1):
    threshold = params
    from sklearn.preprocessing import StandardScaler
    std = StandardScaler()
    X_train = train_data1
    X_test = test_data1
    y_train = train_label
    std.fit(X_train)
    X_train_std = std.transform(X_train)
    X_test_std = std.transform(X_test)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1000,random_state=0)
    lr.fit(X_train_std, y_train)
    lr.predict_proba(X_test_std[1,:]) # 查看第一个测试样本属于各个类别的概率
    val_y_pre = lr.predict_proba(X_test_std)[:,1]
    val_y_pred = [int(i >= threshold) for i in val_y_pre]
    return val_y_pre,val_y_pred



#train = feature(input,extend  = True)
train = feature(input1,extend = False)
test_b = feature(input2,extend = False)


import warnings
warnings.filterwarnings("ignore")


final_train=train.fillna(100)
final_test_b=test_b.fillna(100)
 

train_data1 ,train_label = final_train.drop(['id','label'], axis=1).astype(float),final_train.label
#lrcv(0.5,train_data1,train_label)
test_data1 = final_test_b.drop(['id','label'], axis=1).astype(float)
final_test_b['label_prob'] = 0
final_test_b['label_prob'] ,final_test_b["label"]= lr_pre(0.5,train_data1,train_label,test_data1)

negative_number=len(final_test_b)-final_test_b.label.sum()
print("negative number:",negative_number)
id = final_test_b[final_test_b.label>0]['id']



result = sc.parallelize(list(id))
print(result.first())
result.saveAsTextFile(output)