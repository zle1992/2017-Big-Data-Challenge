import os,re,sys,math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



train_path = 'data/dsjtzs_txfz_training.txt'
test_path = 'data/dsjtzs_txfz_testB.txt'


def inf_to_nan(x):
    for i in range(len(x)):
        if x[i] == float("inf") or x[i] == float("-inf"):
            x[i] = float('nan') # or x or return whatever makes sense
    return x


def feature(path):
    train = pd.read_csv(path,sep=' ',header=None,encoding='utf-8',names=['id','data','target','label'])
    train['data'] = train['data'].apply(lambda x:[list(map(float,point.split(','))) for point in x.split(';')[:-1]])
    
    train['target'] = train['target'].apply(lambda x: list(map(float,x.split(","))))
    train['target_x'] = train['target'].apply(lambda x: x[0])
    train['target_y'] = train['target'].apply(lambda x: x[1])
    train['data_x'] = train['data'].apply(lambda x: [i[0] for i in x ])
    train['data_y'] = train['data'].apply(lambda x: [i[1] for i in x ])
    train['data_t'] = train['data'].apply(lambda x: [i[2] for i in x ])
    train['data_x'] = train['data_x'].apply(lambda x: np.array(x))
    train['data_y'] = train['data_y'].apply(lambda x: np.array(x))
    train['data_t'] = train['data_t'].apply(lambda x: np.array(x))
    del train['data']
    
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


   
    feats2=[ 'id',
      'speed_x','acc_speed_x','data_t','data_x',
      'delt_t','target_x','target_y','data_y',
      ]
  
    return  train[feats2]
train = feature(train_path)


for i in range(2850,2900):
    plt.figure(figsize=(16,9))
    print(i)
    plt.plot(train.iloc[i]['data_t'][0:],train.iloc[i]['data_x'][0:]/7,'b-')
    plt.plot(train.iloc[i]['data_t'][0:],train.iloc[i]['data_x'][0:]/7,'r*')
#     plt.plot(train.iloc[i]['target_x'],train.iloc[i]['target_y'],'r+')
    plt.ylim((100,225))
    plt.xlim((0,300))
    plt.title('id{0}'.format(train.iloc[i]['id']))
    plt.savefig('plot/train/0811_2/xt{0}.png'.format(train.iloc[i]['id']))

