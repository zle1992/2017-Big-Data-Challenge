import numpy as np 
import sys
from pyspark import SparkContext
output = sys.argv[1]
input = sys.argv[2]
def fea_extra_one(a_sample_str):
    all = a_sample_str.split(" ")
    id = all[0]
    data = [list(map(float,point.split(','))) for point in all[1].split(';')[:-1]]
    data_x = np.array(data)[:,0]
    data_y = np.array(data)[:,1]
    data_t = np.array(data)[:,2]
    
    target = list(map(float,all[2].split(",")))
    target_x = target[0]
    target_y = target[1]
    
    
    first_x,first_y = data_x[0],data_y[0]
    len_x = data_x.shape[0]
    
    
    
    #fea_list=[all[3],first_x,first_y,len_x]
    fea_list=[id,first_x,first_y,len_x]
    #print(data)
    
    
    
    
    fea_str_list = ['{}:{}'.format(i+1,j) for i,j in enumerate(fea_list)]
    #fea_str_list = [str(label)] + fea_str_list
    fea_str =' '.join(fea_str_list)
    return fea_str

    
sc = SparkContext(appName="test")
rdd = sc.textFile(input)
result = rdd.map(fea_extra_one)
result.saveAsTextFile(output)
print(result.first())


#$${output_prefix}/ zle/feature/train_0729
# fea_list=[first_x,first_y,len_x]
#27s
#${common_prefix}/ download_334

#{output_prefix}/zle/model/train_0729
#{output_prefix}/zle/predict/xgb_0729