import numpy as np 
import sys
from pyspark import SparkContext
output = sys.argv[1]
input = sys.argv[2]
sc = SparkContext(appName = 'test')
rdd = sc.textFile(input)
result = rdd.collect()
result_final = []
for i ,item in enumerate(result):
    print(i,item)
    item=float(item.split('\t')[1])
    print(item)
    if(item<0.5):
        result_final.append(i+1)
print("machine nums:",len(result_final))
result_rdd = sc.parallelize(result_final)
result_rdd.saveAsTextFile(output)