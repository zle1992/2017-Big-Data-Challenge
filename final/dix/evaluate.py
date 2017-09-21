import sys
from pyspark import SparkContext
output = sys.argv[1]
input = sys.argv[2]
sc = SparkContext(appName="test")
rdd = sc.textFile(input)
result = rdd.map(lambda line:line.split(" ")).map(lambda fields:fields[0])
result.saveAsTextFile(output)