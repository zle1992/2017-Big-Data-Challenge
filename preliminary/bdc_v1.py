# -*- coding: utf-8 -*-
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import recall_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold  
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold


def train_val():
	train_id,train_x,train_y,train_t,train_x_d,train_y_d,train_label = readtrain()
	################# train ######################
	cols=15 #选择要选择的列数 #重要参数！！！3 最好
	train_feature = feature3(cols,train_x,train_y,train_t)
	#train_feature = feature2(cols,x,y,t,x_d,y_d,0.7)
	#处理missing value
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	train_feature= imp.fit_transform(train_feature)
	train_feature = (train_feature-np.mean(train_feature)) / np.std(train_feature)
	#归一化
	#feature = preprocessing.scale(feature)
	#划分数据集
	train_x, val_x, train_y, val_y= train_test_split( train_feature,train_label, test_size=0.2, random_state=64)#随机选择训练与测试数据

	#########################################  测试准确率##############
	clf = AdaBoostClassifier(n_estimators=1000)
	#clf = RandomForestClassifier(max_depth=5, n_estimators=200,oob_score=True)
	#clf = SVC(C=1)
	#clf = svm.SVC(kernel='linear', C=1).fit(train_x, train_y)
	#clf = linear_model.SGDClassifier()

	clf.fit(train_x,train_y)


	###################################计算得分########################
	val_y_pre = clf.predict(val_x)
	#print ('accuracy_score:{0:.3f}'.format(accuracy_score(test_y, test_y_pre)))
	#print ('precision_recall_fscore_support:{0:.3f}'.format(precision_recall_fscore_support(test_y, test_y_pre)))

	confusion_matrixs = confusion_matrix(val_y, val_y_pre)
	recall =float(confusion_matrixs[0][0]) / float(confusion_matrixs[0][1]+confusion_matrixs[0][0])
	precision = float(confusion_matrixs[0][0]) / float(confusion_matrixs[1][0]+confusion_matrixs[0][0])
	print(confusion_matrixs)
	print("recall:",recall)
	print("precision:",precision)
	F = 5*precision* recall/(2*precision+3*recall)*100
	print("score :",F)

def readtrain():
	####################读入文件  返回的是x y t x_d y_d label id  list
	train_path = "data/dsjtzs_txfz_training.txt"
	id = []
	a2 = []
	a3 = []
	label = []
	with open(train_path,"r") as f:
		for line in f:
			line = line.split()
			id.append(line.pop(0))
			a2.append(line.pop(0))
			a3.append(line.pop(0))
			label.append(line.pop(0))
	xyt = [[]]
	x,y,t = [],[],[]
	for i in range(len(a2)):
		xyt = a2[i].split(";")
		xyt.pop(-1)
		x.append([])
		y.append([])
		t.append([])
		for j in range(len(xyt)):
			x[i].append(float(xyt[j].split(",").pop(0)))
			y[i].append(float(xyt[j].split(",").pop(1)))
			t[i].append(float(xyt[j].split(",").pop(2)))


	xy = [[]]
	x_d,y_d = [] ,[]
	for i in range(len(a3)):
		xy = a3[i].split(",")
		x_d.append(float(xy[0]))
		y_d.append(float(xy[1]))
		#########id lable np.array##################
	id = np.array(id).astype(int)
	label = np.array(label).astype(int)
	return id,x,y,t,x_d,y_d,label
##############plot##########
def plot_time():
	i=6
	print(label[i])
	t[i] = np.array(t[i])
	x = np.arange(len(t[i]-1))
	plt.plot(x,t[i])# use pylab to plot x and y
	plt.ylabel("time")
	plt.xlabel("等间隔")
	#plt.scatter(x_d[i],y_d[i])
	plt.show()# show e plot on the screen

#plot_time()

#####################辅助函数##################
#inuput x 
#outpuut len_x  N*1  x[i]的长度
def len_x(x):
	len_x = []
	for i in range(len(x)):
		lenxi = len(x[i])
		len_x.append([lenxi])
	return len_x

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#input : 一维list  1 * N
#output 二维矩阵，完全相同的列向量 N*cols

def reshape(x,cols):
	x = np.array(x)
	x = np.mat(x)
	a=np.ones((1,cols))
	a = np.mat(a)
	return x.transpose() * a

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#input x :2维List
#output: 指定列数的mat

def sample(x,cols):  #
	for i in range(len(x)):
		if cols > len(x[i]):
			cu = (cols -len(x[i]))/2+1
			oneshead = np.ones(int(cu)) * x[i][0]
			onestail = np.ones(int(cu)) * x[i][-1]
			x[i] = np.hstack((oneshead,x[i],onestail))
		cut = int(len(x[i]) /cols)    
		x[i] = x[i][0:len(x[i]):cut]
		x[i] = x[i][:cols]
	return np.mat(x)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#input x :2维List
#output: 指定列数的mat
#以0代替木有的值   
def sample2(x,cols):  #
	for i in range(len(x)):
		if cols > len(x[i]):
			cu = (cols -len(x[i]))/2+1
			oneshead = np.ones(int(cu)) * 0
			onestail = np.ones(int(cu)) * 0
			x[i] = np.hstack((oneshead,x[i],onestail))
		cut = int(len(x[i]) /cols)    
		x[i] = x[i][0:len(x[i]):cut]
		x[i] = x[i][:cols]
	return np.mat(x)

####提取特征
'''没有考虑当前坐标与目标坐标的位置
'''
def feature1(cols,x,y,t):
	lenx = np.array(len_x(x))
	print(lenx.shape)

	# x = np.mat(x)
	# y = np.mat(y)
	# t = np.mat(t)
	x = sample(x,cols)
	y = sample(y,cols)
	t = sample(t,cols)
	#移动偏移量
	delt_x = x[:,1:] -x[:,:-1]#对于水平移动,计算垂直方向上的偏移量,
	delt_y = y[:,1:] -y[:,:-1]#,对于垂直移动,计算水平偏移量。
	delt_t = t[:,1:] -t[:,:-1]

	move_time = t[:,-1] - t[:,0] #移动持续时间#
	#一次移动的持续时间是一次移动中首尾两个数据点的时间之差。
	

	delt_xy = np.multiply(delt_x,delt_x) + np.multiply(delt_y,delt_y)#效果不好！！
	delt_xy =  np.sqrt(np.abs(delt_xy))
	#对于斜线方向移动的处理也需要先进行坐标变换
	speed_x = np.divide(delt_x ,delt_t,out=np.zeros_like(delt_x),where=delt_t!=0)
	speed_y = np.divide(delt_y ,delt_t,out=np.zeros_like(delt_y),where=delt_t!=0)
	delt_speed_x = speed_x[:,1:] -speed_x[:,:-1]
	delt_speed_y = speed_y[:,1:] -speed_y[:,:-1]
	delt_speed_t = delt_t[:,1:] -delt_t[:,:-1]
	acc_speed_x = np.divide(delt_speed_x ,delt_speed_t,out=np.zeros_like(delt_speed_x),where=delt_speed_t!=0) 
	acc_speed_y = np.divide(delt_speed_y ,delt_speed_t,out=np.zeros_like(delt_speed_y),where=delt_speed_t!=0) 
	feature1 = np.hstack((delt_xy,delt_x,delt_y,delt_t,speed_x,speed_y,delt_speed_x,delt_speed_y,acc_speed_x,acc_speed_y))
	feature1 = np.hstack((feature1,lenx))
	print("feature1 shape:",feature1.shape)
	return feature1
def divide(delt_x,delt_t):
	return np.divide(delt_x ,delt_t,out=np.zeros_like(delt_x),where=delt_t!=0)
def feature3(cols,x,y,t):
	lenx = np.array(len_x(x))
	print(lenx.shape)

	# x = np.mat(x)
	# y = np.mat(y)
	# t = np.mat(t)
	x = sample(x,cols)
	y = sample(y,cols)
	t = sample(t,cols)
	#移动偏移量
	delt_x = x[:,1:] -x[:,:-1]#对于水平移动,计算垂直方向上的偏移量,
	delt_y = y[:,1:] -y[:,:-1]#,对于垂直移动,计算水平偏移量。
	delt_t = t[:,1:] -t[:,:-1]

	move_time = t[:,-1] - t[:,0] #移动持续时间#
	#一次移动的持续时间是一次移动中首尾两个数据点的时间之差。
	

	delt_xy = np.multiply(delt_x,delt_x) + np.multiply(delt_y,delt_y)#效果不好！！
	delt_xy =  np.sqrt(np.abs(delt_xy))
	#对于斜线方向移动的处理也需要先进行坐标变换
	speed_x = np.divide(delt_x ,delt_t,out=np.zeros_like(delt_x),where=delt_t!=0)
	speed_y = np.divide(delt_y ,delt_t,out=np.zeros_like(delt_y),where=delt_t!=0)
	delt_speed_x = speed_x[:,1:] -speed_x[:,:-1]
	delt_speed_y = speed_y[:,1:] -speed_y[:,:-1]
	delt_speed_t = delt_t[:,1:] -delt_t[:,:-1]
	acc_speed_x = np.divide(delt_speed_x ,delt_speed_t,out=np.zeros_like(delt_speed_x),where=delt_speed_t!=0) 
	acc_speed_y = np.divide(delt_speed_y ,delt_speed_t,out=np.zeros_like(delt_speed_y),where=delt_speed_t!=0) 
	delt_delt_speed_x =  delt_speed_x[:,1:] - delt_speed_x[:,:-1]
	delt_delt_speed_y =  delt_speed_y[:,1:] - delt_speed_y[:,:-1]
	delt_delt_speed_t =  delt_speed_t[:,1:] - delt_speed_t[:,:-1]
	accc_speed_x = divide(delt_delt_speed_x ,delt_delt_speed_t) 
	accc_speed_y = divide(delt_delt_speed_x ,delt_delt_speed_t) 
	feature1 = np.hstack((delt_xy,delt_x,delt_y,delt_t,
						speed_x,speed_y,
						#delt_speed_x,delt_speed_y,
						acc_speed_x,acc_speed_y,
						#  accc_speed_x,accc_speed_y
						 ))
	#feature1 = np.hstack((feature1,lenx))
	print("feature1 shape:",feature1.shape)
	return feature1
####提取特征
'''考虑当前坐标与目标坐标的位置
当前位置减去目标位置 作为新的坐标

实验证明 feature1 2 不影响结果
'''
def feature2(cols,x,y,t,x_d,y_d,b):
	lenx = len_x(x)
	x = sample(x,cols)
	y = sample(y,cols)
	t = sample(t,cols)
	x_d = reshape(x_d,cols)
	y_d = reshape(y_d,cols)

	x = x - x_d
	y = y - y_d
	delt_x = x[:,1:] -x[:,:-1]
	delt_y = y[:,1:] -y[:,:-1]
	delt_t = t[:,1:] -t[:,:-1]
	speed_x = np.divide(delt_x ,delt_t,out=np.zeros_like(delt_x),where=delt_t!=0)
	speed_y = np.divide(delt_y ,delt_t,out=np.zeros_like(delt_y),where=delt_t!=0)
	delt_speed_x = speed_x[:,1:] -speed_x[:,:-1]
	delt_speed_y = speed_y[:,1:] -speed_y[:,:-1]
	delt_speed_t = delt_t[:,1:] -delt_y[:,:-1]
	acc_speed_x = np.divide(delt_speed_x ,delt_speed_t,out=np.zeros_like(delt_speed_x),where=delt_speed_t!=0) 
	acc_speed_y = np.divide(delt_speed_y ,delt_speed_t,out=np.zeros_like(delt_speed_y),where=delt_speed_t!=0) 
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	speed_x= imp.fit_transform(speed_x)
	speed_x = preprocessing.scale(speed_x) 
	speed_y= imp.fit_transform(speed_y)
	speed_y = preprocessing.scale(speed_y)
	acc_speed_x= imp.fit_transform(acc_speed_x)
	acc_speed_x = preprocessing.scale(acc_speed_x) 
	acc_speed_y= imp.fit_transform(acc_speed_y)
	acc_speed_y = preprocessing.scale(acc_speed_y)
	lenx = preprocessing.scale(lenx) 



	feature2 = np.hstack((b*speed_x,(1-b)*speed_y,b*acc_speed_x,(1-b)*acc_speed_y)) 
	feature2 = np.hstack((feature2,lenx))
	print("feature2 shape:",feature2.shape)
	return feature2


def readtest():
	####################读入文件  返回的是x y t x_d y_d label id  list
	test_path = "data/dsjtzs_txfz_test1.txt"
	id = []
	a2 = []
	a3 = []
	with open(test_path,"r") as f:
		for line in f:
			line = line.split()
			id.append(line.pop(0))
			a2.append(line.pop(0))
			a3.append(line.pop(0))
	xyt = [[]]
	x,y,t = [],[],[]
	for i in range(len(a2)):
		xyt = a2[i].split(";")
		xyt.pop(-1)
		x.append([])
		y.append([])
		t.append([])
		for j in range(len(xyt)):
			x[i].append(float(xyt[j].split(",").pop(0)))
			y[i].append(float(xyt[j].split(",").pop(1)))
			t[i].append(float(xyt[j].split(",").pop(2)))


	xy = [[]]
	x_d,y_d = [] ,[]
	for i in range(len(a3)):
		xy = a3[i].split(",")
		x_d.append(float(xy[0]))
		y_d.append(float(xy[1]))

	#########id lable np.array##################
	id = np.array(id).astype(int)
	return id,x,y,t,x_d,y_d
def submission():
	train_id,train_x,train_y,train_t,train_x_d,train_y_d,train_label = readtrain()
	test_id,test_x,test_y,test_t,test_x_d,test_y_d = readtest()
	################# train ######################
	cols=5  #选择要选择的列数 #重要参数！！！3 最好
	train_feature = feature3(cols,train_x,train_y,train_t)
	test_feature = feature3(cols,test_x,test_y,test_t)
	#feature = feature2(cols,x,y,t,x_d,y_d,0.7)
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	train_feature= imp.fit_transform(train_feature)
	test_feature= imp.fit_transform(test_feature)
	train_feature = (train_feature-np.mean(train_feature)) / np.std(train_feature)
	test_feature = (test_feature-np.mean(test_feature)) / np.std(test_feature)
	test_y_pre = adaval(train_feature, test_feature, train_label)
	print("negative number:",len(test_id) - test_y_pre.sum()) 
	with open("feature3_15_ada_100.txt",'w') as su:
		for i in range(len(test_y_pre)):
			if test_y_pre[i] ==0 :
				su.writelines(str(test_id[i]))
				su.writelines("\n")
			else: 
				pass
def adaval(train_x, val_x, train_y):
	from sklearn import svm 
	from sklearn import linear_model 
	#clf = svm.SVC()
	clf = linear_model.SGDClassifier()
	clf = AdaBoostClassifier(n_estimators=100)
	clf.fit(train_x,train_y)
	
	val_y_pre = clf.predict(val_x)
	return val_y_pre

if __name__ == '__main__':
	train_val()
	submission()