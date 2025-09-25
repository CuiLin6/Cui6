#码农
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime
from sklearn.metrics import r2_score
import pandas as pd
import os

def get_data(type = "train"):
    Timelen = 740  #每个样本有740个时间点
    ROI=200       #每个样本有200个ROI
    file='./'+type
    if type=="train":  ##如果读取训练集数据，则有年龄和性别标签，即第2-3列
        Ywight = 3
    else:
        Ywight = 1
    datanames = os.listdir(file+'/') ##遍历训练集或测试集文件夹中的所有文件，文件数=被试数
    Samplesize = len(datanames) #计算文件数的大小
    print(datetime.now().strftime('%H:%M:%S'), ',Import dataset：',type)
    X = np.array([[0] * ROI * Timelen] * Samplesize, dtype=np.float32)  ##创建空集用于
    Y = np.array([[0] * Ywight] * Samplesize, dtype=np.float32)
    Y1 = pd.read_csv(file+'_subjs.csv', header=0)
    ##以下循环将2维数据拉成1维数据，并编排对应的被试号
    for i in range(Samplesize):
      Temp = pd.read_csv(file +'/'+ datanames[i], header=None)
      Temp2 = Temp.values
      print('Import X:' + str(i) + '/' + str(Samplesize))
      if Temp2.shape[1] >= Timelen: ## 数据比200*740更大
          Temp3 = Temp2[:, :Timelen].reshape((1, -1))      ##将原始的200ROI×740个时间点的二维数据拉成1×148000的一维数据
          X[i, :] = Temp3
          YTemp = list(Y1.ID).index(datanames[i][:-4]) ##找到被试号对应的位置
          Y[i, 0] = datanames[i][4:-4]  ##文件名前4个字符是sub_，后四个是.csv, 如sub_2.csv。因此用4：-4可以调用被试号，
          if type =='train':  #如果是测试集，则没有正确标签，仅输出预测结果。本次演练中不涉及测试集
              Y[i, 1] = Y1.sex[YTemp] #如果是训练集，文件中有性别和年龄的标签
              Y[i, 2] = Y1.age[YTemp]
    #将X，Y作为函数运行的输出内容
    return X,Y

#加载并划分数据为训练集和验证集
X,Y = get_data(type = "train")
Randomindex = np.arange(X.shape[0])
Randomindex = np.random.permutation(Randomindex)

#生成随机数,将随机的前90%的数据归为训练集,后10%的数据归为验证集
spilt = int(X.shape[0]*0.9)
X_train = X[Randomindex[:spilt],:]
Y_train = Y[Randomindex[:spilt],2] #第二列是性别标签
X_val = X[Randomindex[spilt:],:]
Y_val = Y[Randomindex[spilt:],2]   #第二列是性别标签
#将X,Y置空 可释放部分内存
X,Y=None,None

TimeStr1 = datetime.now().strftime("%m-%d-%H-%M")
#设置模型参数
kernel='linear'
C=0.1
random_state = np.random.RandomState(0)
classifierSVR = make_pipeline(StandardScaler(), svm.SVR(kernel=kernel,C=C))

#训练模型
time_1=datetime.now()
print('SVR start:',time_1)
classifierSVR.fit(X_train, Y_train)

#将训练模型预测训练集的年龄，并计算相关系数，按理说，训练集的正确率会比较高
Y_train_score = classifierSVR.predict(X_train) #用模型预测训练集的年龄
trainr2 = r2_score(Y_train_score, Y_train) #用r2_score函数计算模型预测训练集的年龄和真实年龄的决定系数
train_MAE = sum(abs(Y_train_score - Y_train)) / len(Y_train) #计算训练集预测的MAE

#将训练模型预测验证集的年龄，并计算相关系数
Y_score = classifierSVR.predict(X_val)  #用模型预测验证集的年龄
valr2 = r2_score(Y_score, Y_val)    #计算验证集的预测年龄和真实年龄的决定系数
val_MAE = sum(abs(Y_score - Y_val)) / len(Y_val)  #计算验证集预测的MAE

print('SVR finish:', datetime.now().strftime("%m-%d-%H-%M"),
      ',train_R2=', str(round(trainr2, 4)),',val_R2=',  str(round(valr2, 4)))

#输出训练和验证结果到txt文件
time_2=datetime.now()
timedif = (time_2-time_1).seconds
f = open('./svm_r.txt', "a", encoding="utf - 8")
f.write('/trainR2/' + str(round(trainr2, 4))+
        '/val_R2/' + str(round(valr2, 4))+
        '/train_MAE/'+str(train_MAE)+
        '/val_MAE/'+str(val_MAE)+
        '/timedif/'+str(timedif))
f.close()
print(  '/trainR2/' + str(round(trainr2, 4))+
        '/val_R2/' + str(round(valr2, 4))+
        '/train_MAE/' + str(train_MAE) +
        '/val_MAE/' + str(val_MAE) +
        '/timedif/'+str(timedif))

##输出测试集结果
test_X,test_Y = get_data(type = "test")  ##读取测试集数据
predicted = classifierSVR.predict(test_X)

data1 = np.array([np.squeeze(test_Y),predicted]).T
colNames = ['SubjNum', 'predict_age']
# 创建CSV文件写入器
test = pd.DataFrame(data=data1, index=None, columns=colNames)
test.to_csv('predict_age.csv', encoding='utf-8')