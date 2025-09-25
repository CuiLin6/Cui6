#码农
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime
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

#调用前面定义的函数，加载训练集数据
X,Y = get_data(type = "train")
Randomindex = np.arange(X.shape[0])
Randomindex = np.random.permutation(Randomindex)

#生成随机数,将随机的前90%的数据归为训练集,后10%的数据归为验证集
spilt = int(X.shape[0]*0.9)
X_train = X[Randomindex[:spilt],:]
Y_train = Y[Randomindex[:spilt],1]  #第一列是性别标签
X_val = X[Randomindex[spilt:],:]
Y_val = Y[Randomindex[spilt:],1]    #第一列是性别标签
#将X,Y置空 可释放部分内存
X,Y=None,None

## 计算当下时间，用于计算运行时间。
TimeStr1 = datetime.now().strftime("%m-%d-%H-%M")
#设置模型参数
kernel='poly'
degree=3
coef0=0
C=1
random_state = np.random.RandomState(0)
classifierSVM = make_pipeline(StandardScaler(), svm.SVC(kernel=kernel,
                                                        probability=True,
                                                        coef0=coef0,
                                                        degree=degree,
                                                        C=C,
                                                        random_state=random_state))
#训练模型
time_1=datetime.now()
print('SVM start:',datetime.now().strftime("%m-%d-%H-%M"))
classifierSVM.fit(X_train, Y_train)

#将训练模型预测训练集的性别，并计算正确率，按理说，训练集的正确率会比较高
Y_train_score = classifierSVM.decision_function(X_train)
Y_train_score_ACC = np.arange(len(Y_train_score))
for i in range(len(Y_train_score)):### 由于预测的结果可能是任意实数，所以将某一范围的数值划为男性，另外的划为女性。
    if Y_train_score[i]>0:         ### 这里将预测数值大于0的，划为男性，小于0的为女性
        Y_train_score_ACC[i] = 1
    else:
        Y_train_score_ACC[i] = 0
    if Y_train_score_ACC[i] == Y_train[i]:  ###比较预测的训练集标签是否与真实标签一致。
        Y_train_score_ACC[i] = 1
    else:
        Y_train_score_ACC[i] = 0
train_ACC = round(np.sum(Y_train_score_ACC, axis=0)/Y_train_score_ACC.shape[0],4)

#将训练模型预测验证集的性别，并计算正确率，如果验证集正确率与训练集正确率差不多，模型的性能会比较稳定。
Y_score = classifierSVM.decision_function(X_val)
Y_score_ACC = np.arange(len(Y_score))
for i in range(len(Y_score)):###同上
    if Y_score[i]>0:         ###同上
        Y_score_ACC[i] = 1
    else:
        Y_score_ACC[i] = 0
    if Y_score_ACC[i] == Y_val[i]: ###同上
        Y_score_ACC[i] = 1
    else:
        Y_score_ACC[i] = 0
val_ACC = round(np.sum(Y_score_ACC, axis=0)/Y_score_ACC.shape[0],4)
print('SVM finish:',datetime.now().strftime("%m-%d-%H-%M"), ',val_ACC:',str(val_ACC),
                ',train_ACC:',str(train_ACC))

#输出保存到txt文件
time_2 = datetime.now()
timedif = (time_2 - time_1).seconds
f = open('./svm_c.txt', "a", encoding="utf - 8")
f.write(
        '/train_ACC=' + str(train_ACC)+
        '/val_ACC=' + str(val_ACC)+
        '/timedif='+str(timedif))
f.close()
print(
        '/train_ACC=' + str(train_ACC) +
        '/val_ACC=' + str(val_ACC) +
        '/timedif=' + str(timedif))

##输出测试结果
test_X,test_Y = get_data(type = "test")  ##读取测试集数据
predicted = classifierSVM.predict(test_X)
for i in range(len(predicted)):
    if predicted[i]>0:
        predicted[i] = 1
    else:
        predicted[i] = 0
data1 = np.array([np.squeeze(test_Y),predicted]).T
colNames = ['SubjNum', 'predict_sex']
# 创建CSV文件写入器
test = pd.DataFrame(data=data1, index=None, columns=colNames)
test.to_csv('predict_sex.csv', encoding='utf-8')