from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import r2_score
from datetime import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
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
    try:
        X = np.load(type + '_X.npy', allow_pickle=True)
        Y = np.load(type + '_Y.npy', allow_pickle=True)
    except:
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
        np.save(type + '_X.npy', X)
        np.save(type + '_Y.npy', Y)
    #将X，Y作为函数运行的输出内容
    return X,Y

#调用前面定义的函数，加载训练集数据
X,Y = get_data(type = "train")
#Hyper-parameters
hidden_size1=1000          ##第一层全连接层的神经元数
hidden_size2=600           ##第二层全连接层的神经元数
learning_rate=0.005        ##梯度下降的学习率
batch_size=40              ##每次训练用的样本数
input_size = X.shape[1]    ##一个样本的特征数
num_classes = 1            ##分类数为1时,输出连续变量,用于回归,若大于1,输出分类变量,用于分类.
num_epochs = 20            ##训练代数
spilt = int(X.shape[0]*0.9)##样本量,划分训练集和验证集的比例
#产生随机数,用于划分训练集和验证集
Randomindex = np.arange(X.shape[0])
Randomindex = np.random.permutation(Randomindex)

#创建自己的数据集
class myDataset(Dataset):
    def __init__(self,X,Y,type):
        if type=='train':
            Randomindex2 = Randomindex[:spilt]
            X = X[Randomindex2, :]
            Y = Y[Randomindex2, 2]  #取第2列年龄变量作回归任务
        elif type =='val':
            Randomindex2 = Randomindex[spilt:]
            X = X[Randomindex2,:]
            Y = Y[Randomindex2, 2]  #取第2列年龄变量作回归任务
        elif type =='test':
            X = X
            Y = Y
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
    def __len__(self):
        return len(self.X)
    def __getitem__ (self,idx):
        data = (self.X[idx],self.Y[idx])
        return data
#实例化对象
#将数据集导入DataLoader，进行shuffle以及选取batch_size
#Windows里num_works只能为0，其他值会报错
#训练集
Trainset = DataLoader(myDataset(X,Y,'train'),batch_size=batch_size,shuffle=True,num_workers=0)
#验证集
Valset = DataLoader(myDataset(X,Y,'val'),batch_size=batch_size,shuffle=True,num_workers=0)

##定义模型，其中input_size为每个样本的特征值长度，
# hidden_size1为第一个全连接层的神经元数量，
# hidden_size2为第二个全连接层的神经元数量，
# num_classes 为标签数（年龄：连续变量），这里为1，输出一个连续变量
class NeuralNet(nn.Module):
    def __init__(self,input_size, hidden_size1,hidden_size2,num_classes):
        super(NeuralNet,self).__init__()
        ##input_size=200*740, hidden_size1=1000, 特征值148000转为1000个神经元上的特征
        self.fc1 = nn.Linear(input_size, hidden_size1)
        ##hidden_size1=1000, hidden_size2=600, 特征值1000转为600个神经元上的特征
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        ##hidden_size2=600, num_classes=1, 特征值600转为1输出（年龄）
        self.fc3 = nn.Linear(hidden_size2, num_classes)  # 输出层
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
##识别有无cuda（GPU），若没有则在CPU计算
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##将model放入device（cuda或cpu）
model = NeuralNet(input_size, hidden_size1, hidden_size2, num_classes).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
# Train the model
total_step = len(Trainset)
for i in range(num_epochs):  # 循环
    for j, (images, labels) in enumerate(Trainset):
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        images.requires_grad = True  # 自动求导
        # 输出值
        out = model(images)
        # 损失值
        loss = criterion(out.squeeze(1), labels)  # squeeze的用法主要就是对数据的维度进行压缩或者解压。
        optimizer.zero_grad()  # 清空过往梯度
        loss.backward()
        # 更新参数
        optimizer.step()  # 根据梯度更新网络参数
        # 对比预测年龄和实际年龄
        labels2 = labels.cpu().numpy()        # 将标签转为向量
        outputs2 = out.cpu().detach().numpy() # 将预测结果转为向量
        r2 = r2_score(outputs2, labels2)      # 使用r2_score计算R²值

        if j % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.2f}, R^2: {:.2f}'
                  .format(i + 1, num_epochs, j + 1, total_step, loss.item(),round(r2, 2)))

# 告诉 PyTorch 不要在接下来的代码块中计算任何张量的梯度
# 将训练模型用于训练集，得到训练集的预测决定系数
with torch.no_grad() :
    train_label = []
    train_output = []
    for i, (images,labels) in enumerate (Trainset):
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        # 输出值
        outputs = model(images)
        # 积累预测年龄和实际年龄
        labels2 = labels.cpu().numpy()        # 将标签转为向量
        outputs2 = outputs.cpu().detach().numpy() # 将预测结果转为向量
        train_label = np.append(train_label, labels2)             #每个batch的预测结果拼接在一起
        train_output = np.append(train_output,outputs2.squeeze(1))#每个batch的标签拼接在一起
    # 计算预测年龄和实际年龄的回归决定系数R^2和均方误差MAE
    train_R2 = round(r2_score(train_label, train_output), 2)
    train_MAE = sum(abs(train_label - train_output)) / len(train_label)

# 将训练模型用于验证集，得到验证集的预测决定系数
with torch.no_grad():
    val_label = []
    val_output =  []
    for i,(images,labels) in enumerate (Valset):
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        # 输出值
        outputs = model(images)
        # 积累预测年龄和实际年龄
        labels2 = labels.cpu().numpy()        # 将标签转为向量
        outputs2 = outputs.cpu().detach().numpy() # 将预测结果转为向量

        val_label = np.append(val_label, labels2)             #每个batch的预测结果拼接在一起
        val_output = np.append(val_output,outputs2.squeeze(1))#每个batch的标签拼接在一起
    # 计算预测年龄和实际年龄的回归决定系数R^2和均方误差MAE
    val_R2 = round(r2_score(val_label, val_output), 4)
    val_MAE = sum(abs(val_label-val_output))/len(val_label)

print(  '/trainR2/'+str(train_R2)+
        '/testR2/'+str(val_R2))
f = open('./nn_R.txt',"a",encoding = "utf - 8")
f.write('/trainR2/'+str(train_R2)+
        '/testR2/'+str(val_R2))
f.close()

##将模型运用与测试集，输出预测的年龄（仅有预测结果）
X,Y = get_data(type = "test")
test_set = DataLoader(myDataset(X,Y,'test'),batch_size=batch_size,shuffle=False,num_workers=0)
with torch.no_grad() :
    test_label = []
    test_output = []
    for images,labels in test_set:
        images = images.reshape(-1, input_size).to(device) #将数据传至device（GPU）
        outputs = model(images)   #将数据输入到模型中

        labels2 = labels.cpu().numpy()        # 将标签转为向量
        outputs2 = outputs.cpu().detach().numpy() # 将预测结果转为向量

        test_label = np.append(test_label, labels2)             #每个batch的预测结果拼接在一起
        test_output = np.append(test_output,outputs2.squeeze(1))#每个batch的标签拼接在一起

data1 = np.array([test_label, test_output]).T
colNames = ['SubjNum', 'predict_age']
# 创建CSV文件写入器
test = pd.DataFrame(data=data1, index=None, columns=colNames)
test.to_csv('NN_predict_age.csv', encoding='utf-8')