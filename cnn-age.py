#码农
import torch
import os
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import r2_score

def get_data(type = "train"):
    Timelen = 740  #每个样本有740个时间点
    file='./'+type
    if type=="train":  ##如果读取训练集数据，则有年龄和性别标签，即第2-3列
        Ywight = 3
    else:
        Ywight = 1
    datanames = os.listdir(file+'/') ##遍历训练集或测试集文件夹中的所有文件，文件数=被试数
    Samplesize = len(datanames) #计算文件数的大小
    print(datetime.now().strftime('%H:%M:%S'), ',Import dataset：',type)
    try: #若有封装好的数据，直接加载
        X = np.load(type + '_X.npy', allow_pickle=True)
        Y = np.load(type + '_Y.npy', allow_pickle=True)
    except: #做没有，则逐个加载被试的数据
        X = []  ##创建空集用于
        Y = np.array([[0] * Ywight] * Samplesize, dtype=np.float32)
        Y1 = pd.read_csv(file+'_subjs.csv', header=0)
        ##以下循环将2维数据拉成1维数据，并编排对应的被试号
        for i in range(Samplesize):
          Temp = pd.read_csv(file +'/'+ datanames[i], header=None) #读取单个被试数据
          Temp2 = Temp.values
          print('Import X:' + str(i) + '/' + str(Samplesize))
          if Temp2.shape[1] >= Timelen: ## 数据比200*740更大
              X.append(Temp2[:, :Timelen])
              YTemp = list(Y1.ID).index(datanames[i][:-4]) ##找到被试号对应的位置
              Y[i, 0] = datanames[i][4:-4]  ##文件名前4个字符是sub_，后四个是.csv, 如sub_2.csv。因此用4：-4可以调用被试号，
              if type =='train':  #如果是测试集，则没有正确标签，仅输出预测结果。本次演练中不涉及测试集
                  Y[i, 1] = Y1.sex[YTemp] #如果是训练集，文件中有性别和年龄的标签
                  Y[i, 2] = Y1.age[YTemp]
        X = np.array(X)
        Y = Y[(Y != 0).any(axis=1)]
        np.save(type + '_X.npy', X)
        np.save(type + '_Y.npy', Y)
    #将X，Y作为函数运行的输出内容
    return X,Y

#调用前面定义的函数，加载训练集数据
X,Y = get_data(type = "train")
X = X.astype(np.float32)

#产生随机数
spilt = int(X.shape[0]*0.9)  ##样本量,划分训练集和验证集的比例
Randomindex = np.arange(X.shape[0]) # 产生和样本量一样长的序列
Randomindex = np.random.permutation(Randomindex)  #打乱序列

#利用torch的DataLoader类创建自己的数据集
class myDataset(Dataset):
    def __init__(self,X,Y,type):
        if type=='train':
            Randomindex2 = Randomindex[:spilt]  # 提取前90%的特征值
            X = X[Randomindex2, :,:] # 提取前90%的特征值
            Y = Y[Randomindex2, 2]  # 取第2列年龄变量作回归任务
        elif type =='val':
            Randomindex2 = Randomindex[spilt:]# 提取前90%的特征值
            X = X[Randomindex2,:,:]# 提取前90%的特征值
            Y = Y[Randomindex2, 2]  # 取第2列年龄变量作回归任务
        elif type =='test':
            X = X
            Y = Y
        self.X = torch.from_numpy(X)#转成张量
        self.Y = torch.from_numpy(Y)#转成张量
    def __len__(self):
        return len(self.X)
    def __getitem__ (self,idx):
        data = (self.X[idx],self.Y[idx])
        return data


#Hyper-parameters
n_channel1 = 50              ##第一层卷积层的通道数量
n_channel2 = 100             ##第二层卷积层的通道数量
learning_rate=0.001          ##梯度下降的学习率
batch_size=20                ##每次训练用的样本数
input_size = X.shape[1]      ##一个样本的通道数
num_classes = 1              ##分类数为1时,输出连续变量,用于回归,若大于1,输出分类变量,用于分类.
num_epochs = 20              ##训练代数

#实例化对象
#将数据集导入DataLoader，进行shuffle以及选取batch_size
Trainset = DataLoader(myDataset(X,Y,'train'),batch_size=batch_size,shuffle=True,num_workers=0)
Valset = DataLoader(myDataset(X,Y,'val'),batch_size=batch_size,shuffle=True,num_workers=0)
#Windows里num_works只能为0，其他值会报错


class ConvNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ConvNet, self).__init__()
        # 定义layer1层
        self.layer1 = nn.Sequential(
            # 一维卷积，通道数为X.shape[1]，输出通道数为n_channel1，卷积核大小6，卷积步长为2
            nn.Conv1d(X.shape[1], n_channel1, kernel_size=6, stride=2),
            nn.BatchNorm1d(n_channel1),  # 一维标准化
            nn.ReLU(),  # ReLu激活
            nn.MaxPool1d(kernel_size=6, stride=1))
        # 定义layer2层
        self.layer2 = nn.Sequential(
            # 一维卷积，通道数为n_channel1，输出通道数为n_channel2，卷积核大小6，卷积步长为2
            nn.Conv1d(n_channel1, n_channel2, kernel_size=6, stride=2),
            nn.BatchNorm1d(n_channel2),  # 一维标准化
            nn.ReLU(),  # ReLu激活
            nn.MaxPool1d(kernel_size=6, stride=1))
        # 定义fc层，全连接层，通道数需要手动计算
        self.fc = nn.Linear(17400, num_classes)
    def forward(self, x):
        out = self.layer1(x) #第一层为layer1
        out = self.layer2(out)#第二层为layer2
        out = out.reshape(out.size(0), -1) #拉成一维向量
        out = self.fc(out) #进行全连接
        return out #返回输出值

# 读取device 看有无cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#将model放入device（cuda或cpu）
model = ConvNet(num_classes).to(device)

# 损失函数为交叉熵函数CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
# 优化器为Adam
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):  # 循环
    running_loss = 0.0
    #以进度条形式显示训练进度
    with tqdm(Trainset, unit="batch") as tepoch:
        for images, labels in tepoch:
            # 显示进度条
            tepoch.set_description(f"Epoch {epoch + 1}")
            # 将特征值和标签转到device中
            images = images.to(device)
            labels = labels.to(device)
            # 前馈，计算损失
            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)

            # 反馈，优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # 输出本次迭代结果
            labels2 = labels.cpu().numpy()
            outputs2 = outputs.cpu().detach().numpy()
            # 计算决定系数
            rsquare = round(r2_score(labels2, outputs2),3)  # 计算决定系数
            tepoch.set_postfix(loss=loss.item(),rsquare=f'{rsquare:.2f}')

# 在训练集上查看模型性能
with torch.no_grad() :#不更新梯度，即不学习新知识
    train_label = []
    train_output = []
    for images,labels in Trainset:
        # 将特征值和标签转到device中
        images = images.to(device)
        labels = labels.to(device)
        # 输出预测值
        outputs = model(images).squeeze(1)
        train_label = np.append(train_label, labels.cpu().numpy())
        train_output = np.append(train_output, (outputs.cpu().detach().numpy()))
    #计算整体决定系数
    train_R2 = round(r2_score(train_label, train_output), 3)
# 在验证集集上查看模型性能
with torch.no_grad():
    val_label = []
    val_output =  []
    for i,(images,labels) in enumerate (Valset):
        # 将特征值和标签转到device中
        images = images.to(device)
        labels = labels.to(device)
        # 输出预测值
        outputs = model(images).squeeze(1)
        val_label = np.append(val_label, labels.cpu().numpy())
        val_output = np.append(val_output, (outputs.cpu().detach().numpy()))
    # 计算整体决定系数
    test_R2 = round(r2_score(val_label, val_output), 3)

f = open('./torch_cnn_huigui.txt',"a",encoding = "utf - 8")
f.write('\n'+
        '/learning_rate='+str(learning_rate)+
        '/batch_size='+str(batch_size)+
        '/n_channel1='+str(n_channel1)+
        '/n_channel2=' + str(n_channel2) +
        '/trainR2/' + str(train_R2) +
        '/testR2/' + str(test_R2))
f.close()
print(
        '/learning_rate='+str(learning_rate)+
        '/batch_size='+str(batch_size)+
        '/n_channel1='+str(n_channel1)+
        '/n_channel2=' + str(n_channel2) +
        '/trainR2/' + str(train_R2) +
        '/testR2/' + str(test_R2))