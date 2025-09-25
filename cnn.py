import torch
import os
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm

# 设置全局随机种子
SEED = 42  # 可以设置为任意整数
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # 如果使用多GPU
np.random.seed(SEED)
random.seed(SEED)
# 确保卷积操作的确定性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 加载外部数据
def get_data(type="train"):
    Timelen = 740  # 每个样本有740个时间点
    file = './' + type
    if type == "train":  # 如果读取训练集数据，则有年龄和性别标签
        Ywight = 3
    else:
        Ywight = 1
    datanames = os.listdir(file + '/')  # 遍历文件夹中的所有文件
    Samplesize = len(datanames)  # 计算文件数的大小
    print(datetime.now().strftime('%H:%M:%S'), ',Import dataset：', type)
    try:  # 若有封装好的数据，直接加载
        X = np.load(type + '_X.npy', allow_pickle=True)
        Y = np.load(type + '_Y.npy', allow_pickle=True)
    except:  # 做没有，则逐个加载被试的数据
        X = []  # 创建空集用于存储数据
        Y = np.array([[0] * Ywight] * Samplesize, dtype=np.float32)
        Y1 = pd.read_csv(file + '_subjs.csv', header=0)
        # 以下循环将2维数据拉成1维数据，并编排对应的被试号
        for i in range(Samplesize):
            Temp = pd.read_csv(file + '/' + datanames[i], header=None)  # 读取单个被试数据
            Temp2 = Temp.values
            print('Import X:' + str(i) + '/' + str(Samplesize))
            if Temp2.shape[1] >= Timelen:  # 数据比200*740更大
                X.append(Temp2[:, :Timelen])
                YTemp = list(Y1.ID).index(datanames[i][:-4])  # 找到被试号对应的位置
                Y[i, 0] = datanames[i][4:-4]  # 文件名前4个字符是sub_，后四个是.csv
                if type == 'train':  # 如果是训练集，有性别和年龄的标签
                    Y[i, 1] = Y1.sex[YTemp]
                    Y[i, 2] = Y1.age[YTemp]
        X = np.array(X)
        Y = Y[(Y != 0).any(axis=1)]
        np.save(type + '_X.npy', X)
        np.save(type + '_Y.npy', Y)
    # 将X，Y作为函数运行的输出内容
    return X, Y


# 调用前面定义的函数，加载训练集数据
X, Y = get_data(type="train")
X = X.astype(np.float32)

# 产生随机数
spilt = int(X.shape[0] * 0.9)  # 样本量,划分训练集和验证集的比例
Randomindex = np.arange(X.shape[0])  # 产生和样本量一样长的序列
np.random.seed(SEED)  # 设置numpy随机种子
Randomindex = np.random.permutation(Randomindex)  # 打乱序列


# 利用torch的DataLoader类创建自己的数据集
class myDataset(Dataset):
    def __init__(self, X, Y, type):
        if type == 'train':
            Randomindex2 = Randomindex[:spilt]  # 提取前90%的特征值
            X = X[Randomindex2, :, :]  # 提取前90%的特征值
            Y = Y[Randomindex2, 1]  # 取第1列性别变量作分类任务
        elif type == 'val':
            Randomindex2 = Randomindex[spilt:]  # 提取后10%的特征值
            X = X[Randomindex2, :, :]  # 提取后10%的特征值
            Y = Y[Randomindex2, 1]  # 取第1列性别变量作分类任务
        elif type == 'test':
            X = X
            Y = Y
        self.X = torch.from_numpy(X)  # 转成张量
        self.Y = torch.from_numpy(Y)  # 转成张量

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = (self.X[idx], self.Y[idx])
        return data


# Hyper-parameters
n_channel1 = 50  # 第一层卷积层的通道数量
n_channel2 = 100  # 第二层卷积层的通道数量
learning_rate = 0.001  # 梯度下降的学习率
batch_size = 20  # 每次训练用的样本数
input_size = X.shape[1]  # 一个样本的通道数
num_classes = 2  # 分类数为1时,输出连续变量,用于回归,若大于1,输出分类变量,用于分类.
num_epochs = 20  # 训练代数

# 实例化对象
# 将数据集导入DataLoader，进行shuffle以及选取batch_size
# 使用固定的随机种子生成器
generator = torch.Generator().manual_seed(SEED)
Trainset = DataLoader(myDataset(X, Y, 'train'), batch_size=batch_size,
                      shuffle=True, num_workers=0, generator=generator)
Valset = DataLoader(myDataset(X, Y, 'val'), batch_size=batch_size,
                    shuffle=True, num_workers=0, generator=generator)


class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        # 定义layer1层
        self.layer1 = nn.Sequential(
            # 一维卷积，通道数为X.shape[1]，输出通道数为n_channel1，卷积核大小6，卷积步长为2，不足处补1
            nn.Conv1d(X.shape[1], n_channel1, kernel_size=6, stride=2, padding=1),
            nn.BatchNorm1d(n_channel1),  # 一维标准化
            nn.ReLU())  # ReLu激活
        # 定义layer2层
        self.layer2 = nn.Sequential(
            # 一维卷积，通道数为n_channel1，输出通道数为n_channel2，卷积核大小6，卷积步长为2，不足处补1
            nn.Conv1d(n_channel1, n_channel2, kernel_size=6, stride=2, padding=1),
            nn.BatchNorm1d(n_channel2),  # 一维标准化
            nn.ReLU())  # ReLu激活
        # 定义fc层，全连接层，通道数需要手动计算
        self.fc = nn.Linear(18300, num_classes)

    def forward(self, x):
        out = self.layer1(x)  # 第一层为layer1
        out = self.layer2(out)  # 第二层为layer2
        out = out.reshape(out.size(0), -1)  # 拉成一维向量
        out = self.fc(out)  # 进行全连接
        return out  # 返回输出值


# 读取device 看有无cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 将model放入device（cuda或cpu）
model = ConvNet(num_classes).to(device)

# 损失函数为交叉熵函数CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
# 优化器为Adam
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    running_loss = 0.0
    # 以进度条形式显示训练进度
    with tqdm(Trainset, unit="batch") as tepoch:
        for images, labels in tepoch:
            # 显示进度条
            tepoch.set_description(f"Epoch {epoch + 1}")
            # 将特征值和标签转到device中
            images = images.to(device)
            labels = labels.to(device)
            # 前馈，计算损失
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            # 反馈，优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # 输出本次迭代结果、计算正确率
            _, predicted = torch.max(outputs.data, 1)  # 获得预测结果
            total = labels.size(0)  # 当前batch中的样本总数
            correct = (predicted == labels).sum().item()  # 正确预测的样本数
            accuracy = 100 * correct / total  # 计算准确率
            tepoch.set_postfix(loss=loss.item(), accuracy=f'{accuracy:.2f}%')

# 在训练集上查看模型正确率
with torch.no_grad():  # 不更新梯度，即不学习新知识
    correct, total = 0, 0
    for images, labels in Trainset:
        # 将特征值和标签转到device中
        images = images.to(device)
        labels = labels.to(device)
        # 输出预测值
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # 判断预测值是否正确
        total += labels.size()[0]
        correct += (predicted == labels).sum().item()
    train_acc = round(100 * correct / total, 2)
    print('Accuracy of the network on the train set: {}%'.format(train_acc))

# 在验证集集上查看模型正确率
with torch.no_grad():  # 不更新梯度，即不学习新知识
    correct, total = 0, 0
    for images, labels in Valset:
        # 将特征值和标签转到device中
        images = images.to(device)
        labels = labels.to(device)
        # 输出预测值
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # 判断预测值是否正确
        total += labels.size()[0]
        correct += (predicted == labels).sum().item()
    test_acc = round(100 * correct / total, 2)
print('Accuracy of the network on the test set: {}%'.format(test_acc))

f = open('./result_cnn_fenlei.txt', "a", encoding="utf-8")
f.write('\n' +
        '/SEED/' + str(SEED) +
        '/learning_rate/' + str(learning_rate) +
        '/batch_size/' + str(batch_size) +
        '/n_channel1/' + str(n_channel1) +
        '/n_channel2/' + str(n_channel2) +
        '/train_acc/' + str(train_acc) +
        '/test_acc/' + str(test_acc))

f.close()
print('/SEED/' + str(SEED) +
      '/learning_rate/' + str(learning_rate) +
      '/batch_size/' + str(batch_size) +
      '/n_channel1/' + str(n_channel1) +
      '/n_channel2/' + str(n_channel2) +
      '/train_acc/' + str(train_acc) +
      '/test_acc/' + str(test_acc))
