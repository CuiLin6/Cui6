###导入算法
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# 加载四个特征文件
pca_feature = pd.read_excel('sin_pca.xlsx')
max_feature = pd.read_excel('sin_max.xlsx')
std_feature = pd.read_excel('sin_sta.xlsx')
stable_feature = pd.read_excel('sin.xlsx')

# 合并四个特征
combined_features = pd.concat([
    pca_feature['species'].rename('PCA'),  # PCA主成分
    max_feature.drop(columns=['species']),  # 最大值特征
    std_feature.drop(columns=['species']),  # 标准差特征
    stable_feature.drop(columns=['species'])  # 稳定值特征
], axis=1)

# 添加标签列
combined_features['species'] = pca_feature['species']

# 将物种标签转换为数值编码
label_encoder = LabelEncoder()
combined_features['species'] = label_encoder.fit_transform(combined_features['species'])

# 设置随机种子
random.seed(42)
np.random.seed(42)

### 数据集划分
train, test = train_test_split(combined_features, test_size=0.3, random_state=42)
print(train.shape)
print(test.shape)

### 训练集和测试集
# 获取所有特征列名（排除species）
feature_cols = [col for col in combined_features.columns if col != 'species']

train_X = train[feature_cols]
train_y = train.species
test_X = test[feature_cols]
test_y = test.species

### SVM
model = svm.SVC()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of the SVM is:', metrics.accuracy_score(prediction, test_y))

### LR
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of the Logistic Regression is', metrics.accuracy_score(prediction, test_y))

### DT
model = DecisionTreeClassifier()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of the Decision Tree is', metrics.accuracy_score(prediction, test_y))

### KNN
model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('The accuracy of the KNN is', metrics.accuracy_score(prediction, test_y))

### KNN不同K值准确率
a_index = list(range(1, 11))
a = pd.Series(dtype='float64')
for i in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    a = pd.concat([a, pd.Series([metrics.accuracy_score(prediction, test_y)])])

plt.plot(range(1, 11), a.values)
plt.xticks(range(1, 11))
plt.show()
