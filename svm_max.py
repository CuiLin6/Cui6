#郑大数据
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


#load data
#path: the path of iris data in your laptop
sin = pd.read_excel('sin_max.xlsx')
sin.head()

sin.info()

random.seed(42)
np.random.seed(42)

###数据集划分
train, test = train_test_split(sin, test_size = 0.3,random_state=42)
print(train.shape)
print(test.shape)

###训练集和测试集
train_X = train[['TGS2600','TGS2612','TGS2611','TGS2610','TGS2602','TGS2602#','TGS2620','TGS2620#']]
train_y=train.species
test_X= test[['TGS2600','TGS2612','TGS2611','TGS2610','TGS2602','TGS2602#','TGS2620','TGS2620#']]
test_y =test.species

### SVM
model = svm.SVC()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))

### LR
model = LogisticRegression()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_y))

### DT
model=DecisionTreeClassifier()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))

### KNN
model=KNeighborsClassifier(n_neighbors=3)
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y))

a_index=list(range(1,11))
# 使用pd.concat()
a = pd.Series(dtype='float64')
for i in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X, train_y)
    prediction = model.predict(test_X)
    # 使用pd.concat()
    a = pd.concat([a, pd.Series([metrics.accuracy_score(prediction, test_y)])])

plt.plot(range(1, 11), a.values)
plt.xticks(range(1, 11))



