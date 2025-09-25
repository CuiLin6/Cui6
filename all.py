# 郑大数据分析 - 多特征整合（每个文件有标签）
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import os

# 设置随机种子
random.seed(42)
np.random.seed(42)

# 1. 加载所有特征数据
feature_files = {
    'pca': 'sin_pca.xlsx',  # PCA主成分特征（1个主成分）
    'max': 'sin_max.xlsx',  # 最大值特征
    'std': 'sin_sta.xlsx',  # 标准差特征
    'stable': 'sin.xlsx'  # 稳定值特征
}

# 2. 加载所有特征文件
feature_dfs = {}
for name, file in feature_files.items():
    if os.path.exists(file):
        df = pd.read_excel(file)
        feature_dfs[name] = df
        print(f"已加载特征: {name}, 形状: {df.shape}")
        print(f"特征列: {df.columns.tolist()}")
    else:
        print(f"错误: 未找到文件 {file}")
        exit()

# 3. 验证所有文件有相同的样本数和标签
# 获取第一个文件的标签
first_key = list(feature_dfs.keys())[0]
reference_labels = feature_dfs[first_key]['species']

# 验证所有文件有相同的标签
for name, df in feature_dfs.items():
    # 验证样本数量相同
    if len(df) != len(reference_labels):
        print(f"错误: {name}特征文件有不同数量的样本 ({len(df)} vs {len(reference_labels)})")
        exit()

    # 验证标签相同
    if not (df['species'] == reference_labels).all():
        print(f"警告: {name}特征文件的标签不完全匹配参考标签")
        # 强制使用参考标签（如果标签不一致）
        df['species'] = reference_labels

# 4. 提取并合并特征
all_features = pd.DataFrame()

# 添加标签列
all_features['species'] = reference_labels

# 添加每个特征
for name, df in feature_dfs.items():
    # 提取特征列（排除标签列）
    feature_columns = [col for col in df.columns if col != 'species']

    # 添加特征到总数据集
    for col in feature_columns:
        # 如果特征列名已经存在，添加前缀
        if col in all_features.columns:
            new_col = f"{name}_{col}"
            all_features[new_col] = df[col]
        else:
            all_features[col] = df[col]

print("\n整合后的特征矩阵:")
print(all_features.head())
print(f"特征矩阵形状: {all_features.shape}")
print(f"特征列: {all_features.columns.tolist()}")

# 5. 数据集划分
train, test = train_test_split(all_features, test_size=0.3, random_state=42, stratify=all_features['species'])
print("\n训练集形状:", train.shape)
print("测试集形状:", test.shape)

# 6. 准备训练和测试数据
# 提取特征列（排除标签列）
feature_columns = [col for col in all_features.columns if col != 'species']

train_X = train[feature_columns]
train_y = train['species']
test_X = test[feature_columns]
test_y = test['species']

# 7. 特征标准化
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.transform(test_X)

# 8. 模型定义
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "DT": DecisionTreeClassifier(random_state=42),
    "LR": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": svm.SVC(probability=True, random_state=42)
}

# 9. 训练和评估模型
results = {}
print("\n模型评估结果:")

for name, model in models.items():
    # 训练模型
    model.fit(train_X_scaled, train_y)

    # 预测
    y_pred = model.predict(test_X_scaled)

    # 评估
    acc = metrics.accuracy_score(test_y, y_pred)
    f1 = metrics.f1_score(test_y, y_pred, average='weighted')
    report = classification_report(test_y, y_pred)

    # 存储结果
    results[name] = {
        'accuracy': acc,
        'f1_score': f1,
        'classification_report': report
    }

    # 打印结果
    print(f"\n{name}模型:")
    print(f"准确率: {acc:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("分类报告:")
    print(report)

    # 混淆矩阵
    cm = confusion_matrix(test_y, y_pred)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{name}模型混淆矩阵')
    plt.colorbar()
    classes = np.unique(np.concatenate([test_y, y_pred]))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name}.png', dpi=300)
    plt.close()

# 10. 模型性能比较
# 准备比较数据
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
f1_scores = [results[name]['f1_score'] for name in model_names]

# 准确率比较
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'purple'])
plt.title('模型准确率比较')
plt.ylabel('准确率')
plt.ylim(0, 1.05)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.02, f'{acc:.4f}', ha='center', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.savefig('model_accuracy_comparison.png', dpi=300)
plt.close()

# F1分数比较
plt.figure(figsize=(10, 6))
plt.bar(model_names, f1_scores, color=['blue', 'green', 'red', 'purple'])
plt.title('模型F1分数比较')
plt.ylabel('F1分数')
plt.ylim(0, 1.05)
for i, f1 in enumerate(f1_scores):
    plt.text(i, f1 + 0.02, f'{f1:.4f}', ha='center', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.savefig('model_f1_comparison.png', dpi=300)
plt.close()

# 11. KNN参数优化（只针对KNN模型）
if 'KNN' in models:
    print("\nKNN参数优化:")
    a_index = list(range(1, 21))
    a_values = []

    plt.figure(figsize=(12, 6))
    for k in a_index:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_X_scaled, train_y)
        y_pred = knn.predict(test_X_scaled)
        acc = metrics.accuracy_score(test_y, y_pred)
        a_values.append(acc)
        print(f"K={k} 准确率: {acc:.4f}")

    # 可视化KNN参数优化结果
    plt.plot(a_index, a_values, marker='o')
    plt.title('KNN不同K值的准确率')
    plt.xlabel('K值')
    plt.ylabel('准确率')
    plt.xticks(a_index)
    plt.grid(True)
    plt.savefig('knn_optimization.png', dpi=300)
    plt.close()

# 12. 特征重要性分析（针对决策树模型）
if 'DT' in models:
    dt_model = models['DT']
    dt_importances = dt_model.feature_importances_
    indices = np.argsort(dt_importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title('决策树特征重要性')
    plt.bar(range(len(feature_columns)), dt_importances[indices], align='center')
    plt.xticks(range(len(feature_columns)), [feature_columns[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance_dt.png', dpi=300)
    plt.close()

# 13. 逻辑回归系数分析（针对LR模型）
if 'LR' in models:
    lr_model = models['LR']
    # 逻辑回归可能有多个类别的系数，这里取第一个（如果是二分类）
    if len(lr_model.coef_) > 1:
        lr_coef = np.abs(lr_model.coef_[0])
    else:
        lr_coef = np.abs(lr_model.coef_)

    indices = np.argsort(lr_coef)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title('逻辑回归特征系数（绝对值）')
    plt.bar(range(len(feature_columns)), lr_coef[indices], align='center')
    plt.xticks(range(len(feature_columns)), [feature_columns[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_importance_lr.png', dpi=300)
    plt.close()

print("\n分析完成! 所有结果已保存为图片文件。")
