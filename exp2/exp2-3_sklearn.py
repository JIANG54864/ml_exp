import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 读取CSV文件中的数据
with open(r'input/diagnosis_result.csv') as file:
    reader = csv.DictReader(file)
    datas = [row for row in reader]

# 获取除了 'id' 和 'diagnosis_result' 之外的特征名称
feature_keys = [col for col in reader.fieldnames if col not in ['id', 'diagnosis_result']]

# 提取特征和标签
X = [[float(data[key]) for key in feature_keys] for data in datas]
y = [data['diagnosis_result'] for data in datas]

# 转换为numpy数组
X = np.array(X)
y = np.array(y)

# 数据归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

# 初始化KNN分类器
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# 训练模型
knn.fit(X_train, y_train)

# 预测测试集
y_pred = knn.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"正确率: {accuracy:.4f}")
