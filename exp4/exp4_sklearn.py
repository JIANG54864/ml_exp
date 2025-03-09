from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# 加载数据
data = pd.read_csv('input/cars.csv')

# 这个库中使用的是CART算法，不需要对数据进行离散化处理
# 数据预处理
X = data.drop(columns=['brand'])  # 特征
y = data['brand']  # 标签

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树分类器
clf = DecisionTreeClassifier(max_depth=5, random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 评估准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"决策树预测准确率为: {accuracy:.2f}")
