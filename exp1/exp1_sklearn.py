import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

from exp1 import preprocessing

# 加载并预处理训练集
train_data = pd.read_csv('input/train.csv')  # 假设训练集文件名为 train_set.csv
train_data = preprocessing(train_data)

# 加载并预处理测试集
test_data = pd.read_csv('input/test.csv')  # 假设测试集文件名为 test_set.csv
test_data = preprocessing(test_data)

# 此模型只支持数值类数据
lst = ['total_loan', 'interest', 'region', 'post_code', 'monthly_payment',
       'debt_loan_ratio', 'scoring_low', 'scoring_high', 'known_outstanding_loan',
       'recircle_b', 'recircle_u', 'f0', 'f2', 'f3', 'f4', 'early_return_amount',
       'early_return_amount_3mon']

# 划分特征和标签（训练集）
X_train = train_data[lst]
y_train = train_data['isDefault']

# 划分特征和标签（测试集）
X_test = test_data[lst]
y_test = test_data['isDefault']

# 初始化并训练高斯朴素贝叶斯分类器
model = GaussianNB()
model.fit(X_train, y_train)

# 预测并评估
y_pred = model.predict(X_test)
print("分类报告：")
print(classification_report(y_test, y_pred))
print("准确率：", accuracy_score(y_test, y_pred))

