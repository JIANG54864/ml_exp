import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('input/train.csv')
test_data = pd.read_csv('input/test.csv')

y = train_data["Survived"]
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
# 计算正确率
y_test = test_data["Survived"]
accuracy = (predictions == y_test).mean()
print(f"预测正确率: {accuracy:.2%}")