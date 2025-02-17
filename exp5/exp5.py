import math
import pandas as pd

# 贝叶斯分类器
def BayesClassifier(x, train: pd.DataFrame):
    # 0表示遇难 1表示存活
    type0 = train[train['Survived'] == 0]
    type1 = train[train['Survived'] == 1]

    sum_type0 = type0.count().values[0]
    sum_type1 = type1.count().values[0]

    # 先验概率
    prior_0 = sum_type0 / (sum_type0 + sum_type1)
    prior_1 = sum_type1 / (sum_type0 + sum_type1)

    g0 = math.log(prior_0)
    g1 = math.log(prior_1)

    # 计算所有列的似然/类条件概率密度
    for column in train.columns:
        # 去除预测标签的影响
        if column != 'Survived':
            # 计算拉普拉斯平滑后的似然
            likelihood0 = (type0[type0[column] == x[column]].count().values[0] + 1) / (
                    sum_type0 + train[column].nunique())
            likelihood1 = (type1[type1[column] == x[column]].count().values[0] + 1) / (
                    sum_type1 + train[column].nunique())
            # 取对数处理
            ln_likelihood0 = math.log(likelihood0)
            ln_likelihood1 = math.log(likelihood1)

            g0 += ln_likelihood0
            g1 += ln_likelihood1

    if g0 >= g1:
        return 0
    else:
        return 1


# 填充缺失值
def processData(dataframe: pd.DataFrame):
    # print(dataframe.info())
    # 删除无用特征
    for name in dataframe.columns:
        unique = dataframe[name].unique().shape[0]
        full = dataframe[name].shape[0]
        # print({name}, '的不同取值个数：', unique, '/', full)
        if (unique == 1) | (unique == full):
            dataframe = dataframe.drop(name, axis=1)
            # print({name}, '被删除')
    dataframe = dataframe.drop(['Cabin', 'Ticket'], axis=1)
    # 填充缺失值
    missing = dataframe.isnull().any(axis=0)
    for index, value in missing.items():
        if value:
            # 用众数填充缺失值
            dataframe[index].fillna(dataframe[index].mode()[0], inplace=True)
    # 进行数值化替换方便处理
    mapper = {
        'Sex': {'male': 0,'female': 1},
        'Embarked': {'C': 0,'Q': 1,'S': 2,}
    }
    dataframe.replace(mapper, inplace=True)
    for col in ['Age', 'Fare']:
        dataframe[col] = pd.cut(dataframe[col],2, labels=[0, 1])  # 将数据分为2个等频区间


# 读取数据
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

# 数据预处理
processData(train)
processData(test)

Array = []
for i in range(5):
    # 随机抽样
    Array.append(train.sample(frac=1, replace=True, axis=0))

# 储存分类结果
isSurvived = []
for index, row in test.iterrows():
    t = []
    # 每一个测试数据放到5个子分类器中进行分类
    for i in range(5):
        t.append(BayesClassifier(row, Array[i]))
    isSurvived.append(t)

# 投票决定最终预测
vote = []
for item in isSurvived:
    SurvivedCount = str(item).count('1')
    if SurvivedCount > 2:
        vote.append(1)
    else:
        vote.append(0)
test.loc[:, 'forecast'] = vote

# 输出预测与实际数据的对比
TP = 0
TN = 0
FP = 0
FN = 0
for index, row in test.iterrows():
    if row['forecast'] == 1:
        if row['Survived'] == 1:
            print(str(index+1) + " 实际: 幸存, 预测: 幸存, 预测正确")
            TP += 1
        else:
            print(str(index+1) + " 实际: 遇难, 预测: 幸存, 预测错误")
            FP += 1
    else:
        if row['Survived'] == 0:
            print(str(index+1) + " 实际: 遇难, 预测: 遇难, 预测正确")
            TN += 1
        else:
            print(str(index+1) + " 实际: 幸存, 预测: 遇难, 预测错误")
            FN += 1

print('测试集上的精确度为:', (TP + TN) / (TP + TN + FP + FN))