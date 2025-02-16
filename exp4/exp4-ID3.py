import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sb

def CulEntropy(data: pd.DataFrame, forecast_label: str) -> float:
    """
    计算数据集关于预测标签的信息熵。

    参数:
    data (pd.DataFrame): 包含特征和标签的数据集。
    forecast_label (str): 预测标签的列名。

    返回值:
    float: 信息熵值。
    """
    total = data.shape[0]
    kinds = data[forecast_label].value_counts()
    Entropy = 0.0
    for i in range(kinds.shape[0]):
        prior_probability = kinds[i] / total
        Entropy += (prior_probability * np.log2(prior_probability)) #信息熵的公式
    return -Entropy

def InfoGain(data: pd.DataFrame, label: str, forecast_label: str) -> float:
    """
    计算某个特征关于预测标签的信息增益。

    参数:
    data (pd.DataFrame): 包含特征和标签的数据集。
    label (str): 特征列名。
    forecast_label (str): 预测标签的列名。

    返回值:
    float: 信息增益值。
    """
    total_entropy = CulEntropy(data, forecast_label)
    gain = total_entropy
    sub_frame = data[[label, forecast_label]] #选择指定特征列 label 和标签列 'brand'，生成一个新的子数据集
    group = sub_frame.groupby(label)
    for key, df in group:
        gain -= (df.shape[0] / data.shape[0]) * CulEntropy(df, 'brand')
        # 将子集的信息熵乘以子集大小与总数据集大小的比例，然后从总信息熵 gain 中减去该值，更新信息增益
    return gain

def createmyID3Tree(data: pd.DataFrame) -> dict:
    """
    递归创建决策树。

    参数:
    data (pd.DataFrame): 包含特征和标签的数据集。

    返回值:
    dict: 表示决策树的字典结构。
    """
    # 如果当前分支下的实例只有一种分类，则返回该分类
    if len(data['brand'].value_counts()) == 1:
        return data['brand'].iloc[0]

    bestGain = 0
    bestFeature = -1

    # 遍历所有特征，选择信息增益最大的特征作为当前节点的划分特征
    for column in data: #按列遍历
        if column != 'brand':
            gain = InfoGain(data, column, 'brand')
            if bestGain < gain:
                bestGain = gain
                bestFeature = column

    # 如果所有特征的信息增益都为0，则返回数据集中数量最多的类别
    if bestFeature == -1:
        valueCount = data['brand'].value_counts()
        return valueCount.index[0]

    # 构建当前节点的子树
    myTree = {bestFeature: {}} # 注意最佳特征可能有多个
    valueList = set(data[bestFeature]) # 获取划分特征中所有唯一取值
    for value in valueList:
        # 对每个取值，筛选出 data 中 bestFeature 等于 value 的子数据集。递归地为该子数据集构建子树
        myTree[bestFeature][value] = createmyID3Tree(data[data[bestFeature] == value])
    return myTree

def decision(tree: dict, testVector: pd.Series):
    """
    使用决策树对测试样本进行分类预测。

    参数:
    tree (dict): 决策树结构。
    testVector (pd.Series): 测试样本的特征向量。

    返回值:
    int: 预测的类别标签。
    """
    forecastLabel = 0
    firstFeature = next(iter(tree)) # 获取决策树中第一个特征
    childTree = tree[firstFeature]
    for key in childTree.keys():
        # 根据特征值选择子树
        if testVector[firstFeature] == key:
            # 如果子树是分支节点，则递归预测
            if type(childTree[key]) == dict:
                forecastLabel = decision(childTree[key], testVector)
            # 如果子树是叶节点，则直接返回类别
            else:
                forecastLabel = childTree[key]
    return forecastLabel
def splitdata(data: pd.DataFrame, random_seed: int = 66):
    """
    将数据集随机分割为训练集和测试集。

    参数:
    data (pd.DataFrame): 包含特征和标签的数据集。
    random_seed (int): 随机种子，确保结果可重复。

    返回值:
    tuple: 包含训练集和测试集的元组。
    """
    random.seed(random_seed)
    data_size = len(data)
    test_size = data_size // 5
    # 打乱索引
    shuffled_indices = random.sample(range(data_size), data_size)
    # 分割索引
    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]
    # 获取训练集和测试集
    trainSet = data.iloc[train_indices]
    testSet = data.iloc[test_indices]
    return trainSet, testSet

if __name__ == '__main__':
    Data = pd.read_csv('input/cars.csv')

    # sb.pairplot(Data.dropna(), hue='brand') # 数据可视化
    # plt.show()

    # 对数据进行离散化处理
    for col in ['mpg', 'year']:
        Data[col] = pd.qcut(Data[col], q=3, labels=[0, 1, 2])  # 将数据分为3个等频区间
    for col in ['cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60']:
        Data[col] = pd.qcut(Data[col], q=2, labels=[0, 1])  # 将数据分为2个等频区间

    # 分割训练集与测试集
    train_set, test_set = splitdata(Data)
    # 训练决策树
    ID3tree = createmyID3Tree(train_set)
    # 初始化统计参数
    T1 = 0
    N1 = 0
    for i in range(test_set.shape[0]):
        vector = test_set.iloc[i, :-1]
        trueLabel = test_set.iloc[i]['brand']
        forecastLabel1 = decision(ID3tree, vector)
        if forecastLabel1 == trueLabel:
            T1 += 1
            print(f'({i + 1}) 真实品牌：{trueLabel}, 预测品牌：{forecastLabel1}, 正确')
        else:
            N1 += 1
            print(f'({i + 1}) 真实品牌：{trueLabel}, 预测品牌：{forecastLabel1}, 错误')

    print(f'ID3预测准确率为:' + str(T1 / (T1 + N1)))
