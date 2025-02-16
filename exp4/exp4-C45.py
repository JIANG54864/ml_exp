import pandas as pd
import numpy as np
import random

def CulEntropy(data: pd.DataFrame, forecast_label: str) -> float:
    """
    计算给定数据集中某一列的熵值。

    参数:
    data (pd.DataFrame): 包含目标列的数据集。
    forecast_label (str): 需要计算熵值的列名。

    返回值:
    float: 计算得到的熵值。
    """
    total = data.shape[0]
    kinds = data[forecast_label].value_counts()
    Entropy = 0.0
    for count in kinds:
        prior_probability = count / total
        Entropy += prior_probability * np.log2(prior_probability)
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

def CulIntrinsicInfo(data: pd.DataFrame, label: str) -> float:
    """
    计算给定数据集中某个标签的固有信息量（Intrinsic Information）。

    固有信息量是信息论中的一个概念，用于衡量数据集在某个标签上的不确定性。
    该函数通过计算每个类别的先验概率，并使用这些概率来计算固有信息量。

    参数:
    data (pd.DataFrame): 包含标签的数据集。
    label (str): 需要计算固有信息量的标签列名。

    返回值:
    float: 计算得到的固有信息量。
    """
    total = data.shape[0]  # 数据集的总样本数
    kinds = data[label].value_counts()  # 获取标签列中每个类别的计数
    intrinsic_info = 0.0  # 初始化固有信息量为0

    # 遍历每个类别的计数，计算固有信息量
    for count in kinds:  # 直接遍历计数值
        prior_probability = count / total  # 计算当前类别的先验概率
        safe_prob = np.where(prior_probability == 0, 1, prior_probability)  # 避免概率为0的情况
        intrinsic_info -= prior_probability * np.log2(safe_prob)  # 累加固有信息量

    return intrinsic_info


def GainRatio(data: pd.DataFrame, label: str, forecast_label: str) -> float:
    """
    计算某个特征关于预测标签的信息增益率（Gain Ratio）。

    参数:
    data (pd.DataFrame): 包含特征和标签的数据集。
    label (str): 特征列名。
    forecast_label (str): 预测标签的列名。

    返回值:
    float: 信息增益率值。
    """
    info_gain = InfoGain(data, label, forecast_label)
    intrinsic_info = CulIntrinsicInfo(data, label)
    if intrinsic_info == 0:
        return 0
    return info_gain / intrinsic_info

def createmyC45Tree(data: pd.DataFrame) -> dict:
    """
    递归创建决策树（C4.5算法）。

    参数:
    data (pd.DataFrame): 包含特征和标签的数据集。

    返回值:
    dict: 表示决策树的字典结构。
    """
    # 如果当前分支下的实例只有一种分类，则返回该分类
    if len(data['brand'].value_counts()) == 1:
        return data['brand'].iloc[0]

    # 计算所有特征的信息增益
    info_gains = {}
    for column in data.columns:
        if column != 'brand':
            info_gains[column] = InfoGain(data, column, 'brand')

    # 计算信息增益的平均值
    avg_info_gain = np.mean(list(info_gains.values()))

    # 筛选出信息增益高于平均值的特征
    candidate_features = [feature for feature, gain in info_gains.items() if gain > avg_info_gain]

    # 如果没有特征满足条件，则返回数据集中数量最多的类别
    if not candidate_features:
        valueCount = data['brand'].value_counts()
        return valueCount.index[0]

    # 从候选特征中选择增益率最高的特征
    bestGainRatio = 0
    bestFeature = None

    for feature in candidate_features:
        gain_ratio = GainRatio(data, feature, 'brand')
        if gain_ratio > bestGainRatio:
            bestGainRatio = gain_ratio
            bestFeature = feature

    # 如果所有特征的增益率都为0，则返回数据集中数量最多的类别
    if bestFeature is None:
        valueCount = data['brand'].value_counts()
        return valueCount.index[0]

    # 构建当前节点的子树
    myTree = {bestFeature: {}}
    valueList = set(data[bestFeature])  # 获取划分特征中所有唯一取值
    for value in valueList:
        # 对每个取值，筛选出 data 中 bestFeature 等于 value 的子数据集。递归地为该子数据集构建子树
        myTree[bestFeature][value] = createmyC45Tree(data[data[bestFeature] == value])
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
    if isinstance(tree, dict):  # 确保 tree 是字典才进行键访问
        firstFeature = next(iter(tree))
        childTree = tree[firstFeature]
        forecastLabel = None  # 初始化为 None，避免默认值为0
        for key in childTree.keys():
            # 根据特征值选择子树
            if testVector[firstFeature] == key:
                # 如果子树是分支节点，则递归预测
                if isinstance(childTree[key], dict):
                    forecastLabel = decision(childTree[key], testVector)
                # 如果子树是叶节点，则直接返回类别
                else:
                    forecastLabel = childTree[key]
                break  # 找到匹配的键后跳出循环
        # 如果没有找到匹配的键，返回训练集中数量最多的类别
        if forecastLabel is None:
            forecastLabel = train_set['brand'].value_counts().idxmax()
        return forecastLabel
    else:
        return tree

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
    # 分割训练集与测试集
    train_set, test_set = splitdata(Data)
    # 训练决策树
    C45tree = createmyC45Tree(train_set)
    # 初始化统计参数
    T1 = 0
    N1 = 0
    for i in range(test_set.shape[0]):
        vector = test_set.iloc[i, :-1]
        trueLabel = test_set.iloc[i]['brand']
        forecastLabel1 = decision(C45tree, vector)
        if forecastLabel1 == trueLabel:
            T1 += 1
            print(f'({i + 1}) 真实品牌：{trueLabel}, 预测品牌：{forecastLabel1}, 正确')
        else:
            N1 += 1
            print(f'({i + 1}) 真实品牌：{trueLabel}, 预测品牌：{forecastLabel1}, 错误')

    print(f'C4.5预测准确率为:' + str(T1 / (T1 + N1)))
