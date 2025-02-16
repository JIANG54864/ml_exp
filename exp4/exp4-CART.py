import pandas as pd
import random

def CulGini(data: pd.DataFrame, forecast_label: str) -> float:
    """
    计算数据集关于预测标签的基尼值（基尼不纯度）。

    参数:
    data (pd.DataFrame): 包含特征和标签的数据集。
    forecast_label (str): 预测标签的列名。

    返回值:
    float: 基尼值。
    """
    total = data.shape[0]
    kinds = data[forecast_label].value_counts()
    gini = 1.0
    for count in kinds.values:
        prior_probability = count / total
        gini -= prior_probability ** 2
    return gini

def GiniIndex(data: pd.DataFrame, label: str, forecast_label: str) -> float:
    """
    计算给定数据集的Gini指数。

    该函数通过分组计算每个类别的Gini指数，并根据类别的比例加权求和，最终得到整个数据集的Gini指数。

    参数:
    data (pd.DataFrame): 包含标签和预测标签的数据集。
    label (str): 数据集中用于分组的标签列名。
    forecast_label (str): 数据集中用于计算Gini指数的预测标签列名。

    返回值:
    float: 整个数据集的Gini指数。
    """
    gain = 0
    # 提取需要的列，减少计算量
    sub_frame = data[[label, forecast_label]]
    # 按标签列进行分组
    group = sub_frame.groupby(label)
    # 遍历每个分组，计算加权Gini指数
    for key, df in group:
        gain += (df.shape[0] / data.shape[0]) * CulGini(df, forecast_label)
    return gain

def find_best_split(data: pd.DataFrame, feature: str, label: str) -> float:
    """
    寻找连续特征的最佳分割阈值（分类任务版本）

    参数:
    data: 包含特征和目标的数据集
    feature: 待分割特征名
    label: 目标变量列名

    返回:
    float: 最优分割阈值
    """
    values = sorted(data[feature].unique())
    best_gini = float('inf')
    best_threshold = values[0]

    for i in range(1, len(values)):
        threshold = (values[i-1] + values[i]) / 2
        left = data[data[feature] <= threshold]
        right = data[data[feature] > threshold]

        if len(left) == 0 or len(right) == 0:
            continue

        # 计算加权基尼指
        total = len(left) + len(right)
        gini = (len(left)/total)*CulGini(left, label) + (len(right)/total)*CulGini(right, label)

        if gini < best_gini:
            best_gini = gini
            best_threshold = threshold

    return best_threshold

def should_stop_split(data: pd.DataFrame, label: str, min_samples=2, max_depth=5) -> bool:
    """
    判断是否停止分裂（预剪枝条件）

    参数:
    data: 当前节点数据集
    label: 目标变量列名
    min_samples: 最小样本数阈值
    max_depth: 最大树深度

    返回:
    bool: 是否停止分裂
    """
    # 样本数不足或达到最大深度
    if len(data) < min_samples or max_depth <= 1:
        return True

    # 当前节点纯度足够高
    if CulGini(data, label) < 0.1:  # 可调阈值
        return True

    return False


def createmyCartTree(data, label, depth=5):
    # 停止条件分支统一返回字典
    if should_stop_split(data, label, max_depth=depth):
        return {'class': data[label].mode()[0]}

    if len(data[label].unique()) == 1:
        return {'class': data[label].iloc[0]}

    if len(data.columns) == 1:
        return {'class': data[label].mode()[0]}

    min_gini = float('inf')
    best_feature = None

    for feature in data.columns.drop(label):
        current_gini = GiniIndex(data, label, feature)
        if current_gini < min_gini:
            min_gini = current_gini
            best_feature = feature

    if best_feature is None:
        return {'class': data[label].mode()[0]}

    best_threshold = find_best_split(data, best_feature, label)

    # 递归调用必须确保返回字典
    left_data = data[data[best_feature] <= best_threshold].drop(columns=[best_feature])
    right_data = data[data[best_feature] > best_threshold].drop(columns=[best_feature])

    return {
        best_feature: {
            'threshold': best_threshold,
            'left': createmyCartTree(left_data, label, depth - 1),  # 注意深度递减
            'right': createmyCartTree(right_data, label, depth - 1)
        }
    }


def decision(tree: dict, sample: pd.Series):
    # 增加类型检查确保安全
    if not isinstance(tree, dict):
        return tree  # 冗余保护

    if 'class' in tree:
        return tree['class']

    # 确保非叶节点有完整结构
    feature = list(tree.keys())[0]
    node = tree[feature]

    # 处理缺失特征值的情况
    if feature not in sample:
        return tree.get('class', sample['brand'].mode()[0])

    # 执行阈值判断
    if sample[feature] <= node['threshold']:
        return decision(node['left'], sample)
    else:
        return decision(node['right'], sample)

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
    CARTtree = createmyCartTree(train_set,'brand')
    # 初始化统计参数
    T2=0
    N2=0
    for i in range(test_set.shape[0]):
        vector = test_set.iloc[i, :-1]
        trueLabel = test_set.iloc[i]['brand']
        forecastLabel1 = decision(CARTtree, vector)
        if forecastLabel1 == trueLabel:
            T2 += 1
            print(f'({i + 1}) 真实品牌：{trueLabel}, 预测品牌：{forecastLabel1}, 正确')
        else:
            N2 += 1
            print(f'({i + 1}) 真实品牌：{trueLabel}, 预测品牌：{forecastLabel1}, 错误')

    print(f'CART预测准确率为:' + str(T2 / (T2 + N2)))