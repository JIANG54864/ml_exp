import pandas as pd


# 数据预处理
def preprocessing(data):
    # 将取值唯一、取值各不相同的特征删除
    for name in data.columns:
        unique = data[name].unique().shape[0]
        full = data[name].shape[0]
        if (unique == 1) | (unique == full):
            data = data.drop(name, axis=1)
    # 查看缺失值
    # print(data.isnull().sum(axis=0).sort_values(ascending=False))
    # 使用中位数填充空值
    # 匿名特征f0-f4，为一些贷款人行为计数特征的处理
    data['f0'].fillna(int(data['f0'].median()), inplace=True)
    data['f1'].fillna(int(data['f1'].median()), inplace=True)
    data['f2'].fillna(int(data['f2'].median()), inplace=True)
    data['f3'].fillna(int(data['f3'].median()), inplace=True)
    data['f4'].fillna(int(data['f4'].median()), inplace=True)
    data['work_year'].fillna('0', inplace=True)
    # 删除无关属性
    drop_elements=['issue_date', 'earlies_credit_mon', 'title']
    for ele in drop_elements:
        data.drop(ele, axis=1, inplace=True)
        # axis=1：表示操作方向为列（axis=0 表示行）
    discrete(data)
    return data

# 分箱
def discrete(data):
    mydata = pd.read_csv('input/train.csv')
    discreteness = ['total_loan', 'interest', 'region', 'post_code', 'monthly_payment', 'debt_loan_ratio', 'scoring_low', 'scoring_high', 'known_outstanding_loan', 'recircle_b', 'recircle_u', 'f0', 'f2', 'f3', 'f4', 'early_return_amount', 'early_return_amount_3mon']
    # 通过describe方法获取分箱的依据，并且对特殊情况进行特殊处理
    for columnName in discreteness:
        temp = mydata[columnName].describe()
        temp1 = data[columnName].describe()
        if temp['min'] != temp['25%']:# 相等的话则前25%的数据点都具有相同的值
            dis = pd.cut(data[columnName], bins=[temp1['min'], temp['25%'], temp['50%'], temp['75%'], temp1['max'] + 1], right=False, include_lowest=True, labels=[0, 1, 2, 3])
        elif temp['25%'] != temp['50%']:# 如果第25百分位数（temp['25%']）不等于中位数（temp['50%']），则说明数据在较低部分有一定的分散性，但中间部分相对集中
            dis = pd.cut(data[columnName], bins=[temp1['min'], temp['50%'], temp['75%'], temp1['max'] + 1], right=False, include_lowest=True, labels=[0, 1, 2])
        else:
            dis = pd.cut(data[columnName], bins=[temp1['min'], temp['75%'], temp1['max'] + 1], right=False, include_lowest=True, labels=[0, 1])

        data[columnName] = dis
    return


# 在train.csv上训练模型
def training(total_map, positive_map):
    data = pd.read_csv('input/train.csv')
    data = preprocessing(data)
    train(data, total_map, positive_map)


# 在test.csv上检测
def test(total_map, positive_map):
    testdata = pd.read_csv('input/test.csv')
    testdata = preprocessing(testdata)
    testdata.drop(testdata['isDefault'], inplace=True)
    lst = predict(testdata, total_map, positive_map)
    results = showerror(testdata, lst)
    return results


# 根据属性排序并统计
def train(data, alldata, positives):
    for column in data.columns:
        datamap = {}
        # 统计每一列数据中各个值的出现次数，并按索引升序排序
        line = data[column].value_counts().sort_index(ascending=True)
        for index in line.index:
            datamap[index] = line[index]
        alldata[column] = datamap
    positive_data = data.loc[data['isDefault'] == 0]# isDefault贷款是否违约（预测标签）
    for column in positive_data.columns:
        datamap = {}
        line = positive_data[column].value_counts().sort_index(ascending=True)
        for index in line.index:
            datamap[index] = line[index]
        positives[column] = datamap


# 通过贝叶斯公式进行预测
def predict(data, alldata, positives):
    """
    基于朴素贝叶斯分类器预测样本类别

    参数:
        data (pd.DataFrame): 待预测的样本数据，每行代表一个样本，每列代表一个特征
        alldata (dict): 全局统计数据，包含每个特征的值分布和类别样本数。
                        isDefault键对应各类别样本数，如alldata['isDefault'][0]表示类别0的样本数
        positives (dict): 各特征在类别y=0下的条件计数，键为特征名，值为字典{特征值: 出现次数}

    返回值:
        list: 预测结果列表，元素为0或1，表示每个样本的预测类别
    """
    lista = []
    # 计算先验概率：P(y=0) 和 P(y=1)
    total_samples = alldata['isDefault'][0] + alldata['isDefault'][1]
    p_y0 = alldata['isDefault'][0] / total_samples
    p_y1 = alldata['isDefault'][1] / total_samples

    # 遍历每个待预测样本
    for index in data.index:
        likelihood_y0 = 1.0
        likelihood_y1 = 1.0

        # 计算每个特征的条件概率（应用拉普拉斯平滑）
        for column in data.columns:
            value = data.loc[index, column]
            unique_count = len(alldata[column].keys())

            # 计算P(x_i|y=0)：特征值在不违约类中的条件概率
            positive_num = positives[column].get(value, 0)
            prob_y0 = (positive_num + 1) / (alldata['isDefault'][0] + unique_count)
            likelihood_y0 *= prob_y0

            # 计算P(x_i|y=1)：特征值在违约类中的条件概率
            global_num = alldata[column].get(value, 0)
            negative_num = global_num - positive_num
            prob_y1 = (negative_num + 1) / (alldata['isDefault'][1] + unique_count)
            likelihood_y1 *= prob_y1

        # 计算归一化后的后验概率
        numerator_y0 = p_y0 * likelihood_y0
        numerator_y1 = p_y1 * likelihood_y1
        denominator = numerator_y0 + numerator_y1
        # 处理分母为零的特殊情况
        posterior_y0 = numerator_y0 / denominator if denominator != 0 else 0.5

        # 根据后验概率生成预测结果
        lista.append(0 if posterior_y0 > 0.5 else 1)

    return lista


def predict_sample(data, alldata, positives):
    # 初始化预测结果列表
    lista = []

    # 计算基础不违约概率，这里假设所有客户的违约概率相同
    probability_0 = alldata['isDefault'][0] / (alldata['isDefault'][0]+alldata['isDefault'][1])

    # 遍历数据集中的每一行数据
    for index in data.index:
        # 初始化当前客户的违约概率为基础不违约概率
        probability = probability_0

        # 遍历数据集中的每一列数据，计算每一特征对违约概率的影响
        for column in data.columns:
            # 获取当前特征在所有客户中的出现次数，未出现则默认为0
            all_num = alldata[column].get(data.loc[index, column], 0)
            # 获取当前特征在不违约客户中的出现次数，未出现则默认为0
            positive_num = positives[column].get(data.loc[index, column], 0)

            # 拉普拉斯平滑（Laplace Smoothing）：避免零概率问题。当某个特征值在目标类别中未出现（positive_num=0）时，分子加1保证概率不为零，同时分母对每个可能的特征值都加1次虚拟计数
            probability = probability * ((positive_num + 1) / (all_num + len(data[column].unique()))) / probability_0

        # 根据计算出的违约概率决定预测结果
        if probability > 0.5:
            lista.append(0)  # 预测为违约
        else:
            lista.append(1)  # 预测为不违约

    # 返回预测结果列表
    return lista

# 对比预测结果和实际结果
def showerror(testdata, lst):
    # 初始化结果统计字典
    results = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

    # 遍历测试数据集的索引
    for index in testdata.index:
        # 从预测结果列表中取出一个预测值
        predict1 = lst.pop(0)
        # 根据索引获取真实结果
        real = testdata.loc[index, 'isDefault']

        # 根据预测值和真实值比较，更新统计结果
        if real == 0 and predict1 ==0:
            results['TP']+=1
        elif real == 0 and predict1 ==1:
            results['FP']+=1
        elif real== 1 and predict1==1:
            results['TN']+=1
        elif real ==1  and predict1 == 0:
            results['FN']+=1

    # 返回统计结果
    return results

if __name__ == '__main__':
    total = {}
    positive = {}
    training(total, positive)
    resultsSet = test(total, positive)
    print(resultsSet)
    print('预测正确率:' + str((resultsSet['TP'] + resultsSet['TN']) / (
            resultsSet['TP'] + resultsSet['TN'] + resultsSet['FP'] + resultsSet['FN'])))
    print('查准率:' + str((resultsSet['TP'] / (resultsSet['TP'] + resultsSet['FP']))))
    print('查全率:' + str(resultsSet['TP'] / (resultsSet['TP'] + resultsSet['FN'])))
