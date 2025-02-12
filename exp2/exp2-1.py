import numpy as np
import pandas as pd


# 计算一个数据集的平均向量
def calculateAvg(vectors: pd.DataFrame):
    """
    计算数据集的平均向量。

    参数:
    vectors (pd.DataFrame): 包含向量数据的DataFrame。

    返回:
    np.array: 平均向量数组。
    """
    avg = pd.Series(index=vectors.columns, dtype=float)
    for column in vectors.columns:
        # 计算每个特征的平均值
        avg[column] = vectors[column].mean()
    return np.array(avg)


# 计算一个数据集的估计协方差矩阵
def calculateCov(vectors: pd.DataFrame):
    """
    计算数据集的协方差矩阵。

    参数:
    vectors (pd.DataFrame): 包含向量数据的DataFrame。

    返回:
    np.array: 协方差矩阵。
    """
    mu = np.matrix(calculateAvg(vectors)).T
    dimension = vectors.shape[1]
    Cov = np.zeros((dimension, dimension))
    for index, row in vectors.iterrows():
        xi = np.matrix(row).T
        diff = xi - mu
        Cov += diff * diff.T
    return Cov / vectors.shape[0]


# 创建数据帧
trainSet_1 = pd.read_csv('input/w1.csv')
trainSet_2 = pd.read_csv('input/w2.csv')

# (1) 计算并打印每个数据集的每个特征的平均值和协方差
print("(1): ")
print("类1：")
trainSet_1_x1 = trainSet_1['x1'].to_frame()
print("x1的最大似然估计:μ：" + str(calculateAvg(trainSet_1_x1)) + " 𝜎^2: " + str(calculateCov(trainSet_1_x1)))
trainSet_1_x2 = trainSet_1['x2'].to_frame()
print("x2的最大似然估计:μ：" + str(calculateAvg(trainSet_1_x2)) + " 𝜎^2: " + str(calculateCov(trainSet_1_x2)))
trainSet_1_x3 = trainSet_1['x3'].to_frame()
print("x3的最大似然估计:μ：" + str(calculateAvg(trainSet_1_x3)) + " 𝜎^2: " + str(calculateCov(trainSet_1_x3)))
print("类2：")
trainSet_2_x1 = trainSet_2['x1'].to_frame()
print("x1的最大似然估计:μ：" + str(calculateAvg(trainSet_2_x1)) + " 𝜎^2: " + str(calculateCov(trainSet_2_x1)))
trainSet_2_x2 = trainSet_2['x2'].to_frame()
print("x2的最大似然估计:μ：" + str(calculateAvg(trainSet_2_x2)) + " 𝜎^2: " + str(calculateCov(trainSet_2_x2)))
trainSet_2_x3 = trainSet_2['x3'].to_frame()
print("x3的最大似然估计:μ：" + str(calculateAvg(trainSet_2_x3)) + " 𝜎^2: " + str(calculateCov(trainSet_2_x3)))

# (2) 计算并打印每个数据集的每对特征的平均值和协方差矩阵
print("(2): ")
print("类1：")
trainSet_1_x1x2 = trainSet_1[['x1', 'x2']]
print("(x1,x2)的最大似然估计:")
print("μ：" + str(calculateAvg(trainSet_1_x1x2)))
print("𝜎^2: ")
print(calculateCov(trainSet_1_x1x2))
trainSet_1_x1x3 = trainSet_1[['x1', 'x3']]
print("(x1,x3)的最大似然估计:")
print("μ：" + str(calculateAvg(trainSet_1_x1x3)))
print("𝜎^2:")
print(calculateCov(trainSet_1_x1x3))
trainSet_1_x2x3 = trainSet_1[['x2', 'x3']]
print("(x2,x3)的最大似然估计:")
print("μ：" + str(calculateAvg(trainSet_1_x2x3)))
print("𝜎^2: ")
print(calculateCov(trainSet_1_x2x3))

print("类2：")
trainSet_2_x1x2 = trainSet_2[['x1', 'x2']]
print("(x1,x2)的最大似然估计:")
print("μ：" + str(calculateAvg(trainSet_2_x1x2)))
print("𝜎^2: ")
print(calculateCov(trainSet_2_x1x2))
trainSet_2_x1x3 = trainSet_2[['x1', 'x3']]
print("(x1,x3)的最大似然估计:")
print("μ：" + str(calculateAvg(trainSet_2_x1x3)))
print("𝜎^2: ")
print(calculateCov(trainSet_2_x1x3))
trainSet_2_x2x3 = trainSet_2[['x2', 'x3']]
print("(x2,x3)的最大似然估计:")
print("μ：" + str(calculateAvg(trainSet_2_x2x3)))
print("𝜎^2: ")
print(calculateCov(trainSet_2_x2x3))

# (3) 计算并打印每个数据集的整体平均向量和协方差矩阵
print("(3)")
print("类1")
print("(x1,x2,x3)的最大似然估计: µ" + str(calculateAvg(trainSet_1)))
print("Σ:")
print(calculateCov(trainSet_1))
print("类2")
print("(x1,x2,x3)的最大似然估计: µ" + str(calculateAvg(trainSet_2)))
print("Σ:")
print(calculateCov(trainSet_2))

# 取出每一个特征
trainSet_1_x1 = trainSet_1['x1'].to_frame()
trainSet_1_x2 = trainSet_1['x2'].to_frame()
trainSet_1_x3 = trainSet_1['x3'].to_frame()
trainSet_2_x1 = trainSet_2['x1'].to_frame()
trainSet_2_x2 = trainSet_2['x2'].to_frame()
trainSet_2_x3 = trainSet_2['x3'].to_frame()

# (4) 计算并打印每个数据集的对角协方差矩阵
print("(4)")
print("类1")
print("(x1,x2,x3)的最大似然估计:")
print("µ" + str(calculateAvg(trainSet_1)))
Cov_1 = np.zeros((3, 3))
Cov_1[0, 0] = calculateCov(trainSet_1_x1)
Cov_1[1, 1] = calculateCov(trainSet_1_x2)
Cov_1[2, 2] = calculateCov(trainSet_1_x3)
print("Σ:")
print(Cov_1)
print("类2")
print("(x1,x2,x3)的最大似然估计:")
print("µ" + str(calculateAvg(trainSet_2)))
Cov_2 = np.zeros((3, 3))
Cov_2[0, 0] = calculateCov(trainSet_2_x1)
Cov_2[1, 1] = calculateCov(trainSet_2_x2)
Cov_2[2, 2] = calculateCov(trainSet_2_x3)
print("Σ:")
print(Cov_2)
