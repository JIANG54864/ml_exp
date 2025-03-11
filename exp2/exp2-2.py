import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
def window(sample: pd.Series, trainSample: pd.Series, h):
    """
    计算测试样本与训练样本之间的窗函数值。

    参数:
    sample - 测试样本
    trainSample - 训练样本
    h - 窗函数的带宽

    返回:
    窗函数值
    """
    vector_s = np.matrix(sample).T# 转置
    vector_ts = np.matrix(trainSample).T
    diff = vector_s - vector_ts # 差值
    return math.exp(-diff.T * diff / (2 * h ** 2)) # 按公式算

# 使用Parzen窗方法估计该类的条件概率密度
def parzen(sample: pd.Series, trainSet: pd.DataFrame):
    """
    使用Parzen窗方法估计样本属于某一类的条件概率密度。

    参数:
    sample - 测试样本
    train_set - 训练集

    返回:
    条件概率密度
    """
    likelihood = 0.0
    for index, row in trainSet.iterrows():
        likelihood += window(sample, row, 1)
    likelihood = likelihood / len(trainSet) #取平均值
    return likelihood

# Parzen窗分类器
def parzen_classifier(sample: pd.Series):
    """
    使用Parzen窗方法对样本进行分类。

    参数:
    sample - 测试样本
    """
    posterior_1 = parzen(sample, train_set_1)
    posterior_2 = parzen(sample, train_set_2)
    posterior_3 = parzen(sample, train_set_3)
    print(sample)
    print("p(w1): " + str(posterior_1))
    print("p(w2): " + str(posterior_2))
    print("p(w3): " + str(posterior_3))
    if posterior_1 > posterior_2:
        if posterior_1 > posterior_3:
            print("Sample belong 类1")
        else:
            print("Sample belong 类3")
    else:
        if posterior_2 > posterior_3:
            print("Sample belong 类2")
        else:
            print("Sample belong 类3")

# 导入训练集数据
train_set_1 = pd.read_csv('input/ww1.csv')
train_set_2 = pd.read_csv('input/ww2.csv')
train_set_3 = pd.read_csv('input/ww3.csv')

parzen_classifier(pd.Series([0.5, 1.0, 0.0]))
parzen_classifier(pd.Series([0.31, 1.51, -0.50]))
parzen_classifier(pd.Series([-0.3, 0.44, -0.1]))

print('--------------------------------------------------------------')
print('k近邻:')

# 1维KNN方法
def one_dimension_KNN(testData: float, trainSet: pd.Series, k: int):
    """
    一维K近邻方法估计测试数据属于某一类的概率密度。

    参数:
    testData - 测试数据
    train_set - 训练集
    k - 近邻数

    返回:
    概率密度
    """
    distance = []
    for i in range(trainSet.shape[0]): # 遍历训练集，shape[0]表示数据的行数
        d = np.abs(testData - trainSet[i])
        distance.append(d)
    distance.sort()
    posterior = (k / trainSet.shape[0]) / (2 * distance[k - 1])
    # 分子 k / train_set.shape[0] 表示标点周围最近的 k 个样本，占训练集总样本的比例。分母 2 * distance[k - 1] 表示目标点附近的区间长度。
    # 相除得到概率密度
    return posterior

# 导入ww3类的x1特征
train_set1 = pd.read_csv('input/ww3.csv')['x1']
# 随机产生n=500个-2~2的1维随机数
dimension1_randoms = np.random.uniform(-2, 2, 500)
# 进行升序排序
dimension1_randoms = np.sort(dimension1_randoms)
# 三个一维数组存储K值不同情况下的后验概率
dimension1_posterior_1 = []
dimension1_posterior_3 = []
dimension1_posterior_5 = []
# 计算后验概率
for i in range(500):
    dimension1_posterior_1.append(one_dimension_KNN(dimension1_randoms[i], train_set1, 1))
    dimension1_posterior_3.append(one_dimension_KNN(dimension1_randoms[i], train_set1, 3))
    dimension1_posterior_5.append(one_dimension_KNN(dimension1_randoms[i], train_set1, 5))
# 画出三张一维的图像
plt.subplot(131)
plt.plot(dimension1_randoms, dimension1_posterior_1)
plt.title('one_dimension k=1')
plt.subplot(132)
plt.plot(dimension1_randoms, dimension1_posterior_3)
plt.title('one_dimension k=3')
plt.subplot(133)
plt.plot(dimension1_randoms, dimension1_posterior_5)
plt.title('one_dimension k=5')
plt.show()

# 2维knn方法
def two_dimension_KNN(x1: np.matrix, x2: np.matrix, trainSet: pd.DataFrame, k: int):
    """
    二维K近邻方法估计测试数据属于某一类的概率密度。

    参数:
    x1, x2 - 测试数据的网格
    train_set - 训练集
    k - 近邻数

    返回:
    概率密度矩阵
    """
    posteriorMatrix = np.zeros((x1.shape[0], x1.shape[1]))
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            distance = []
            for index, row in trainSet.iterrows():
                d = np.sqrt((x1[i, j] - row[0]) ** 2 + (x2[i, j] - row[1]) ** 2)
                distance.append(d)
            distance.sort()
            posterior = (k / trainSet.shape[0]) / (np.pi * (distance[k - 1] ** 2) + np.spacing(1))
            # 同样是计算距离，只不过是二维欧氏距离
            posteriorMatrix[i, j] = posterior
    return posteriorMatrix

# 导入ww2类的x1,x2特征
train_set2 = pd.read_csv('input/ww2.csv')[['x1', 'x2']]

test_x1 = np.arange(-3, 2, 0.05)
test_x2 = np.arange(0, 4, 0.05)

matrix_x1, matrix_x2 = np.meshgrid(test_x1, test_x2)
# 将两个一维数组转换为二维网格矩阵
posterior1 = two_dimension_KNN(matrix_x1, matrix_x2, train_set2, 1)
posterior3 = two_dimension_KNN(matrix_x1, matrix_x2, train_set2, 3)
posterior5 = two_dimension_KNN(matrix_x1, matrix_x2, train_set2, 5)

fig = plt.figure(figsize=(12, 6), facecolor='w')
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(matrix_x1, matrix_x2, posterior1,
                 rstride=1,
                 cstride=1,
                 cmap=plt.get_cmap('rainbow'))
ax1.set_xlabel('x2')
ax1.set_ylabel('x1')
ax1.set_zlabel('likelihood')
plt.title('two_dimension k=1')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(matrix_x1, matrix_x2, posterior3,
                 rstride=1,
                 cstride=1,
                 cmap=plt.get_cmap('rainbow'))
ax2.set_xlabel('x2')
ax2.set_ylabel('x1')
ax2.set_zlabel('likelihood')
plt.title('two_dimension k=3')

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(matrix_x1, matrix_x2, posterior5,
                 rstride=1,
                 cstride=1,
                 cmap=plt.get_cmap('rainbow'))
ax3.set_xlabel('x2')
ax3.set_ylabel('x1')
ax3.set_zlabel('likelihood')
plt.title('two_dimension k=5')

plt.show()

# 导入三个类别的全部特征
train_set3_1 = pd.read_csv('input/ww1.csv')
train_set3_2 = pd.read_csv('input/ww2.csv')
train_set3_3 = pd.read_csv('input/ww3.csv')
train_set = [train_set3_1, train_set3_2, train_set3_3]

# 三维的KNN方法
def three_dimension_KNN(testData: np.matrix, k: int):
    """
    三维K近邻方法对测试数据进行分类。

    参数:
    testData - 测试数据
    k - 近邻数

    返回:
    最大概率密度对应的类别索引
    """
    distance = [[], [], []]
    posterior = []
    for i in range(len(train_set)):
        for j in range(train_set[i].shape[0]):
            # 仍然是计算欧氏距离，只不过是三维
            d = np.sqrt((testData[0, 0] - train_set[i].iloc[j]['x1']) ** 2 +
                        (testData[1, 0] - train_set[i].iloc[j]['x2']) ** 2 +
                        (testData[2, 0] - train_set[i].iloc[j]['x3']) ** 2)
            distance[i].append(d)
        distance[i].sort()
        V = 4 * np.pi * (distance[i][k - 1] ** 3) / 3 # 计算球体体积
        posterior.append(k / (train_set[i].shape[0]) / V)

    print("类条件概率密度数组:" + str(posterior))
    return posterior.index(max(posterior))

print("(-0.41,0.82,0.88)属于类别" + str(three_dimension_KNN(np.matrix([[-0.41], [0.82], [0.88]]), 3)))
print("(0.14,0.72,4.1)属于类别" + str(three_dimension_KNN(np.matrix([[0.14], [0.72], [4.1]]), 3)))
print("(-0.81,0.61,-0.38)属于类别" + str(three_dimension_KNN(np.matrix([[-0.81], [0.61], [-0.38]]), 3)))
