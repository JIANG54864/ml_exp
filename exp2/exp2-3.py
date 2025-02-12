import csv
import random

# 读取CSV文件中的数据
with open(r'input/diagnosis_result.csv') as file:
    reader = csv.DictReader(file)
    datas = [row for row in reader]

# 归一化，先获取最大值最小值
# 获取除了 'id' 和 'result' 之外的特征名称
feature_keys = [col for col in reader.fieldnames if col not in ['id', 'diagnosis_result']]
# print(feature_keys)
max_min_values = {}  # 存储各特征的最大最小值

for key in feature_keys:
    values = [float(data[key]) for data in datas]
    max_val = max(values)
    min_val = min(values)
    max_min_values[key] = (max_val, min_val)  # 保存到字典
    # 归一化处理
    for data in datas:
        data[key] = (float(data[key]) - min_val) / (max_val - min_val)

# 对数据进行随机排序,可以保证随机性，但会导致每次实验正确率不稳定
# 在测试不同训练集和测试集划分以及不同k值时将这一步骤注释掉以比较准确率
random.shuffle(datas)

# 改变训练集与测试集的划分:改变n的数值
# 通过不同的取值发现n取3时正确率最高
n = len(datas) // 3
test_set = datas[0:n]
train_set = datas[n:]

# 计算两个数据之间的欧氏距离
def distance(d1, d2):
    res = 0
    # 对每个特征计算欧氏距离
    for key1 in feature_keys:
        res += (float(d1[key1]) - float(d2[key1])) ** 2
    return res ** 0.5

# 改变k的取值
# 通过不同的取值发现k=5时正确率最高
k = 5

# K近邻算法实现
def knn(data):
    # 计算测试数据与训练集中每个数据的距离（列表推导式）
    res = [
        {"result": train["diagnosis_result"], "distance": distance(data, train)}
        for train in train_set
    ]

    # 按距离排序（匿名函数）
    res = sorted(res, key=lambda item: item['distance'])
    # 取前K个值
    res2 = res[0:k]
    # 加权平均
    result = {'Positive': 0, 'Negative': 0}

    # 总长度
    sum_dist = 0
    for r1 in res2:
        sum_dist += r1['distance']

    # 逐个分类加和
    for r2 in res2:
        result[r2["result"]] += 1 - (r2["distance"] / sum_dist) # 距离越小权重越大


    # 返回分类结果
    if result['Positive'] > result['Negative']:
        return 'Positive'
    else:
        return 'Negative'

# 求正确率
correct = 0
for test in test_set:
    result1 = test['diagnosis_result']
    result2 = knn(test)
    print('id:', test['id'], 'predicted:', result2, 'original:', result1)

    if result1 == result2:
        correct = correct + 1

print("正确率：" + str(correct / len(test_set)))


# 封装预测函数
def predict_new_data(new_data):
    # 归一化新数据
    normalized_data = {}
    for key in feature_keys:
        max_val, min_val = max_min_values[key]
        normalized_data[key] = (float(new_data[key]) - min_val) / (max_val - min_val)
    normalized_data['diagnosis_result'] = 'Unknown'  # 占位符

    # 预测结果
    return knn(normalized_data)

new_case = {'radius':11, 'texture':22, 'perimeter':333, 'area':444,
            'smoothness':0.5, 'compactness':0.06, 'symmetry':0.07, 'fractal_dimension':0.8}

# 检查特征是否完整
assert all(key in new_case for key in feature_keys), "缺少必要特征！"

# 预测并输出
prediction = predict_new_data(new_case)
print(f"输入样本预测诊断结果：{prediction}")
