import numpy as np
import matplotlib.pyplot as plt

def getTestDataset():
    data_file = open("input/mnist_test.csv", "r")
    data_list = data_file.readlines()[1:]
    data_file.close()
    # 随机选取1000张样本作为测试集
    np.random.shuffle(data_list)
    return data_list[:1000]


# 定义一个神经网络的类
class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes1, hiddennodes2, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes1 = hiddennodes1
        self.hnodes2 = hiddennodes2
        self.onodes = outputnodes
        self.lr = learningrate

        # 初始化激活函数
        self.activation_func = lambda x: np.maximum(x, 0) # ReLU函数

        # 初始化链接权重
        self.wih = np.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes1, self.inodes))
        self.whh = np.random.normal(0.0, pow(self.hnodes2, -0.5), (self.hnodes2, self.hnodes1))
        self.who = np.random.normal(0.0, pow(self.hnodes2, -0.5), (self.onodes, self.hnodes2))

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T

        hidden_inputs1 = np.dot(self.wih, inputs)
        hidden_outputs1 = self.activation_func(hidden_inputs1)

        hidden_inputs2 = np.dot(self.whh, hidden_outputs1)
        hidden_outputs2 = self.activation_func(hidden_inputs2)

        final_inputs = np.dot(self.who, hidden_outputs2)
        final_output = self.activation_func(final_inputs)

        return final_output

    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        hidden_inputs1 = np.dot(self.wih, inputs)
        hidden_outputs1 = self.activation_func(hidden_inputs1)

        hidden_inputs2 = np.dot(self.whh, hidden_outputs1)
        hidden_outputs2 = self.activation_func(hidden_inputs2)

        final_inputs = np.dot(self.who, hidden_outputs2)
        final_output = self.activation_func(final_inputs)

        output_errors = targets - final_output

        loss = np.sqrt(np.sum(np.square(final_output - targets)) / len(final_output))

        hidden_errors2 = np.dot(self.who.T, output_errors)
        hidden_errors1 = np.dot(self.whh.T, hidden_errors2)

        output_gradient = final_output > 0
        hidden_gradient2 = hidden_outputs2 > 0
        hidden_gradient1 = hidden_outputs1 > 0

        self.who += self.lr * np.dot((output_errors * output_gradient), np.transpose(hidden_outputs2))
        self.whh += self.lr * np.dot((hidden_errors2 * hidden_gradient2), np.transpose(hidden_outputs1))
        self.wih += self.lr * np.dot((hidden_errors1 * hidden_gradient1), np.transpose(inputs))

        return loss

# 首先训练模型
def TrainNetwork():
    # 输入层有 28*28个神经元，隐藏层1有50个神经元，隐藏层2有30个神经元，输出层有10个神经元
    input_nodes = 784
    hidden_nodes1 = 50
    hidden_nodes2 = 40
    output_nodes = 10
    learning_rate = 0.0001
    mynet = NeuralNetwork(inputnodes=input_nodes, hiddennodes1=hidden_nodes1, hiddennodes2=hidden_nodes2, outputnodes=output_nodes,
                          learningrate=learning_rate)
    # 导入训练数据
    data_file = open("input/mnist_train.csv", "r")
    data_list = data_file.readlines()[1:] # 跳过第一行标题行
    data_file.close()

    # 随机打乱数据
    np.random.shuffle(data_list)
    # 划分为59000张训练集和1000张验证集
    train_data = data_list[:59000]
    validation_data = data_list[59000:60000]

    epoch_losses = [] # 用于记录每个epoch的平均损失
    validation_accuracies = []  # 用于记录每个epoch的验证集准确率
    for i in range(epochs):
        epoch_loss = 0
        for record in train_data:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01 # 归一化
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99 # 将正确数字对应位置值设为0.99，其余元素设为0.01
            loss = mynet.train(inputs, targets)
            epoch_loss += loss
        avg_loss = epoch_loss / len(train_data)
        epoch_losses.append(avg_loss)
        # print(f"Epoch {i+1}, Loss: {avg_loss:.4f}")

        # 在验证集上评估模型
        correct = 0
        for record in validation_data:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            predict_result_percent = mynet.query(inputs)
            predict_result = predict_result_percent.argmax()
            if predict_result == int(all_values[0]):
                correct += 1
        accuracy = correct / len(validation_data)
        validation_accuracies.append(accuracy)
        print(f"Epoch {i+1}, Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

    return mynet, epoch_losses, validation_accuracies


# 在测试集上测试
def singleTest(x, n, c):
    all_values = x.split(',')
    actural_result = all_values[0]
    formatInput = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    predict_result_percent = n.query(formatInput)
    predict_result = predict_result_percent.argmax()
    isCorrect = int(actural_result) == int(predict_result)
    c['count'] += 1
    if isCorrect:
        c['correct'] += 1
    else:
        c['incorrect'] += 1


# 执行程序，统计结果
def Run():
    dataset = getTestDataset()
    mynet, train_losses, validation_accuracies = TrainNetwork()
    statistics = {
        "correct": 0,
        "incorrect": 0,
        "count": 0
    }
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, 'b-o', label='Training Loss')
    plt.plot(range(1, epochs + 1), validation_accuracies, 'r-o', label='Validation Accuracy')
    plt.title('Training Loss and Validation Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.legend() # 显示图例
    plt.grid(True)
    plt.show()

    [singleTest(data, mynet, statistics) for data in dataset]
    print('完成第', epochs, '次学习')
    print(f"正确率为 {statistics.get('correct') / statistics.get('count')}")

if __name__ == '__main__':
    epochs = 10
    Run()
