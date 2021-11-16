from numpy import *
import numpy as np
from pprint import pprint
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


# k-近邻算法代码1
def createDataSet():
    # 定义6个训练样本集，每个样本有2个特征
    group = np.array([[3, 4, 104],
                      [2, 3, 100],
                      [1, 2, 81],
                      [101, 10, 7],
                      [99, 5, 9],
                      [98, 2, 15]])
    # group = np.array([[3, 104],
    #                   [2, 100],
    #                   [1, 81],
    #                   [101, 10],
    #                   [99, 5],
    #                   [98, 2]])
    # 定义每个样本点的标签值
    labels = ['A', 'A', 'A', 'B', 'B', 'B']
    return group, labels


# k-近邻算法代码2
# inX：待分类点；dataSet：训练数据集；labels：标签；k：最近邻居数目
def classify0(inX, dataSet, labels, k):
    import operator as op
    # dataSet.shape[1] ：第一维的长度（行）；dataSet.shape[0]=6 ：第二维的长度（列）
    dataSetSize = dataSet.shape[0]

    # 步骤1：计算距离
    # 作差：首先将待分类点inX复制成dataSetSize相同行，列，然后再和训练数据集中的数据作差（对应元素相减）,
    # (dataSetSize, 1)表示生成6行
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 平方
    sqDiffMat = diffMat ** 2
    # print(sqDiffMat)
    # 按行求和
    sqDistances = sqDiffMat.sum(axis=1)
    # print(sqDistances)
    # 开根
    distances = sqDistances ** 0.5
    # print(distances)
    # argsort()：返回的是数组值从小到大的索引值（注意是索引值）
    sortedDistIndicies = distances.argsort()
    # print(sortedDistIndicies)
    # 步骤2：选择距离最小的k个点
    # classCount：字典 key/value键值,存某个类型出现的次数
    classCount = {}
    for i in range(k):
        # 返回距离较小的k个点labels值, voteIlabel =A
        voteIlabel = labels[sortedDistIndicies[i]]
        # 字典中的key取值value
        # get()：返回指定键voteIlabel的值，如果值存在字典就加1，如果值不在字典中返回0.
        # 第一次字典没有a为0+1=1，第二次有a为1+1=2
        # print(classCount.get(voteIlabel, 0))
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # print(classCount)
    # 步骤3：对字典classCount进行排序
    # 逆向排序，从大到小（出现次数value值）
    # sortedClassCount：对字典排序 ，返回的是List列表,不再是Dictionary字典。
    sortedClassCount = sorted(classCount.items(), key=op.itemgetter(1), reverse=True)
    # print(sortedClassCount)
    # 返回出现频率最高的元素标签
    return sortedClassCount[0][0]


#  读取文件到矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)  # 得到文件行数
    # print(numberOfLines)
    returnMat = zeros((numberOfLines, 3))  # 定义一个全为0的矩阵
    # print(returnMat)
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        # strip()方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        # 注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符
        line = line.strip()
        listFromLine = line.split('\t')  # 按中间跳格符\t分割为list。 \t 跳格  \r 回车 \n 换行
        # print(listFromLine[0:3])
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))  # -1表示列表中的最后一列元素，
        index += 1
    return returnMat, classLabelVector


# 归一化数值
def autoNorm(dataSet):
    import sklearn.preprocessing as sp
    mms = sp.MinMaxScaler(feature_range=(0, 1))
    # 用范围缩放器实现特征值的范围缩放,.reshape(-1, 1),-1=?,1=1列
    normDataSet = mms.fit_transform(dataSet)
    minvals = dataSet.min(0)
    maxvals = dataSet.max(0)
    # print(maxvals)
    ranges = maxvals - minvals
    # normDataSet = zeros(shape(dataSet))
    # m = dataSet.shape[0]
    # normDataSet = dataSet - tile(minvals, (m, 1))
    # tile()函数将变量内容复制成输入矩阵同样大小的矩阵
    # normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minvals


# 测试算法
def datingClassTest():
    hoRatio = 0.10  # 设置测试集比重，前10%作为测试集，后90%作为训练集
    # datingLabels为1，2，3标签
    datingDataMat, datingLabels = file2matrix(r"D:\ana\envs\py36\mywagtailone\datatables\my_test\machinelearninginaction\Ch02\datingTestSet2.txt")
    # print(datingLabels)
    # print(datingDataMat)
    normMat, ranges, minvals = autoNorm(datingDataMat)
    # print(normMat)
    m = normMat.shape[0]  # 得到样本数量m=1000
    # print(m)
    numTestVecs = int(m * hoRatio)  # 得到测试集最后一个样本的位置=100
    # print(numTestVecs)
    errorCount = 0.0   # 初始化定义错误个数为0
    for i in range(numTestVecs):
        # 测试集中元素逐一放进分类器测试，k = 3
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        # 输出分类结果与实际label
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        # 若预测结果与实际label不同，则errorCount+1
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
        # 输出错误率 = 错误的个数 / 总样本个数
        print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


# 神经网络图片识别
def get_mnist_picture_data():
    from tensorflow.keras.datasets import mnist
    from tensorflow import keras
    from tensorflow.keras import layers

    # 该数据是由28 * 28像素的图像构成，训练集有60000个，测试集有10000个。
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # print(train_images.shape)
    # print(len(train_labels))
    # print(test_images.shape)
    # print(len(test_labels))
    # print(test_labels)
    # 神经网络的核心是layer，它是一种数据处理模块，可以看作数据过滤器。层从数据中提取出表示。
    # 本例中使用2个Dense层，它们是全连接的神经层。第二层是一个10路的softmax层，它将返回由10个概率值组成的数组。对应12345。。。
    model = keras.Sequential([layers.Dense(512, activation="relu"), layers.Dense(10, activation="softmax")])
    # 编译，它需要三个参数：
    # 损失函数（loss function）：网络衡量在训练数据上的性能，即网络如何朝着正确的方向前进。
    # 优化器（optimizer）: 基于训练数据和损失函数来更新网络的机制。
    # 在训练和测试过程中需要监控的指标（metric）：本例只关心精度，即正确分类的图像所占的比例。
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # 对数据进行预处理，转化为网络要求的形状，并缩放到所有值都在[0，1]区间。之前训练图像保存在一个uint8类型的数组中，
    # 形状为(60000, 28, 28)，取值区间为[0, 255]，需要转化为float32数组，形状为(60000, 28 * 28)，取值范围为0 ~ 1。
    # print(train_images[0])
    # 把train_images转换为6万行，28*28列，原来6万行: 28行*28列为一个图像
    train_images = train_images.reshape((60000, 28 * 28))
    # print(train_images[0])
    # 转换为0-1
    train_images = train_images.astype('float32') / 255
    model.fit(train_images, train_labels, epochs=5, batch_size=128)
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    # test_digits = test_images[0:10]
    # predictions = model.predict(test_digits)
    # print("a", predictions)
    # print("b", predictions[0])
    # print("c", predictions[0].argmax)
    # print("d", predictions[0][7])
    # print("r", test_labels[0])
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"test_acc: {test_acc}")


# 学习基础
def base_learning():
    from tensorflow.keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # ndim轴
    # print(train_images.ndim)
    # print(train_images.dtype)
    import matplotlib.pyplot as plt
    digit = train_images[4]
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()


# 电影评论2分类
def film_review_imdb():
    from tensorflow.keras.datasets import imdb
    from tensorflow import keras
    from tensorflow.keras import layers
    import matplotlib.pyplot as plt

    # 参数 num_words=10000 的意思是仅保留训练数据中前 10 000 个最常出现的单词
    # 数据集被分为用于训练的 25 000 条评论与用于测试的 25 000 条评论，训练集和测试
    # 集都包含 50% 的正面评论和 50% 的负面评论
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    # print(train_data[0])
    # print(train_labels[0])
    # ma = max([max(sequence) for sequence in train_data])
    # print(ma)
    # 获取对应的单词word_index
    # word_index = imdb.get_word_index()
    # word_index字典里 键值对是  单词是key 数字是value  我们交换一下
    # reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    # print(reverse_word_index.get(1-3, '?'))
    # print(reverse_word_index.get(14-3, '?'))
    # print(reverse_word_index.get(22-3, '?'))
    # print(reverse_word_index.get(16-3, '?'))
    # print(reverse_word_index.get(43-3, '?'))
    # print(reverse_word_index.get(530-3, '?'))
    # print(train_data[0])
    # print("000z", train_data[0][0])
    # print("001z", train_data[0][1])
    # print("002z", train_data[0][2])
    # print("003z", train_data[0][3])
    # print(train_data[0][4])
    # 合成出train_data[0]代表的语句,i-3是因为0.1.2是填充，序列开始，未知词,大于2才是表示单词
    # decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
    # print(decoded_review)
    # 处理输入数据
    # print(train_data.shape)
    # print(train_data[0])
    x_train = vectorize_sequences(train_data)
    print(x_train.shape)
    # print(x_train[0])
    x_test = vectorize_sequences(test_data)
    # 将标签向量化
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    model = keras.Sequential([
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
    # 验证数据集取一些
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]
    # 将模型训练20次。与此同时，还要监控留出的10000个样本上的损失函数和精度，
    # 可以通过将验证数据传入validation_data参数来完成。
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=4,
                        batch_size=512,
                        validation_data=(x_val, y_val))
    # 当我们调用model.fit() 的时候，返回一个History对象。这个对象有一个成员history，它是一个字典，包含训练过程中的所有数据。
    # history_dict = history.history
    # # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
    # print(history_dict.keys())
    # # 训练损失和验证损失
    # loss_values = history_dict["loss"]
    # val_loss_values = history_dict["val_loss"]
    # epochs = range(1, len(loss_values) + 1)
    # plt.plot(epochs, loss_values, "bo", label="Training loss")
    # plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    # plt.title("Training and validation loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()
    # 训练精度 和  验证精度
    # cla()   # Clear axis
    # clf()   # Clear figure
    # close() # Close a figure window
    # plt.clf()
    # acc = history_dict["accuracy"]
    # val_acc = history_dict["val_accuracy"]
    # plt.plot(epochs, acc, "bo", label="Training acc")
    # plt.plot(epochs, val_acc, "b", label="Validation acc")
    # plt.title("Training and validation accuracy")
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy")
    # plt.legend()
    # plt.show()
    results = model.evaluate(x_test, y_test)
    print(results)
    res = model.predict(x_test)
    print(len(res))


# 电影评论2分类2,对列表进行one-hot编码 ：比如序列[3, 5]将会被转换为10000维向量，只有索引为3和5的元素是1，其余元素是0
# 注意sequences数据是数字，不是单词，比如455，代表在10000list里索引为455的单词
def vectorize_sequences(sequences, dimension=10000):
    # print("len", len(sequences))
    # results 25000,10000
    results = np.zeros((len(sequences), dimension))
    # print(results.shape)
    # seasons = ['Spring', 'Summer', 'Fall', 'Winter'],
    # list(enumerate(seasons))
    # [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
    # list(enumerate(seasons, start=1))
    #  下标从 1 开始,
    # [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]

    # seq = ['one', 'two', 'three']
    # for i, element in enumerate(seq):
    #     print i, element
    # 0
    # one
    # 1
    # two
    # 2
    # three
    # sequence里是数字的list，不是单词
    for i, sequence in enumerate(sequences):
        # print(i, sequence)
        for j in sequence:
            results[i, j] = 1.
    return results


# 路透社新闻多分类
def learn_reuters():
    from tensorflow.keras.datasets import reuters
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    # print(len(train_data))
    # print(len(test_data))
    # print(train_data[0])
    # word_index = reuters.get_word_index()
    # reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    # decoded_newswire = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])
    # print("bv", decoded_newswire)
    # print(train_labels[10:11])
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    # print("bv", test_labels[0])
    # y_train = to_one_hot(train_labels)
    # y_test = to_one_hot(test_labels)
    # print("bbv", y_test[0:1])
    from tensorflow.keras.utils import to_categorical
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)
    # print("bbbv", y_test[0:1])

    from tensorflow import keras
    from tensorflow.keras import layers
    # 最后一层46维向量输出，用softmax激活函数，网络输出在46个类别上的概率分布。46个概率总和为1. 然后使用损失函数
    # categorical_crossentropy（分类交叉熵）。它用于衡量两个概率分布之间的距离。
    # 对比的两个分别是网络输出的概率分布和标签的真实分布。通过让这两个分布之间的距离最小化，来训练网络
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(46, activation="softmax")
    ])
    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = y_train[:1000]
    partial_y_train = y_train[1000:]
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=8,
                        batch_size=512,
                        validation_data=(x_val, y_val))
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.clf()
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    plt.plot(epochs, acc, "bo", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    results = model.evaluate(x_test, y_test)
    print(results)
    predictions = model.predict(x_test)
    print(predictions[0].shape)
    print(np.sum(predictions[0]))
    print(np.argmax(predictions[0]))
    print(predictions)


# 路透社46哥标签独热编码,被from tensorflow.keras.utils import to_categorical代替
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    # print(results)
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


# 房价预测
def predict_house_price():
    """
英文简称	详细含义
CRIM	城镇的人均犯罪率
ZN	    25,000平方英尺的地块的住宅用地比例。
INDUS	每个镇上平均的非零售企业比例
CHAS	查尔斯河虚拟变量（如果环河，则等于1；否则等于0）
NOX	一氧化氮的浓度（百万分之几）
RM	每个住宅的平均房间数
AGE	1940年之前建造的自有住房的比例
DIS	到五个波士顿就业中心的加权距离
RAD	径向公路通达性的指标
TAX	每一万美元的全值财产税率
PTRATIO	各镇的师生比率
B	计算方法为 1000 ( B k − 0.63 ) 平方2 其中Bk是按城镇划分的非裔美国人的比例
LSTAT	底层人口的百分比(%)
price	自有住房数的中位数，单位（千美元）
    """
    from tensorflow.keras.datasets import boston_housing
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
    # print(train_data.shape)
    # print(test_data.shape)
    # print(train_targets)
    # pprint(train_data[0:2])
    """
    axis，这个代表的是运算的方向，我们的train_data是两个维度的，为(404,13)，由于python从0开始计数，
    axis=0的意思就是考虑404这个维度上的数据，这里mean就是求这404个数据的均值，0为列
    std则求这404个数据的标准差，然后剩下的就是减去均值除以标准差，这个过程叫做归一化，属于正则化的一种，
    """
    # 13列的均值
    mean = train_data.mean(axis=0)
    # pprint(mean[0:])
    train_data -= mean
    # pprint(train_data[0:2])
    std = train_data.std(axis=0)
    # pprint(std)
    train_data /= std
    # pprint(train_data[0:2])
    test_data -= mean
    test_data /= std
    # ss = np.array([
    #     [1., 2., 3.],
    #     [2., 2., 3.],
    #     [3., 2., 3.],
    # ])
    # m = ss.mean(axis=0)
    # pprint(m)
    # ss -= m
    # pprint(ss)
    # s = ss.std(axis=0)
    # print(s)

    k = 4
    # 但 // 取的是结果的最小整数，而 / 取得是实际的除法结果
    num_val_samples = len(train_data) // k
    # print(num_val_samples)
    # predict_house_price1(k, num_val_samples, train_data, train_targets)
    # predict_house_price2(k, num_val_samples, train_data, train_targets)
    model = build_model()
    model.fit(train_data, train_targets, epochs=130, batch_size=16, verbose=0)
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
    # print(test_mse_score)
    # print(test_mae_score)
    predictions = model.predict(test_data)
    print(predictions.shape)
    print(predictions[0])


# 房价预测son2 k折交叉验证
def predict_house_price2(k, num_val_samples, train_data, train_targets):
    num_epochs = 130
    all_mae_histories = []
    for i in range(k):
        print(f"Processing fold #{i}")
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples],
             train_targets[(i + 1) * num_val_samples:]],
            axis=0)
        model = build_model()
        """
        fit 中的 verbose
        verbose：日志显示
        verbose = 0 为不在标准输出流输出日志信息
        verbose = 1 为输出进度条记录
        verbose = 2 为每个epoch输出一行记录
        注意： 默认为 1

        evaluate 中的 verbose
        verbose：日志显示
        verbose = 0 为不在标准输出流输出日志信息
        verbose = 1 为输出进度条记录
        注意： 只能取 0 和 1；默认为 1
        """
        history = model.fit(partial_train_data, partial_train_targets,
                            validation_data=(val_data, val_targets),
                            epochs=num_epochs, batch_size=16, verbose=0)
        # 保存每一次交换训练的结果
        mae_history = history.history["val_mae"]
        # print(mae_history)
        all_mae_histories.append(mae_history)
    # print(all_mae_histories)
    # len average_mae_history = len num_epochs,求列mean.  计算训练后的均值：
    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
    # print(average_mae_history)
    plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    plt.xlabel("Epochs")
    plt.ylabel("Validation MAE")
    plt.show()
    truncated_mae_history = average_mae_history[10:]
    plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
    plt.xlabel("Epochs")
    plt.ylabel("Validation MAE")
    plt.show()


# 房价预测son1 k折交叉验证
def predict_house_price1(k, num_val_samples, train_data, train_targets):
    num_epochs = 100
    all_scores = []
    # print(range(k))
    """
        k重验证集
    数据量小导致了我们在进行验证的时候很可能结果会依赖于我们选取的验证集，为了解决这个问题，我们就把测试集划成几部分，
    这几部分轮流当验证集，分别训练验证，得到结果，最后平均一下，得到的结果就更具有说服力，这就叫做k-fold。
    concatenate，这个就是把两个数组连在一起。
        """
    for i in range(k):
        print(f"Processing fold #{i}")
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                             train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                                train_targets[(i + 1) * num_val_samples:]], axis=0)
        model = build_model()
        model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=16, verbose=0)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)
    print("Proc", all_scores)
    print("Processing", np.mean(all_scores))


# 房价预测model
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    """
    ReLU函数(Rectified Linear Units)其实就是一个取最大值函数，注意这并不是全区间可导的，但是我们可以取次梯度(subgradient)。
    这里的loss采用了mse，均方误差，这在回归问题中很常用。metrics采用mae，代表绝对误差，即预测值和目标值的差值的绝对值。
    """
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


# 温度预测
def predict_c():
    import os
    fname = os.path.join(r"D:\360极速浏览器下载\new\jena_climate_2009_2016.csv")
    with open(fname) as f:
        data = f.read()
    lines = data.split('\n')  # 按行切分
    header = lines[0].split(',')  # 列，，，每行按，切分
    lines = lines[1:]  # 去除第0行，第0行为标记
    # print(len(header))
    # print(len(lines))
    """header  15列                 温度
    ['"Date Time"', '"p (mbar)"', '"T (degC)"', '"Tpot (K)"', '"Tdew (degC)"', '"rh (%)"', '"VPmax (mbar)"',
    '"VPact (mbar)"', '"VPdef (mbar)"', '"sh (g/kg)"', '"H2OC (mmol/mol)"', '"rho (g/m**3)"', '"wv (m/s)"', '"max. wv (m/s)"',
    '"wd (deg)"']
    """
    # 将所有的数据转为float=420451,14型数组,header=15,len(lines)=420451,
    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
        values = [float(x) for x in line.split(',')[1:]]   # 去掉第一列
        float_data[i, :] = values

    # temp = float_data[:, 1]
    # plt.figure()
    # plt.plot(range(len(temp)), temp)
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(range(1440), temp[:1440])
    # plt.show()
    # 数据标准化，减去平均值，除以标准值
    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    # print("std", std)
    # print("std", std[1])
    float_data /= std
    lookback = 1440  # 给定10天的观测数据
    step = 6  # 每6个采样一次，即每小时一个数据点
    delay = 144  # 目标是未来24小时之后的数据
    batch_size = 128
    # shuffle = True,打乱数据
    train_gen = generator(float_data, lookback=lookback, delay=delay,
                          min_index=0, max_index=200000, shuffle=True,
                          step=step, batch_size=batch_size)
    val_gen = generator(float_data, lookback=lookback, delay=delay,
                        min_index=200001, max_index=300000, step=step,
                        batch_size=batch_size)
    test_gen = generator(float_data, lookback=lookback, delay=delay,
                         min_index=300001, max_index=None, step=step,
                         batch_size=batch_size)
    # 为了查看整个验证集，需要从 val_gen中抽取多少次
    val_steps = (300000 - 200001 - lookback) // batch_size
    # 为了查看整个测试集，需要从test_gen中抽取多少次
    test_steps = (len(float_data) - 300001 - lookback) // batch_size
    # evaluate_naive_method(val_steps, val_gen, std[1])
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import RMSprop
    """keras.layers.Flatten(input_shape=[])用于将输入层的数据压成一维的数据，一般用再卷积层和全连接层之间
    （因为全连接层只能接收一维数据，而卷积层可以处理二维数据，就是全连接层处理的是向量，而卷积层处理的是矩阵）"""

    # 密集连接模型DNN
    model = Sequential()
    # print("lookback // step", lookback // step)  # lookback // step=240
    # print("float_data.shape", float_data.shape)
    # print("float_data.shape[-1]", float_data.shape[-1])  # float_data.shape[-1]=14列
    """输入240行14列数据"""
    model.add(layers.Flatten(input_shape=[lookback // step, float_data.shape[-1]]))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit_generator(train_gen, steps_per_epoch=1, epochs=1, validation_data=val_gen,
                                  validation_steps=val_steps)
    # history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20, validation_data=val_gen,
    #                               validation_steps=val_steps)

    # 基于GRU的模型
    # model = Sequential()
    # model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
    # model.add(layers.Dense(1))
    # model.compile(optimizer=RMSprop(), loss='mae')
    # history = model.fit_generator(train_gen, steps_per_epoch=100, epochs=2,
    #                               validation_data=val_gen, validation_steps=val_steps)
    '''
    门控循环单元（GRU，gated recurrent unit）层的工作原理与LSTM相同。但它做了一些简化，因此运 行的计算代价更低
    （虽然表示能力可能不如LSTM）。机器学习中到处可以见到这种计算代价与表示能力之间的折中。
    '''
    # model = Sequential()
    # model.add(layers.GRU(32,  dropout=0.1, recurrent_dropout=0.1, input_shape=(None, float_data.shape[-1])))
    # model.add(layers.Dense(1))
    # model.compile(optimizer=RMSprop(), loss='mae')
    # history = model.fit_generator(train_gen, steps_per_epoch=100, epochs=3,
    #                               validation_data=val_gen, validation_steps=val_steps)

    # 循环差堆叠
    # model = Sequential()
    # model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.1, return_sequences=True,
    #                      input_shape=(None, float_data.shape[-1])))
    # model.add(layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.1))
    # model.add(layers.Dense(1))
    #
    # model.compile(optimizer=RMSprop(), loss='mae')
    # history = model.fit_generator(train_gen, steps_per_epoch=200, epochs=15,
    #                               validation_data=val_gen, validation_steps=val_steps)

    # # 使用双向GRU
    # model = Sequential()
    # model.add(layers.Bidirectional(layers.GRU(32), input_shape=(None, float_data.shape[-1])))
    # model.add(layers.Dense(1))
    # model.compile(optimizer=RMSprop(), loss='mae')
    # history = model.fit_generator(train_gen, steps_per_epoch=300, epochs=30,
    #                               validation_data=val_gen, validation_steps=val_steps)

    # 使用一维CNN
    # model = Sequential()
    # model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None, float_data.shape[-1])))
    # model.add(layers.MaxPooling1D(3))
    # model.add(layers.Conv1D(32, 5, activation='relu'))
    # model.add(layers.MaxPooling1D(3))
    # model.add(layers.Conv1D(32, 5, activation='relu'))
    # model.add(layers.GlobalMaxPool1D())
    # model.add(layers.Dense(1))
    # model.compile(optimizer=RMSprop(), loss='mae')
    # history = model.fit_generator(train_gen, steps_per_epoch=300, epochs=10,
    #                               validation_data=val_gen, validation_steps=val_steps)

    # 一维卷积基与GRU融合
    # model = Sequential()
    # model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None, float_data.shape[-1])))
    # model.add(layers.MaxPooling1D(3))
    # model.add(layers.Conv1D(32, 5, activation='relu'))
    # model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.1))
    # model.add(layers.Dense(1))

    # model.compile(optimizer=RMSprop(), loss='mae')
    # history = model.fit_generator(train_gen, steps_per_epoch=300, epochs=10,
    #                               validation_data=val_gen, validation_steps=val_steps)

    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    # plt.figure()
    # plt.plot(epochs, loss, 'bo', label='training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and Validation loss')
    # plt.legend()
    # plt.show()
"""
定义一个生成器。它生成了一个元组 (samples,  targets)，其中 samples 是输入数据的一个批量，
targets 是对应的目标温度数组
lookback = 1440  # 给定10天的观测数据
step = 6  # 每6个采样一次，即每小时一个数据点
delay = 144  # 目标是未来24小时之后的数据
batch_size = 128
"""


def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
        print(len(data), max_index)
    # print(min_index, lookback)
    # val_gen(min_index, lookback)=200001 1440
    i = min_index + lookback
    # val_gen i=201441
    # print(i)
    while 1:
        if shuffle:  # 打乱顺序
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            # print("i + batch_size >= max_index", i + batch_size >= max_index)
            if i + batch_size >= max_index:
                i = min_index + lookback  # 超过记录序号时，从头开始

            # print("min(i+batch_size, max_index)", min(i+batch_size, max_index))
            # print(i)
            # 生成rows 数组，数组得范围在从i 开始到 i(val_gen i=201441) + batch_size(128)
            rows = np.arange(i, min(i+batch_size, max_index))
            # print("rows", len(rows))
            #  更新i len(rows)=128
            i += len(rows)
        #  创建一个smaples 数据的格式[batch_size,采样数据的个数，14个与天气有关的值]
        # lookback = 1440  # 给定10天的观测数据 step = 6  # 每6个采样一次，即每小时一个数据点,lookback // step=240
        # data.shape=(420451, 14)
        # print("data.shape", data.shape)
        # print("data.shape[-1]", data.shape[-1])
        # samples=(128, 240, 14)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        # print("samples", samples.shape)
        #  创建一个目标数据：[batch_size,],len(rows)=128
        targets = np.zeros((len(rows), ))
        # print("rows", rows.shape)
        # print("rows", rows)
        """rows=128= [ 28744 173536 115066  59059 174526  28990 144542 165588  15508 186409
 110321  43755  22324 154600  90923  54416 114964 197800 148356 134115
 147327  26281 142542  79155 149591  75313 113036 127757   6940  57598
  30089 130499 147907 133986  10072  73848  14529  71715  31429  79950
 169900   2080  28194  83687  61877 157377   9744 186772  48934  18913
  72229 120416 133776 162688 159368  74286 185674 157300 125514  17648
 197163  96377   4385  52612 111066 162294  22886 180266  19488 159053
  69850  45525  96885  79741   2352   2876 197563  36127  35245 113446
   8216 131299 160061  55150  34989  71452  55372  92441 116682  68512
   2242  62182  30612  29109 124186  52232 189276 181378 127129 143153
 102811 141481 121768 120164 163327 197452  75429 105770  75363 138662
 181136 116915 189315  59876 196834  66602 146085 138919 166072 169648
  16837 141416 150080  17634 157057   9521 103098 185660]"""
        # j= 0.1.2.3....127
        for j, row in enumerate(rows):
            # print("j", j)
            # print("rows", row)
            #  获得数据样本的index值,lookback = 1440  # 给定10天的观测数据
            # step = 6  # 每6个采样一次，即每小时一个数据点
            """range(0, 10, 3)  # 步长为 3[0, 3, 6, 9]"""
            indices = range(rows[j] - lookback, rows[j], step)
            # indices=range(298432, 299872, 6)
            # print("indices", indices)
            # print("data[indices]", data[indices])
            # print("data[indices]", data[indices].shape)
            # print("data", data.shape)
            # data[indices].shape =  (240, 14),即1440/6=240
            # "data", data.shape=420451,14
            # samples=(128, 240, 14)
            samples[j] = data[indices]
            # print("samples", samples.shape)
            #  获得对于的预测目标数据,也就是后面的第delay(144)个数据：在第二列所以是[1]
            # 这里，温度随时间序列是连续的，并且具有每天的周期性变化。因此，
            # 一种基于常识的方法就是始终预测 24 小时后的温度等于现在的温度。我们使用平均绝对误差（MAE）
            # 指标来评估这种方法
            targets[j] = data[rows[j] + delay][1]
            # print("targets", targets)
        yield samples, targets


# 计算mae,std [ 8.48043388  8.85249908  8.95324185  7.16584991 16.72731652  7.68914559 4.19808168  4.84034436
    # 2.66564926  4.25206364 42.48884277  1.53666449  2.33067298 86.61322998]


# 温度预测，评估天然的，没有用智能的准确度
def evaluate_naive_method(val_steps, val_gen, std):
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        # print("samples", samples)
        # print("samples[:, -1, 1]", samples[:, -1, 1])
        # print("targets", targets)
        """samples[:, -1, 1],所有层的最后一排的第二列
        """
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print('mae=', np.mean(batch_maes))
    print('mae=', np.mean(batch_maes)*std)


# 股票预测 p为数据库路径
def predict_stock(p):
    import os
    if os.path.isfile(p):
        float_data = stock_predict_read_data(p)  # 股票数据的读取和整理
        """检查数据中是否有inf或者nan的情况。普通numpy数组可用,np.isfinite有限返回true，否则false
        np.isfinite([np.log(-1.),1.,np.log(0)])= array([False,  True, False]),,
        np.any()是或操作，任意一个元素为True，输出为True。np.all()是与操作，所有元素为True，输出为True
"""
        print("mis", np.all(np.isfinite(float_data)))
        # temp = float_data[:, -1]  # 取每行最后一列
        # plt.figure()
        # plt.plot(range(len(temp)), temp)
        # plt.legend()
        # plt.show()
        # print(float_data.shape)
        # print(float_data.shape[0])
        x_train, y_train, x_test, y_test = stock_predict_normal(float_data)  # 数据标准化，减去平均值，除以标准值,2维转3维
        from tensorflow.keras.models import Sequential
        model = Sequential()
        history = stock_predict_dnn(x_train, y_train, model)  # 密集连接模型DNN
        # history = stock_predict_grn(x_train, y_train)  # 基于GRU的模型 门控循环单元
        # history = stock_predict_grn_optimization(x_train, y_train)  # 门控循环单元（GRU，gated recurrent unit）优化
        # history = stock_predict_grn_recurrent(x_train, y_train)  # 循环差堆叠 门控循环单元
        # history = stock_predict_grn_grn(x_train, y_train)  # 使用双向GRU
        # history = stock_predict_cnn(x_train, y_train)  # 使用一维卷积神经网络CNN
        # history = stock_predict_cnn_gru(x_train, y_train)  # 一维卷积基与GRU融合
        # 评估模型,不输出预测结果
        loss = model.evaluate(x_test, y_test)
        # loss, accuracy = model.evaluate(x_test, y_test)
        print('test loss', loss)
        # print('accuracy', accuracy)
        # 模型预测,输入测试集,输出预测结果
        y_pred = model.predict(x_test, batch_size=1)
        print('y_pred', y_pred)
        print('y_pred', y_pred.shape)
        stock_predict_plt(history)  # 股票预测可视化


# 一维卷积基与GRU融合
def stock_predict_cnn_gru(x_train, y_train):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import RMSprop
    model = Sequential()
    """filters：整数,输出空间的维数(即卷积中的滤波器数)=32.kernel_size：单个整数的整数或元组/列表,指定1D卷积窗口的长度=5"""
    model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None, x_train.shape[-1]), padding='same'))
    """pool_size:一个整数或者一个单个整数的tuple/list,表示池化窗口的大小
    input_shape = (2420, 1140, 32)
    x = tf.random.normal(input_shape)
    max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='valid')
    t = max_pool_1d(x) = (2420, 1139, 32)
    print(t.shape)
    max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2,strides=2, padding='valid')
    t = max_pool_1d(x)
    print(t.shape) = (2420, 570, 32)
    max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2,strides=1, padding='same')
    t = max_pool_1d(x)
    print(t.shape) = (2420, 1140, 32)"""
    model.add(layers.MaxPooling1D(pool_size=3, padding='same'))
    model.add(layers.Conv1D(32, 5, activation='relu', padding='same'))
    model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.1))
    model.add(layers.Dense(1))
    # model.compile(optimizer=RMSprop(), loss='mae', metrics=['accuracy'])
    """Adam：Adaptive Moment Estimation这个算法是另一种计算每个参数的自适应学习率的方法。相当于 RMSprop + Momentum
除了像 Adadelta 和 RMSprop 一样存储了过去梯度的平方 vt 的指数衰减平均值 ，也像 momentum 一样保持了过去梯度 mt
的指数衰减平均值"""
    # model.compile(optimizer=RMSprop(), loss='mae')
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x=x_train, y=y_train, epochs=100, batch_size=200, validation_split=0.1)
    return history


# 使用一维卷积神经网络CNN
def stock_predict_cnn(x_train, y_train):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import RMSprop
    model = Sequential()
    """如果padding参数为valid，这意味着将出现卷积过程中发生的自动降维，则可能会得到负尺寸。
解决方法：将padding的参数替换成same。"""
    model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(None, x_train.shape[-1]), padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2, padding='same'))
    model.add(layers.Conv1D(32, 5, activation='relu', padding='same'))
    """最大值池化层： 为了减少输出的复杂度和防止数据的过拟合，在 CNN 层之后经常会使用池化层。在我们的示例中，
    我们选择了大小为 2 的池化层。这意味着这个层的输出矩阵的大小只有输入矩阵的三分之一"""
    model.add(layers.MaxPooling1D(pool_size=2, padding='same'))
    model.add(layers.Conv1D(32, 5, activation='relu', padding='same'))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit(x=x_train, y=y_train, epochs=100, batch_size=200, validation_split=0.1)
    return history


# 使用双向GRU
def stock_predict_grn_grn(x_train, y_train):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import RMSprop
    model = Sequential()
    model.add(layers.Bidirectional(layers.GRU(32), input_shape=(None, x_train.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit(x=x_train, y=y_train, epochs=100, batch_size=200, validation_split=0.1)
    return history


# 循环差堆叠 门控循环单元
def stock_predict_grn_recurrent(x_train, y_train):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import RMSprop
    model = Sequential()
    model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.1, return_sequences=True,
                         input_shape=(x_train.shape[1], x_train.shape[-1])))
    model.add(layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.1))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit(x=x_train, y=y_train, epochs=100, batch_size=200, validation_split=0.1)
    return history


# 门控循环单元（GRU，gated recurrent unit）优化
def stock_predict_grn_optimization(x_train, y_train):
    """门控循环单元（GRU，gated recurrent unit）层的工作原理与LSTM相同。但它做了一些简化，因此运 行的计算代价更低
        （虽然表示能力可能不如LSTM）。机器学习中到处可以见到这种计算代价与表示能力之间的折中。"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import RMSprop
    model = Sequential()
    model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.1, input_shape=(x_train.shape[1], x_train.shape[-1])))
    # model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.1, input_shape=(None, x_train.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae')
    history = model.fit(x=x_train, y=y_train, epochs=150, batch_size=200, validation_split=0.1)
    return history


# 基于GRU的模型 门控循环单元
def stock_predict_grn(x_train, y_train):
    """基于GRU的模型,GRU是LSTM的简化，运算代价更低。（Gated Recurrent Unit, LSTM变体）units=32: 正整数，
        输出空间的维度.反复性神经网络，因为这样的网络能够利用数据间存在的时间联系来分析数据潜在规律进而
        提升预测的准确性，这次我们使用的反复性网络叫GRU"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import RMSprop
    model = Sequential()
    model.add(layers.GRU(32, input_shape=(x_train.shape[1], x_train.shape[-1])))
    # # model.add(layers.GRU(32, input_shape=(None, x_train.shape[-1])))
    model.add(layers.Dense(1))
    # 输出网络结果情况
    # model.summary()
    # 编译模型：优化器为optimizer='adam',metrics=['accuracy'] 评价方法为准确度，
    model.compile(optimizer=RMSprop(), loss='mae')
    # 训练模型，通过fit方法传入数据进行训练，训练10代，每代批次大小为200,每批次取训练集0.1的数据做验证集
    # history = model.fit(x=x_train, y=y_train, epochs=2, batch_size=100)
    history = model.fit(x=x_train, y=y_train, epochs=200, batch_size=200, validation_split=0.1)
    return history


# 股票数据的读取和整理
def stock_predict_read_data(p):
    import sqlite3
    with sqlite3.connect(p) as conn:
        cu = conn.cursor()
        cu.execute("select * FROM dragon_tiger ORDER BY date ASC;")
        # row = cu.fetchone()
        rows = cu.fetchall()
        # rows = cu.fetchmany(200)
        # rows = cu.fetchmany(-1) # 0到倒数1
        # results = [_[0] for _ in cu]
        # # print(results)
        # res = list(set(results))
        # res.sort(key=results.index)
        st = []
        for ii in rows:
            ii = list(ii)
            ii.pop(10)
            ii.pop(9)
            ii.pop(8)
            ii.pop(2)
            # print(ii)
            # ii[0] =2021 - 07 - 01, ii[1]= 002176
            # cu.execute("select * FROM dragon_tiger_k where date>=? and code=?", (ii[0], ii[1]))
            cu.execute("select * FROM dragon_tiger_k where date>? and code=?", (ii[0], ii[1]))
            ro = cu.fetchmany(1)
            ii.pop(1)
            ii.pop(0)
            for i in ro:
                if i[11] == "1":
                    if ii[0] == "" or ii[1] == "" or ii[2] == "" or ii[3] == "" or ii[4] == "" or ii[5] == "" or ii[6] == "" or ii[7] == "" or ii[8] == "":
                        print("数据空", i[1])
                    else:
                        # print("rr", ii[4])
                        if float(ii[4]) < 31:
                            ii.append(i[2])  # open
                            # ii.append(i[3])  # high
                            # ii.append(i[4])  # low
                            # ii.append(i[5])  # close
                            st.append(ii)
                        else:
                            print("涨幅大于30", i[1])
                else:
                    print("停牌", i[1])
        print(st[0:3])
        float_data = np.zeros((len(st), len(st[0])))
        # print(float_data[:, 1])
        for i, line in enumerate(st):
            try:
                values = [float(x) for x in line]
                float_data[i, :] = values
            except:
                print("无法float", line)
        print("float_data", float_data.shape)
        return float_data


# 数据标准化，减去平均值，除以标准值 2维转3维
def stock_predict_normal(float_data):
    t = int(float_data.shape[0] * 0.9)  # 取前2/3
    # float_data_t = float_data[:t]
    # print(float_data[:t])
    float_mean = float_data[:t].mean(axis=0)  # 11列，每一列数据的平均值
    # # print("float_mean", float_mean)
    float_data -= float_mean
    # # print("float_data", float_data)
    float_std = float_data[:t].std(axis=0)  # 11列，每一列数据的标准差
    # print("float_std1", float_std)
    float_data /= float_std
    # print("float_data2", float_data)
    x_train = float_data[:t, 0: -1]
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])  # 2维转3维
    x_test = float_data[t:, 0: -1]
    # print("x_test", x_test.shape)
    # print("x_test", x_test[0:3])
    # print("x_test", x_test.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    y_train = float_data[:t, -1]
    y_test = float_data[t:, -1]
    return x_train, y_train, x_test, y_test


# 股票预测可视化
def stock_predict_plt(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()


# 密集连接模型DNN
def stock_predict_dnn(x_train, y_train, model):
    from tensorflow.keras.optimizers import RMSprop
    """input_shape检索图层的输入形状. 仅适用于图层只有一个输入,即它是否连接到一个输入层,或者所有输入具有相同形状的情况
    ReLU函数(Rectified Linear Units)其实就是一个取最大值函数，注意这并不是全区间可导的
    激活函数是用来加入非线性因素的，解决线性模型所不能解决的问题"""
    model.add(layers.Flatten(input_shape=[x_train.shape[1], x_train.shape[-1]]))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    """Mean Absolute Error Loss平均绝对误差（MAE）是另一种常用的回归损失函数，它是目标值与预测值之差绝对值和的均值，
    表示了预测值的平均误差幅度，而不需要考虑误差的方向（注：平均偏差误差MBE则是考虑的方向的误差，是残差的和）
    metrics采用mae，代表绝对误差，即预测值和目标值的差值的绝对值。
    adam (Adaptive Moment Estimation)吸收了Adagrad（自适应学习率的梯度下降算法）和动量梯度下降算法的优点，
    既能适应稀疏梯度（即自然语言和计算机视觉问题），又能缓解梯度震荡的问题"""
    # model.compile(optimizer=RMSprop(), loss='mae')
    model.compile(loss='mae', optimizer='adam')
    # model.compile(loss='mae', optimizer='adam', metrics=['mae'])
    # model.compile(loss='mae', optimizer='adam ', metrics=['accuracy'])
    """fit(): Method calculates the parameters μ and σ and saves them as internal objects.
解释：简单来说，就是求得训练集X的均值，方差，最大值，最小值,这些训练集X固有的属性。

transform(): Method using these calculated parameters apply the transformation to a particular dataset.
解释：在fit的基础上，进行标准化，降维，归一化等操作（看具体用的是哪个工具，如PCA，StandardScaler等）。

fit_transform(): joins the fit() and transform() method for transformation of dataset.
解释：fit_transform是fit和transform的组合，既包括了训练又包含了转换。
transform()和fit_transform()二者的功能都是对数据进行某种统一处理（比如标准化~N(0,1)，将数据缩放(映射)到某个固定区间，
归一化，正则化等）fit_transform(trainData)对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等
（根据具体转换的目的），然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。

fit(x,y)传两个参数的是有监督学习的算法，fit(x)传一个参数的是无监督学习的算法，比如降维、特征提取、标准化

必须先用fit_transform(trainData)，之后再transform(testData)如果直接transform(testData)，程序会报错
如果fit_transfrom(trainData)后，使用fit_transform(testData)而不transform(testData)，虽然也能归一化，
但是两个结果不是在同一个“标准”下的，具有明显差异。(一定要避免这种情况)
"""
    history = model.fit(x=x_train, y=y_train, epochs=100, batch_size=200, validation_split=0.1)
    return history
