from numpy import *


# 约会网站配对
def machine_learning_in_action_ch02_2():
    # import pandas as pd
    # import numpy as np
    # import matplotlib.pyplot as plt

    pp = r"D:\ana\envs\py36\mywagtailone\datatables\my_test\machinelearninginaction\Ch02\datingTestSet2.txt"

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
            # print(line)
            line = line.strip()
            # print(line)
            listFromLine = line.split('\t')  # 按中间跳格符\t分割为list。 \t 跳格  \r 回车 \n 换行
            # print(listFromLine[0:3])
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))  # -1表示列表中的最后一列元素，
            index += 1
        return returnMat, classLabelVector

    def get_plot():
        import matplotlib.pyplot as plt
        fig = plt.figure()
        # ax = fig.add_subplot(321)  # row = 3, col = 2, index = 1
        ax = fig.add_subplot(111)  # row = 1, col = 1, index = 1  index表示子图位置，it represents the location of son picture
        datingDataMat, datingLabels = file2matrix(pp)
        """
        matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None,
                          vmin=None, vmax=None, alpha=None, linewidths=None,
                          verts=None, edgecolors=None, hold=None, data=None,
                          **kwargs)
        x, y对应了平面点的位置，
        s控制点大小，
        c对应颜色指示值，也就是如果采用了渐变色的话，我们设置c=x就能使得点的颜色根据点的x值变化，
        cmap调整渐变色或者颜色列表的种类
        marker控制点的形状
        alpha控制点的透明度"""
        # ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
        ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
        ax.axis([-2, 25, -0.2, 2.0])
        plt.xlabel('Percentage of Time Spent Playing Video Games')
        plt.ylabel('Liters of Ice Cream Consumed Per Week')
        plt.show()


    def datingClassTest():
        hoRatio = 0.10  # 设置测试集比重，前10%作为测试集，后90%作为训练集
        # datingLabels为1，2，3标签
        datingDataMat, datingLabels = file2matrix(pp)
        print(datingLabels)
        # print(datingDataMat)
        # normMat, ranges, minvals = autoNorm(datingDataMat)
        # # print(normMat)
        # m = normMat.shape[0]  # 得到样本数量m=1000
        # # print(m)
        # numTestVecs = int(m * hoRatio)  # 得到测试集最后一个样本的位置=100
        # # print(numTestVecs)
        # errorCount = 0.0  # 初始化定义错误个数为0
        # for i in range(numTestVecs):
        #     # 测试集中元素逐一放进分类器测试，k = 3
        #     classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        #     # 输出分类结果与实际label
        #     print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        #     # 若预测结果与实际label不同，则errorCount+1
        #     if (classifierResult != datingLabels[i]):
        #         errorCount += 1.0
        #     # 输出错误率 = 错误的个数 / 总样本个数
        #     print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
        #
    datingClassTest()
    get_plot()
