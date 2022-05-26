import pandas as pd
import numpy as np
import sqlite3

your_db_path = r"D:\myzq\axzq\T0002\stock_load\thesis\数学建模\my\school_math.db"  # 数据库路径


# 加载元数据 , f如果有输出总体
def load_data(db_path, f="", tab=''):
    # print(tab)
    with sqlite3.connect(db_path) as conn:
        select_table = """select * from {}""".format(tab)
        df = pd.read_sql(select_table, conn)
        # print(df)
        if f == '4xy':
            # 导入train_test_split用于拆分训练集和测试集
            from sklearn.model_selection import train_test_split
            """*arrays ：需要分割的数据，可以是list、numpy array等类型
            test_size：测试集所占的比例，取值范围在0到1之间
            train_size：训练集所占的比例，默认是等于1减去test_size
            shuffle：是否在分割之前打乱数据集，默认是True
            random_state：是随机数的种子。随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，
            保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。
            不填的话默认值为False，即每次切分的比例虽然相同，但是切分的结果不同。随机数的产生取决于种子，
            随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；种子相同，
            即使实例不同也产生相同的随机数。"""
            # 划分训练集和测试集
            x_ = df.iloc[:, 2:]  # 访问第2-69列
            # y_ = df["成交量"]
            y_ = df["收盘价"]
            x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.15, random_state=7)
            return x_train, x_test, y_train, y_test
        if f == 'xy':
            x_ = df.iloc[:, 2:]  # 访问第2-69列
            y_ = df["成交量"]
            # y_ = df["收盘价"]
            return x_, y_
        if f == 'df':
            return df

# load_data(your_db_path)


#  算法
def my_xg_boost(db_path):
    # 导入需要的库，模块以及数据
    from xgboost import XGBRegressor as XGBR
    # from sklearn.ensemble import RandomForestRegressor as RFR
    # from sklearn.linear_model import LinearRegression as LinearR
    # from sklearn.datasets import load_boston
    from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
    from sklearn.metrics import mean_squared_error as MSE
    # import pandas as pd
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from time import time
    # import datetime
    x, y = load_data(db_path, f='hh', tab='pac_all_integration_other_volume_price')
    # print(x)
    # print(y)
    # 建模，查看其他接口和属性
    xtrain, xtest, ytrain, ytest = TTS(x, y, test_size=0.3, random_state=420)
    reg = XGBR(n_estimators=100).fit(xtrain, ytrain)  # 训练
    reg.predict(xtest)  # 传统接口predict
    score = reg.score(xtest, ytest)  # R^2，shift+tab可以看函数具体原理
    print("得分", score)  # 结果：0.9050988968414799
    mse = MSE(ytest, reg.predict(xtest))
    print('mse', mse)  # 结果：8.830916343629323
    print('均方根误差rmse', mse**0.5)  # 结果：8.830916343629323
    # rr = reg.feature_importances_  # 树模型的优势之一：能够查看模型的重要性分数，可以使用嵌入法(SelectFromModel)进行特征选择
    # print(rr)
# my_xg_boost(your_db_path)


#  算法
def my_lstm(db_path):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    # from tensorflow.keras.callbacks import EarlyStopping
    # from keras.optimizers import Adam
    from tensorflow.keras.layers import LSTM
    from sklearn.metrics import mean_squared_error as MSE
    from sklearn.metrics import mean_absolute_error

    x_train, x_test, y_train, y_test = load_data(db_path, f='4xy', tab='pac_all_integration_other_volume_price')
    # print(x_train)
    # print(y_train)
    # print(y_train.shape)
    # 转换数据
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    # print(x_train)
    # print(x_train.shape)
    trainX = x_train.reshape(-1, 1, 63)
    trainY = y_train.reshape(-1, 1, 1)
    testX = x_test.reshape(-1, 1, 63)
    # testX = y_test.reshape(-1, 1, 1)
    # print(trainX)
    # print(trainY.shape)
    # x_ = df.iloc[:, 2:]  # 访问第2-69列
    # # y_ = df["成交量"]
    # y_ = df["收盘价"]
    """
    LSTM有一个可见层，它有1个输入。
    隐藏层有7个LSTM神经元。
    输出层进行单值预测。
    LSTM神经元使用Relu函数进行激活。
    LSTM的训练时间为100个周期，每次用1个样本进行训练。
    对于LSTM，输入大小应为3: batch_size、nb_timesteps、nb_features
    batch_size是一次往RNN输入的数目，比如是5。seq_len是一个句子的最大长度,input_dim是输入的维度，比如是128。"""
    model = Sequential()
    # 初始化LSTM模型，注意input_shape，它决定了投进模型的数据维度，所以设置为look_back与前面设置的映射关系相符合
    model.add(LSTM(16, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, y_train, epochs=128, batch_size=64, verbose=False)
    # model.fit(trainX, y_train, epochs=128, batch_size=256, verbose=True)
    preds = model.predict(testX)
    # print(preds)
    preds = preds.squeeze()
    # print(preds)
    # print(len(preds))
    # print(len(y_test))
    aa = mean_absolute_error(preds, y_test)
    # print("y_test：", y_test)
    mse = MSE(preds, y_test)
    print('mse', mse)
    print('均方根误差rmse', mse ** 0.5)
    print("平均绝对值误差：", aa)

    dd = pd.DataFrame()
    dd['成交量'] = y_test
    dd['成交量预测'] = preds
    # print("dd：", dd)
    # 写入excel文件
    # dd.to_excel(r"D:\myzq\axzq\T0002\stock_load\thesis\数学建模\my\成交量预测结果.xlsx")
# my_lstm(your_db_path)


#  定义全局函数的部分。
def error(y_predict, y_test):  # 定义计算误差平方和函数，传入的是估算出的值，和测试值
    errs = []
    y_test = list(y_test)
    le = len(y_predict)
    # print("len(y_predict", len(y_predict))
    # print("y_predict[0]", y_predict[0][0])
    print("y_test", len(y_test))
    print("y_test[0]", list(y_test)[0])
    # print("y_test", y_test)
    for i in range(le):
        e = (y_predict[i][0] - y_test[i]) ** 2
        errs.append(e)
    # return (sum(errs)/le)**0.5  # 返回根方误差
    return sum(errs)


#  定义全局函数的部分。
def error2(y_predict, y_test):  # 定义计算误差平方和函数，传入的是估算出的值，和测试值
    errs = []
    # y_test = list(y_test)
    le = len(y_predict)
    # print("len(y_predict", le)
    # print("y_predict[0]", y_predict[0])
    # print("y_test", len(y_test))
    # print("y_test[0]", y_test[0])
    # print("y_test", y_test)
    for i in range(le):
        e = (y_predict[i] - y_test[i]) ** 2
        errs.append(e)
    # return (sum(errs)/le)**0.5  # 返回根方误差
    return sum(errs)


def pls_regression(db_path):
    # from sklearn import preprocessing
    from sklearn.cross_decomposition import PLSRegression  # 偏最小二乘法的实现
    # 导入train_test_split用于拆分训练集和测试集
    from sklearn.model_selection import train_test_split
    # import csv
    # from sklearn.cross_validation import train_test_split
    # from sklearn.decomposition import RandomizedPCA
    # import math
    import matplotlib.pyplot as plt

    # 偏最小二乘法的实现部分。
    df = load_data(db_path, f='df', tab='all_integration_other_volume_price')
    # x_train, x_test, y_train, y_test = load_data(db_path, f='df', tab='all_integration_other_volume_price')
    df = df.iloc[:-48, 2:]
    dfx = df.iloc[:, 2:]
    del dfx['收盘价']
    del dfx['最高价']
    del dfx['最低价']
    del dfx['成交额']
    # print(dfx)
    dfy = df['收盘价']
    # print(dfy)
    # dfy = df['成交量']
    x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.9, random_state=7)
    # print(x_train)
    # print(y_train)
    print(x_train.shape)
    print(y_train.shape)
    x_train2, y_train2 = np.array(x_train), np.array(y_train)
    # print("x_train", x_train)
    # print("y_train2", y_train2)
    # print("y_train2", len(y_train2))
    # return ''
    n_components = 0
    while n_components < x_train.shape[1]:
        n_components += 1  # 在第一遍的时候n_components是1，第二遍循环的时候是2，第n遍循环是n，最大是x的列数，也就是特征的个数，
        pls2 = PLSRegression(n_components=n_components)  # 计算SS （SS这个是全建模 ， PRESS是减去一个进行建模
        # 这个不是偏最小二乘法吗？？？,,这里是循环计算主成分的个数，直到达到满意的精度。
        pls2.fit(x_train, y_train)  # fit也是一个函数，，，两个参数，第一个参数是训练集，第二个参数是目标。
        y_predict0 = pls2.predict(x_train)  # 预测给定训练样本的目标。
        # print("y_predict0", y_predict0)
        SS = error(y_predict0, y_train)  # 这里是预测的值和真正的值之间的误差大小。
        print("ss", SS)
        y_predict1 = []  # 根据模型得到的y的预测值，实际上这个模型是留一法建立的模型。
        # break
        for i in range(x_train.shape[0]):  # 计算PRESS，，，，这个是x_train的行数
            n_components1 = n_components + 1  # 但是我不明白，为什么这里还要加1，主成分不可能是0个吧，所以就从1开始了。
            """np.delete(array,obj,axis)obj:需要处理的位置，比如要删除的第一行或者第一行和第二行
            axis:如果输入为0：按行删除,如果输入为1：按列删除"""
            # print("i:", i)
            x_train1 = np.delete(x_train2, i, 0)  # 删除第i行，，，这里是标准化的数组。留一法的实现
            y_train1 = np.delete(y_train2, i, 0)  # 这个也是删除第i行，这里都是经过标准化的（但这个x是经过标准化的，y却没有用标准化的数据）。，，这个没有用到过，是不是这里写错了？？
            # print("n_components1", n_components1)  #  第一个循环这里n_components1是1
            # print("x_train_st1", len(x_train1))
            # print("y_train_st1", len(y_train1))
            # break
            pls2 = PLSRegression(n_components=n_components1)  # 偏最小二乘法参数的设置，，，这里面一般有5个参数，，但这里只传入了主成分的个数。
            # 参数1：n_components:int ,(default 2) ,,要保留的主成分数量，默认为2
            # 参数2：scale:boolean,(default True),,是否归一化数据，默认为是
            # 参数3：max_iter: an integer,(default 500),,使用NIPALS时的最大迭代次数
            # 参数4：tol： non-negative real（default 1e-06）,,迭代截止条件
            # 参数5：copy： Boolean，（default True）,,
            pls2.fit(x_train1, y_train1)  # 这里是根据前面设置好的参数建模过程，这里的建模过程是不是不太对（这里x是归一化的，y并没有用归一化的），应该都是用归一化的才行呀。？？？

            # 这里主要是进行了数据格式的转换，因为做预测要传进的是矩阵【格式很重要】
            x_train_st_i = [x_train2[i].tolist()]  # 用之前要进行清空，这个很重要。用一个参数之前要进行清空。
            # print('x_train2[i].tolist() ', x_train2[i].tolist())
            # print('x_train_st_i: ', x_train_st_i)  # 输出一下变量，查看格式是否正确，因为下面的predict函数需要用[[1，1，1，1，1，1]] 这种格式的数据
            """预测单条数据"""
            y_predict11 = pls2.predict(x_train_st_i)  # 预测函数，给定之前留一法没有用到的样本，进行建模，预测对应的y值。？可是已经删除了不是吗？ZHE这句出了一点问题？？？？就是数据格式有问题，需要在最外面在加一个[]
            # print('y_predict11:', y_predict11)
            # print('y_predict11:', y_predict11[0][0])
            y_predict1.append(y_predict11[0][0])
            # if i > 3:
            #     break
        print("len(y_predict1)", len(y_predict1))
        # print("y_predict1", y_predict1)
        PRESS = error2(y_predict1, y_train2)  # 可能错误：https://blog.csdn.net/little_bobo/article/details/78861578
        print("PRESS", PRESS)  # PRESS 187.8897154105033  ss 187.45942933260253
        Qh = 1 - float(PRESS / SS)  # PRESS比ss增加一个特征维度，误差会小于ss
        print("Qh", Qh)
        # break
        if Qh < 0.0985:  # 判断精度 模型达到精度要求，可以停止主成分的提取了。
            plt.figure(1)
            plt.scatter(y_predict0, y_train)  # 画了一个图，这个图是预测值，与测量值的图？？？
            plt.figure(2)
            plt.scatter(y_predict1, y_train)
            print('the Qh is ', Qh)
            print('the PRESS is', PRESS)
            print('the SS is', SS)
            break
    return ''
    print('n_components is ', n_components + 1)  # 这里为什么要加1？？？，，，难道因为计数是从0开始的？？
    SECs = []
    errors = []
    e = 100
    for i in range(10):  # 循环测试
        # print i
        x_train, x_test, y_train, y_test = train_test_split(RON, A, test_size=0.5)  # 划分训练集与测试集，这个是一个内置的函数。
        x_test_st = preprocessing.scale(x_test)  # 数据标准化
        y_predict = pls2.predict(x_test_st)  # 进行预测
        SECs.append(np.sqrt(error(y_predict, y_test) / (y_test.shape[0] - 1)))
        errors.append(float(error(y_predict, y_test)))
        if SECs[-1] < e:
            y_predict_min = y_predict
            y_test_min = y_test

    print('the prediced value is ', y_predict.T)  # 画图，打印结果
    print('the true value is', y_test)
    print('the mean error is', float(np.mean(errors)))
    print("the mean SEC is ", float(np.mean(SECs)))

    # plt.figure(3)
    # plt.scatter(y_predict_min, y_test_min)
# pls_regression(your_db_path)


def _calculate_vips(model):
    t = model.x_scores_
    w = model.x_weights_
    q = model.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))  # np.zeros()表示初始化0向量
    s = np.diag(np.matmul(np.matmul(np.matmul(t.T, t), q.T), q)).reshape(h, -1)
    # np.matmul(a,b)表示两个矩阵相乘;np.diag()输出矩阵中对角线上的元素，若矩阵是一维数组则输出一个以一维数组为对角线的矩阵
    total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
        # np.linarg.norm()表示求范数：矩阵整体元素平方和开根号，不保留矩阵二维特性
        vips[i] = np.sqrt(p * (np.matmul(s.T, weight)) / total_s)
        # s.T表示矩阵的转置
    return vips


def pls_regression2(db_path):
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # plt.style.use('seaborn')
    from sklearn.cross_decomposition import PLSRegression
    # 导入train_test_split用于拆分训练集和测试集
    from sklearn.model_selection import train_test_split

    df = load_data(db_path, f='df', tab='all_integration_other_volume_price')
    # x_train, x_test, y_train, y_test = load_data(db_path, f='df', tab='all_integration_other_volume_price')
    df = df.iloc[:-48, 2:]
    dfx = df.iloc[:, 2:]
    del dfx['收盘价']
    del dfx['最高价']
    del dfx['最低价']
    del dfx['成交额']
    # print(dfx)
    dfy = df['收盘价']
    # print(dfy)
    # dfy = df['成交量']
    x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.1, random_state=7)
    # print(x_train)
    # print(y_train)
    # print(x_train.shape)
    # print(y_train.shape)
    # x_train2, y_train2 = np.array(x_train), np.array(y_train)
    pls = PLSRegression(n_components=80)
    # Fit data：拟合数据
    pls.fit(x_train, y_train)
    # Ypredict = pls.predict(x_train)
    # print("Ypredict", Ypredict)
    # Ypredict = Ypredict.flatten()
    # print("Ypredict", Ypredict)
    # 真实值与预测值的确定系数，越接近于1越好
    R2Y = pls.score(x_train, y_train)
    print("R2Y", R2Y)
    # return ''
    # 变量重要性分析，变量对y的影响程度排序，一般认为大于1是有影响的
    df_vip = pd.DataFrame()
    df_vip['X'] = x_train.columns
    # print(df_vip)
    df_vip['vip'] = _calculate_vips(pls)
    # print(df_vip)
    # vip = df_vip.sort_values(by='vip', ascending=True).tail(200)
    # print(vip)
    # VIP的可视化
    # plt.figure(figsize=(16, 16))
    # plt.barh(vip.X, vip.vip, height=0.5)
    # plt.title('VIP')
    # plt.figure(figsize=(15, 8))
    # length = range(len(y_train))
    # plt.plot(length, y_train, marker='o', label='target_y')
    # plt.plot(length, Ypredict, marker='o', label='target_y_predict')
    # plt.legend()
    # df_vip['coef'] = pls.coef_.flatten()
    # df_vip = df_vip.sort_values(by='vip', ascending=False).iloc[150:200]
    # df_vip = df_vip.sort_values(by='vip', ascending=False).iloc[100:150]
    # df_vip = df_vip.sort_values(by='vip', ascending=False).iloc[50:100]
    df_vip = df_vip.sort_values(by='vip', ascending=False)
    # df_vip = df_vip.sort_values(by='vip', ascending=False).iloc[50:80]
    # df_vip = df_vip.sort_values(by='vip', ascending=False).iloc[0:50]
    # df_vip = df_vip.sort_values(by='vip', ascending=False).round(4).head(80)
    print(df_vip)
    with sqlite3.connect(db_path) as conn:
        df_vip.to_sql('数字经济收盘价影响分析', con=conn, if_exists='replace', index=False)

# pls_regression2(your_db_path)


