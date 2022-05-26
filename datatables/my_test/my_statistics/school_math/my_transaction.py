import pandas as pd
import numpy as np
import sqlite3
your_db_path = r"D:\myzq\axzq\T0002\stock_load\thesis\数学建模\school_math.db"  # 数据库路径


# 加载元数据 , f如果df输出总体
def load_data(db_path, f="", tab=''):
    # print(tab)
    with sqlite3.connect(db_path) as conn:
        select_table = """select * from {}""".format(tab)
        # select_table = """select * from {} order by 时间 asc""".format(tab)
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
            x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.5, shuffle=False, random_state=1)
            print("x_tra", x_train)
            print("y_tra", y_train)
            return x_train, x_test, y_train, y_test
        if f == 'xy':
            x_ = df.iloc[:, 2:]  # 访问第2-69列
            y_ = df["成交量"]
            # y_ = df["收盘价"]
            return x_, y_
        if f == 'df':
            return df
# load_data(your_db_path)


# 计算预测涨幅
def create_up_rate(db_path, tab):
    with sqlite3.connect(db_path) as conn:
        # tab = 'predict_price'
        select_table = """select * from {}""".format(tab)
        df = pd.read_sql(select_table, conn)
        df = df.iloc[:, 1:]  # 去 序号
        # print(df)
        dat = []
        for i in range(df.shape[0]-1):
            x = df.iloc[i, -1]  # 预测值
            y = df.iloc[i+1, -2]
            # print(x)
            # print(x.flatten())
            # print(y)
            up = (x - y)/y
            # print(up)
            dat.append(round(up, 5))
            # if i == 0:
            #     break
        # print(dat)
        # print(len(dat))
        dat.append(0)
        return dat
# create_up_rate(your_db_path, '')


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    # look_back = 2
    dataX, dataY = [], []
    # print(len(dataset)-look_back-1)  # test: 1868-1-1=1866
    for i in range(len(dataset)-look_back-1):
        x = dataset[(i+1):(i+1+look_back), 1:]
        y = dataset[i, 0]
        # print(x)
        # print(x.flatten())
        # print(y)
        dataX.append(x)
        dataY.append(y)
        # if i == 2:
        #     break
    # print(dataX)
    # print(np.array(dataX))
    # print(dataY)
    return np.array(dataX), np.array(dataY)


#  算法
def my_lstm_0104(db_path):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import LSTM
    from sklearn.metrics import mean_absolute_error

    # df = load_data(db_path, f='df', tab='data_5')
    with sqlite3.connect(db_path) as conn:
        tab = 'data_5'
        select_table = """select 序号,时间,收盘价,开盘价,最高价,最低价,成交量,成交额 from {}""".format(tab)
        df = pd.read_sql(select_table, conn)
        df2 = df.iloc[:, 2:]  # 去 序号,时间,
        # print(df2)
        scaler = MinMaxScaler(feature_range=(0, 1))
        df2 = scaler.fit_transform(df2)  # x,y总体归一化
        df_len = len(df2)
        print(df_len)
        # print(df)
        test_size = int(df_len * 0.17)  # 划分训练集和测试集
        # test_size = len(dfx) - train_size
        test, train = df2[0:test_size], df2[test_size:df_len]
        # print(test)
        print(len(test))
        print(len(train))
        look_back = 100  # 加多少条数据去预测
        x_test, y_test = create_dataset(test, look_back)
        x_train, y_train = create_dataset(train, look_back)
        print(x_test.shape)
        print(x_train.shape)
        print(y_test.shape)
        print(y_train.shape)
        # return ''
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
        model.add(LSTM(16, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=150, batch_size=256, verbose=False)
        # model.fit(trainX, trainY, epochs=2, batch_size=128, verbose=False)
        preds = model.predict(x_test)
        # print(preds)
        preds = preds.squeeze()
        # print(preds)
        # print(y_test)
        print(len(preds))
        pp = pd.DataFrame()
        pp['1'] = preds
        pp['2'] = preds
        pp['3'] = preds
        pp['4'] = preds
        pp['5'] = preds
        pp['6'] = preds
        pp2 = pd.DataFrame()
        pp2['1'] = y_test
        pp2['2'] = y_test
        pp2['3'] = y_test
        pp2['4'] = y_test
        pp2['5'] = y_test
        pp2['6'] = y_test
        # print(pp)
        test_predict = scaler.inverse_transform(pp)
        test_predict = test_predict[:, 0]
        # print(test_predict)
        test_predict_len = len(test_predict)
        print(test_predict_len)
        # print(test_predict[:, 0])
        test_y = scaler.inverse_transform(pp2)
        test_y = test_y[:, 0]
        # print(test_predict2)
        # print(test_predict2[:, 0])
        # return ''
        aa = mean_absolute_error(test_predict, test_y)
        print("平均绝对值误差：", aa)
        bb = mean_squared_error(test_predict, test_y)
        print("MSE(均方差)：", bb)
        print("RMSE(均方根：", bb**0.5)
        df_vip = pd.DataFrame()
        # print("时间：", df['时间'].iloc[:test_predict_len])
        df_vip['时间'] = df['时间'].iloc[:test_predict_len]
        df_vip['开盘价'] = df['开盘价'].iloc[:test_predict_len]
        df_vip['收盘价'] = test_y
        df_vip['收盘价预测'] = test_predict
        # print(df_vip)
        tab2 = "predict_price"
        df_vip.to_sql(tab2, con=conn, if_exists='replace', index=True)
        dat = create_up_rate(db_path, tab2)
        df_vip['预测涨幅'] = dat
        # print(df_vip)
        # df_vip.to_sql('predict_price_up', con=conn, if_exists='replace', index=True)
# my_lstm_0104(your_db_path)


#  算法
def my_lstm2_0104(db_path):
    # import matplotlib.pyplot as plt
    # import math
    from sklearn.preprocessing import MinMaxScaler
    # from sklearn.metrics import mean_squared_error
    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.layers import Dense
    # from tensorflow.keras.layers import LSTM
    # from sklearn.metrics import mean_absolute_error
    # from sklearn.model_selection import train_test_split

    # df = load_data(db_path, f='df', tab='data_5')
    with sqlite3.connect(db_path) as conn:
        tab = 'data_5'
        select_table = """select 序号,时间,收盘价,开盘价,最高价,最低价,成交量,成交额 from {}""".format(tab)
        df = pd.read_sql(select_table, conn)
        df2 = df.iloc[:, 2:]  # 去 序号,时间,
        print(df2)
        scaler = MinMaxScaler(feature_range=(0, 1))
        df2 = scaler.fit_transform(df2)  # x,y总体归一化
        df_len = len(df2)
        print(df_len)
        # print(df)
        test_size = int(df_len * 0.3)  # 划分训练集和测试集
        # test_size = len(dfx) - train_size
        test, train = df2[0:test_size], df2[test_size:df_len]
        print(test)
        print(len(test))
        print(len(train))
        look_back = 1
        x_test, y_test = create_dataset(test, look_back)
        return ''
        # testX, testY = create_dataset(test, look_back)
        # step = 2
        # x_test, x_train = x_test.reshape(-1, step, col_num_x), x_train.reshape(-1, step, col_num_x)
        # print("x_test2:", len(x_test), x_test)
        return ''
        trainY = y_train.reshape(-1, 1, 1)
        testX = x_test.reshape(-1, 1, 63)
        testY = y_test.reshape(-1, 1, 1)
        print(trainX.shape)
        # print(trainY)
        print(trainY.shape)
        # return ''
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
        model.add(LSTM(4, input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, y_train, epochs=1, batch_size=256, verbose=False)
        # model.fit(trainX, trainY, epochs=2, batch_size=128, verbose=False)
        preds = model.predict(testX)
        preds = preds.squeeze()
        print(preds)
        print(len(preds))
        aa = mean_absolute_error(preds, y_test)
        print("平均误差：", aa)
        # print(y_test)
        # return ''
        print(y_test.shape)
        df_vip = pd.DataFrame()
        df_vip['收盘价'] = y_test
        # df_vip['收盘价'] = y_train_change.iloc[0:3]
        # df_vip['predict'] = [2, 3, 4]
        df_vip['predict'] = preds
        # print(df_vip)
        day_open = df.iloc[(len(preds)-1):, :2]
        day_open.reset_index(inplace=True, drop=True)
        # print(day_open)
        df_all = pd.concat([day_open, df_vip], axis=1)
        print(df_all)
        print(df_all.iloc[:, -2:])
# my_lstm2_0104(your_db_path)


# 计算5分钟收益率和剩余资产
def transaction_5(db_path):
    with sqlite3.connect(db_path) as conn:
        tab = 'predict_price_up'
        select_table = """select * from {} where 预测涨幅>0.01""".format(tab)
        df = pd.read_sql(select_table, conn)
        # df2 = df.iloc[:, 2:]  # 去 序号,时间,
        print(df)
        df_open = df['开盘价']
        df_end = df['收盘价']
        surplus_li = []
        earning_rate_li = []
        for d in range(df.shape[0]):
            surplus = (100/df_open.iloc[d])*df_end.iloc[d]*0.997
            surplus_li.append(round(surplus, 4))
            earning_rate = (surplus-100)/100
            earning_rate_li.append(round(earning_rate, 4))
            # print(df_open.iloc[d])
            # print(df_end.iloc[d])
        df['剩余'] = surplus_li
        df['收益率'] = earning_rate_li
        print(df)
        # df..to_sql('surplus_earning_rate', con=conn, if_exists='replace', index=False)
# transaction_5(your_db_path)


# 计算日收益率和日收益率
def transaction_day(db_path):
    with sqlite3.connect(db_path) as conn:
        tab = 'surplus_earning_rate'
        select_table = """select * from {}""".format(tab)
        df = pd.read_sql(select_table, conn)
        # df2 = df.iloc[:, 2:]  # 去 序号,时间,
        # print(df)
        df_day = df['时间'].str[0:10]
        df['时间'] = df_day
        df = df.iloc[:, 2:]
        # print(df_day)
        print(df)
        """subset:  用来指定特定的列，默认所有列
        keep: {‘first’, ‘last’, False}, default ‘first’ 删除重复项并保留第一次出现的项
        inplace: boolean, default False 是直接在原来数据上修改还是保留一个副本"""
        df_day = df_day.drop_duplicates(keep='first', inplace=False)
        day_up = []
        for dd in df_day.values:
            # d_day = d[0]
            up_d = 0
            for d in df.values:
                # print(d[0], d[-1])
                if dd == d[0]:
                    up_d += d[-1]
                # break
            aa = [dd, up_d]
            # print(dd)
            # print(aa)
            day_up.append(aa)
            # break
        print(day_up)
        day_up2 = pd.DataFrame(data=day_up, columns=['时间', '日收益率'])
        print(day_up2)
        # day_up2..to_sql('day_up', con=conn, if_exists='replace', index=False)
# transaction_day(your_db_path)


# 计算日收益率和中证500日收益率
def transaction_500(db_path):
    with sqlite3.connect(db_path) as conn:
        tab = 'domestic_indicator'
        select_table = r"""select 时间,中证500指数 from {} where 时间>='2022-01-04 00:00:00'""".format(tab)
        df = pd.read_sql(select_table, conn)
        dfd = df['时间'].str[0:10]  # 去 序号,时间,
        # print(dfd)
        df_500 = df['中证500指数']
        # print(df_500)
        # return ''
        """subset:  用来指定特定的列，默认所有列
        keep: {‘first’, ‘last’, False}, default ‘first’ 删除重复项并保留第一次出现的项
        inplace: boolean, default False 是直接在原来数据上修改还是保留一个副本"""
        # df_day = df_day.drop_duplicates(keep='first', inplace=False)
        tab2 = 'day_up'
        select_table2 = r"""select * from {}""".format(tab2)
        df3 = pd.read_sql(select_table2, conn)
        print(df3)
        # print(df3['时间'])
        day_up2 = []
        for i in range(df_500.shape[0]-1):
            for bb in df3['时间'].values:
                # print("bb", bb == dfd[i])
                if bb == dfd[i]:
                    # pass
                    aa = (df_500[i]-df_500[i+1])/df_500[i+1]  # 中证500收益
                    # print(df_500[i])
                    # print(df_500[i+1])
                    # print(aa)
                    day_up2.append(aa)
            # break
        day_up2.append(0)
        # print(day_up2)
        df3['500涨幅'] = day_up2
        df3['超额收益'] = 0
        df3['超额标准差'] = 0
        df3['日均超额'] = 0
        df3['信息比率'] = 0
        print(df3)
        # df3..to_sql('day_up_500', con=conn, if_exists='replace', index=False)
# transaction_500(your_db_path)


# 计算日收益率和中证500日收益率
def transaction_500_information(db_path):
    with sqlite3.connect(db_path) as conn:
        tab = 'day_up_500'
        select_table = r"""select * from {}""".format(tab)
        df = pd.read_sql(select_table, conn)
        # print(df)
        df_day = df['日收益率']
        df_day_500 = df['500涨幅']
        # print(df_day)
        # print(df_day_500)
        day_up = []  # 超额收益
        day_up2 = []  # 日均超额收益率
        for i in range(df_day.shape[0]):
            aa = (df_day[i]-df_day_500[i]*0.9)  # 超额收益
            aa2 = aa/df_day.shape[0]  # 日均超额收益率
            # print(aa)
            day_up.append(aa)
            day_up2.append(aa2)
            # break
        df['超额收益'] = day_up
        df['日均超额'] = day_up2
        # print(df)
        # print(np.std(day_up))   # 超额收益标准差
        # print(np.array(day_up).std())
        day_up_std = np.std(day_up)   # 超额收益标准差
        day_up3 = []  # 信息比率
        for ii in day_up2:
            aa3 = ii/day_up_std  # 信息比率
            # print(aa)
            day_up3.append(aa3)
            # break
        df['信息比率'] = day_up3
        print(df)
        # return ''
        df.to_sql(tab, con=conn, if_exists='replace', index=False)
# transaction_500_information(your_db_path)


# 计算5分钟收益率和剩余资产,修正4问第一小问错误,并计算回撤率
def change_transaction_5(db_path):
    with sqlite3.connect(db_path) as conn:
        tab = 'predict_price_up'
        select_table = """select * from {} where 预测涨幅>0.01 order by 时间 asc""".format(tab)
        df = pd.read_sql(select_table, conn)
        # df2 = df.iloc[:, 2:]  # 去 序号,时间,
        # print(df)
        # return ''
        df_open = df['开盘价']
        df_end = df['收盘价']
        surplus_li = []  # 剩余资产
        earning_rate_li = []  # 回撤率
        first_assets = 100  # 初始资产100
        # surplus_assets = 0
        for d in range(df.shape[0]):
            surplus = (first_assets / df_open.iloc[d]) * df_end.iloc[d] * 0.997
            # surplus_assets = surplus
            surplus_li.append(round(surplus, 4))
            earning_rate = (surplus - first_assets) / first_assets  # 回撤率
            earning_rate_li.append(round(earning_rate, 4))
            first_assets = surplus  # 亏：初始资产变小
        df['剩余'] = surplus_li
        df['回撤率'] = earning_rate_li
        df = df.sort_values(by='时间', ascending=False)
        print(df)
        # df..to_sql('change_surplus_earning_rate', con=conn, if_exists='replace', index=False)


# change_transaction_5(your_db_path)


def open_end_pca(db_path):
    # 加载元数据 f="hh"输出总体
    df = load_data(db_path, f="df", tab="data_5")
    df_pca = load_data(db_path, f="df", tab="pac_all_integration_other_volume_price")
    df_pca = df_pca.iloc[:, 2:]
    open_price = df.iloc[:, 1:6]
    open_price = open_price[:-48]  # 去空行
    # print(open_price)
    # print(df_pca)
    # return ''
    # 把标准化数据更新回原表
    df_all = pd.concat([open_price, df_pca], axis=1)
    print(df_all)
    with sqlite3.connect(db_path) as conn:
        pass
        """如果不存在就创建表"""
        # df_all.to_sql('open_pac_all_integration_other_volume_price', con=conn, if_exists='replace', index=True)
        # df_all.to_sql('open_pac_all_integration_other_volume_price', con=conn, if_exists='replace', index=False)
# open_end_pca(your_db_path)
