from mywagtailone.datatables.my_test import main_test
if __name__ == "__main__":
    # p = r"D:\ana\envs\py36\mywagtailone\datatables\datatable.db"
    # p = r"D:\my2\user.db"
    # main_test.test_sq_lite(p)
    # main_test.test_my()
    # main_test.zip_ya(r"D:\myzq\axzq\T0002", r"E:\T0002", "dd")
    # main_test.test_trade_save()
    # main_test.test_read_xls()
    # 深圳
    # net = "http://datainterface3.eastmoney.com/EM_DataCenter_V3/api/YYBJXMX/GetYYBJXMX?js=&sortfield=&sortdirec=-1&pageSize={}&pageNum=1&tkn=eastmoney&salesCode=80601499&tdir=&dayNum=&startDateTime={}&endDateTime={}&cfg=yybjymx"
    # 上海
    # net = "http://datainterface3.eastmoney.com/EM_DataCenter_V3/api/YYBJXMX/GetYYBJXMX?js=&sortfield=&sortdirec=-1&pageSize={}&pageNum=1&tkn=eastmoney&salesCode=80403915&tdir=&dayNum=&startDateTime={}&endDateTime={}&cfg=yybjymx"
    # main_test.east_dragon_tiger_new(net, 3, "2021-09-26", "2021-09-27", f="")
    # 机构,只能1月求一次
    # net = "http://data.eastmoney.com/DataCenter_V3/stock2016/DailyStockListStatistics/pagesize={},page=1,sortRule=-1,sortType=PBuy,startDate={},endDate={},gpfw=0,js=.html?rt=26985157"
    # main_test.east_dragon_tiger_new(net, 1000, "2021-07-01", "2021-07-31", f="institution")
    # main_test.dragon_tiger_into_tdx()
    # main_test.read_file(r"D:\my2\backup\col_cfgwarn.dat")
    # t = main_test.dragon_tiger_date_mark(r"D:\ana\envs\py36\mywagtailone\datatables\datatable.db")
    # t = main_test.test_tiger_code(r"D:\ana\envs\py36\mywagtailone\datatables\datatable.db", "2021-07-01")
    # par = main_test.test_get_k(t, "2021-07-01", "2021-07-05")
    # print(par)
    # # par = 0
    # main_test.test_scatter_diagram(par)
    main_test.dragon_tiger_add_mark(r"D:\ana\envs\py36\mywagtailone\datatables\datatable.db")

    def stock_robot_test(p):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from tensorflow import keras
        import tensorflow as tf
        from tensorflow.keras.layers import Dropout, Dense, SimpleRNN
        import os
        from sklearn.preprocessing import MinMaxScaler
        #  test_date [630: 729]=100天  0308(731-632)3524
        # test_set 630 - 60 = 570 = > [570: 729, 2: 3]=160   (572, 2020/12/4,  3436.7291)
        # training_set 0305 3463
        # test_set[i - 60:i, 0] = test_set[0:60, 0]  最后0305 3463
        # test_set[60, 0] 0308(731-632)3524
        test_date, test_set, training_set = read_data(p, pd)  # 读取数据21-07-30 = 3398
        # print("eeeeeeeeeee")
        # print(test_set[0:60, 0])
        # print(test_set[60, 0])
        # 归一化
        sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
        # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
        training_set_scaled = sc.fit_transform(training_set)
        # print(training_set_scaled[:10])
        test_set = sc.transform(test_set)  # 利用训练集的属性对测试集进行归一化
        # print(test_set[:10])
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        # 提取训练集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签
        # [i - 60:i, 0]= 0到59共60个。[i, 0]=》 60=i=第61个数,  len(training_set_scaled)=630
        for i in range(60, len(training_set_scaled)):
            x_train.append(training_set_scaled[i - 60:i, 0])
            y_train.append(training_set_scaled[i, 0])
        # print(x_train)
        # print(y_train)
        # 对训练集进行打乱
        np.random.seed(7)
        np.random.shuffle(x_train)
        np.random.seed(7)
        np.random.shuffle(y_train)
        # print(x_train)
        # print(y_train)
        # tf.random.set_seed: 设置全局随机种子。上面写法是tensorflow2 .0 的写法，如果是tensorflow1 .0
        # 则为：set_random_seed（）
        tf.random.set_seed(7)
        # 将训练集由list格式变为array格式
        x_train, y_train = np.array(x_train), np.array(y_train)
        # print(x_train)
        # print(y_train)
        # 使x_train符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。
        # 此处整个数据集送入，送入样本数为x_train.shape[0]即2066组数据；输入60个开盘价，预测出第61天的开盘价，
        # 循环核时间展开步数为60; 每个时间步送入的特征是某一天的开盘价，只有1个数据，故每个时间步输入特征个数为1
        # 送入样本数为x_train.shape[0] 即60个时间步长，每个步长都具有1个特征。因此，我希望有60个simpleRNN单元。
        # print(x_train.shape)
        x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))
        # print(x_train.shape)
        # print(x_train)
        # 利用for循环，遍历整个测试集，提取测试集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签，
        # len(test_set) = 160 =>i最高159 =》[i - 60:i, 0]i=159时取158(少1)。 [i, 0]最高[159,0]
        # 0表示第0列.  test_set[i - 60:i, 0]= test_set[0:60, 0]
        # test_set 630 - 60 = 570 = > [570: 729, 2: 3]=160   (572, 2020/12/4,  3436.7291)
        # test_set[i - 60:i, 0] = test_set[0:60, 0]  最后0305 3463
        # test_set[60, 0] = 0308(731-632)3524
        last = test_set[-60:]
        for i in range(60, len(test_set)):
            x_test.append(test_set[i - 60:i, 0])
            y_test.append(test_set[i, 0])
        # 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))
        dd = "d"
        if dd:
            model = tf.keras.Sequential([
                SimpleRNN(80, return_sequences=True),
                Dropout(0.15),
                SimpleRNN(100),
                Dropout(0.15),
                Dense(1)
            ])
            # optimizer='adam'优化算法是梯度下降法,学习率为0.001 # 损失函数用均方误差. matrics=['acc']
            # 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值
            model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')
            checkpoint_save_path = r"D:\my2\my_test\rnn_stock.ckpt"
            if os.path.exists(checkpoint_save_path + '.index'):
                print('-------------load the model-----------------')
                model.load_weights(checkpoint_save_path)
            # 使用参数save_weights_only时：设置True，则调用model.save_weights()；设置False，则调用model.save()；
            # monitor如果val_loss 提高了就会保存，没有提高就不会保存
            # save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_weights_only=True,
                                                             save_best_only=True, monitor='val_loss')
            # batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，
            # 使目标函数优化一步。epochs：整数，训练的轮数，每个epoch会把训练集轮一遍。
            # callbacks：list，其中的元素是keras.callbacks.Callback的对象。
            # 这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数
            # validation_data：形式为（X，y）的tuple，是指定的验证集，此参数将覆盖validation_spilt
            history = model.fit(x_train, y_train, batch_size=32, epochs=4, validation_data=(x_test, y_test),
                                validation_freq=1, callbacks=[cp_callback])
            model.summary()
            l = "p"
            if l:
                get_para(model)  # 参数提取
                plt_curve(history, plt)  # 绘制曲线
            # 测试集输入模型进行预测
            predicted_stock_price = model.predict(x_test)
            #  evaluate 绘图和评估
            l = "x"
            if l:
                predicted_stock_price, real_stock_price = \
                    predict_curve(predicted_stock_price, sc, test_date, test_set, pd, plt)
                evaluate_error(predicted_stock_price, real_stock_price)
            # 构建数据，预测未来 last为test集的最后60个收盘价
            predicted_price = predict_future(last, np, model, sc)
            # 画未来预测线
            predict_future_curve(predicted_price, plt)

    # lstm算法
    def stock_robot_lstm(p):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dropout, Dense, LSTM
        import tensorflow as tf

        df = pd.read_csv(p)
        # print(df.head())
        # creating dataframe 本数据已经从早到晚排列
        # data = df.sort_index(ascending=True, axis=0)
        # print(data.shape)
        # print(data[0:5])
        # setting index as date
        # print(df.date)
        # 把日期做index去标记x轴
        df['date'] = pd.to_datetime(df.date, format='%Y-%m-%d')
        # print(df)
        df.index = df['date']
        # print(df['close'])
        # plot
        # plt.figure(figsize=(30, 9))
        # plt.plot(df['close'], label='close Price history')
        # plt.show()
        new_data = pd.DataFrame(index=range(0, len(df)), columns=['date', 'close'])
        # print(new_data.shape)
        # print(new_data[0:5])
        for i in range(0, len(df)):
            new_data['date'][i] = df['date'][i]
            new_data['close'][i] = df['close'][i]
        print("new_data", new_data.shape)
        # print(new_data[0:5])
        # print(new_data.index)

        # setting index为日期
        new_data.index = new_data.date
        # print(new_data.index)
        # 查看数据按列的统计信息可显示数据的数量、std：标准差,缺失值、最小最大数、平均值、分位数等信息
        # print(df.describe())
        # print(new_data.describe())
        # inplace：是否作用于原来的df。axis 默认为0，指删除行，因此删除columns时要指定axis=1；
        # inplace = False，默认该删除操作不改变原数据，而是返回一个执行删除操作后的新dataframe；
        # inplace = True，则会直接在原数据上进行删除操作，删除后无法返回
        new_data.drop('date', axis=1, inplace=True)
        # print(new_data.shape)
        # print(new_data[0:5])
        # creating train and test sets
        dataset = new_data.values
        train = dataset[0:700, :]
        valid = dataset[700:, :]
        # print(dataset[0:5])
        # print(train[0:5])
        # print(valid[125:])

        # converting dataset into x_train and y_train
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        # print(scaled_data[0:5])

        x_train, y_train = [], []
        for i in range(60, len(train)):
            x_train.append(scaled_data[i - 60:i, 0])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        print(x_train.shape)
        dd = ""
        if dd:
            # create and fit the LSTM network
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            # model.add(Dropout(0.1))
            model.add(LSTM(units=50))
            # model.add(Dropout(0.1))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

            # predicting 246 values, using past 60 from the train data
            inputs = new_data[len(new_data) - len(valid) - 60:].values
            # print(len(new_data) - len(valid) - 60)
            # print(inputs[0:4])
            # 如果等于 - 1的话，那么Numpy会根据剩下的维度计算出数组的另外一个newshape 属性值。
            # z.reshape(-1, 1)是说，我们不知道新z的行数是多少，但是想让z变成只有一列，行数不知的新数组，
            # 通过z.reshape(-1, 1)，Numpy 自动计算出有16行，新的数组shape属性为(16, 1)，与原来的(4, 4)配套。
            inputs = inputs.reshape(-1, 1)
            # print(inputs[0:4])
            # print(inputs[len(inputs)-4:])
            inputs = scaler.transform(inputs)
            print(inputs[0:4])
            print(inputs.shape)

            X_test = []
            for i in range(60, inputs.shape[0]):
                X_test.append(inputs[i - 60:i, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            closing_price = model.predict(X_test)
            print(closing_price[0:5])
            closing_price = scaler.inverse_transform(closing_price)
            print(closing_price[0:5])
            # power(x, y)函数，计算x的y次方。print(np.power([2,3], [3,4]))分别求 2, 3的 3， 4 次方。[ 8 81]：
            rms = np.sqrt(np.mean(np.power((valid - closing_price), 2)))
            print(rms)
            # 11.772259608962642
            valid = new_data[700:].copy()
            valid['predictions'] = closing_price
            print(valid[0:5])
            plt.figure(figsize=(16, 9))
            # plt.plot(train['close'])
            plt.plot(valid[['close', 'predictions']])
            plt.show()


    def stock_robot_simple_rnn(p):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from tensorflow import keras
        import tensorflow as tf
        from tensorflow.keras.layers import Dropout, Dense, SimpleRNN
        import os
        from sklearn.preprocessing import MinMaxScaler

        test_date, test_set, training_set = read_data(p, pd)  # 读取数据21-07-30 = 3398
        print(test_date)
        print(test_set[-5:])
        # 归一化
        sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
        # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
        training_set_scaled = sc.fit_transform(training_set)
        # print(training_set_scaled[:10])
        test_set = sc.transform(test_set)  # 利用训练集的属性对测试集进行归一化
        # print(test_set[:10])
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        # 测试集：csv表格中前2426-300=2126天数据利用for循环，遍历整个训练集，
        # 提取训练集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签，for循环共构建2426-300-60=2066组数据。
        # [i - 60:i, 0]= 0到59共60个。[i, 0]=》 60=i=第61个数
        # d = 538
        d = 0
        for i in range(60, len(training_set_scaled) - d):
            x_train.append(training_set_scaled[i - 60:i, 0])
            y_train.append(training_set_scaled[i, 0])
        # print(x_train)
        # print(y_train)
        # 对训练集进行打乱
        np.random.seed(7)
        np.random.shuffle(x_train)
        np.random.seed(7)
        np.random.shuffle(y_train)
        # print(x_train)
        # print(y_train)
        # tf.random.set_seed: 设置全局随机种子。上面写法是tensorflow2 .0 的写法，如果是tensorflow1 .0
        # 则为：set_random_seed（）
        tf.random.set_seed(7)
        # 将训练集由list格式变为array格式
        x_train, y_train = np.array(x_train), np.array(y_train)
        # print(x_train)
        # print(y_train)

        # 使x_train符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。
        # 此处整个数据集送入，送入样本数为x_train.shape[0]即2066组数据；输入60个开盘价，预测出第61天的开盘价，
        # 循环核时间展开步数为60; 每个时间步送入的特征是某一天的开盘价，只有1个数据，故每个时间步输入特征个数为1
        # 送入样本数为x_train.shape[0] 即60个时间步长，每个步长都具有1个特征。因此，我希望有60个simpleRNN单元。
        # print(x_train.shape)
        x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))
        # print(x_train.shape)
        # print(x_train)
        # 利用for循环，遍历整个测试集，提取测试集中连续60天的开盘价作为输入特征x_train，第61天的数据作为标签，
        for i in range(60, len(test_set)):
            x_test.append(test_set[i - 60:i, 0])
            y_test.append(test_set[i, 0])
        # 测试集变array并reshape为符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))
        dd = "x"
        if dd:
            model = tf.keras.Sequential([
                SimpleRNN(80, return_sequences=True),
                Dropout(0.15),
                SimpleRNN(100),
                Dropout(0.15),
                Dense(1)
            ])
            # optimizer='adam'优化算法是梯度下降法,学习率为0.001 # 损失函数用均方误差. matrics=['acc']
            # 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值
            model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')
            checkpoint_save_path = r"D:\my2\my_test\rnn_stock.ckpt"
            if os.path.exists(checkpoint_save_path + '.index'):
                print('-------------load the model-----------------')
                model.load_weights(checkpoint_save_path)
            # 使用参数save_weights_only时：设置True，则调用model.save_weights()；设置False，则调用model.save()；
            # monitor如果val_loss 提高了就会保存，没有提高就不会保存
            # save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_weights_only=True,
                                                             save_best_only=True, monitor='val_loss')
            # batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，
            # 使目标函数优化一步。epochs：整数，训练的轮数，每个epoch会把训练集轮一遍。
            # callbacks：list，其中的元素是keras.callbacks.Callback的对象。
            # 这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数
            # validation_data：形式为（X，y）的tuple，是指定的验证集，此参数将覆盖validation_spilt
            history = model.fit(x_train, y_train, batch_size=62, epochs=4, validation_data=(x_test, y_test),
                                validation_freq=1, callbacks=[cp_callback])
            model.summary()
            get_para(model)  # 参数提取
            l = "c"
            if l:
                plt_curve(history, plt)  # 绘制曲线
            # 测试集输入模型进行预测
            predicted_stock_price = model.predict(x_test)
            predicted_stock_price, real_stock_price = predict_curve(predicted_stock_price, sc, test_date, test_set, pd, plt)
            ##########evaluate##############
            evaluate_error(predicted_stock_price, real_stock_price)

    # 读取数据
    def read_data(p, pd):
        df = pd.read_csv(p)  # 读取股票文件
        data_long = df.shape[0] - 50  # 数据行数df.shape[0] - 预测天数20 = 训练天数. 730-100=630
        test_date = df.iloc[data_long:, 0:1]  # [630: 729]=100天  0308(731-632)3524
        # print("124p")
        # print(test_date)
        test_date['date'] = pd.to_datetime(test_date.date, format='%Y-%m-%d')
        # print(dd.info())
        # 0305 3463,表格从0开始计数，2:3 是提取[2:3)列，前闭后开,故提取出C列开盘价
        training_set = df.iloc[0:data_long, 2:3].values
        test_set = df.iloc[data_long - 60:, 2:3].values  # 后20天的开盘价作为测试集 630-60=570 =>[570:729,2:3]=160
        return test_date, test_set, training_set


    # 参数提取
    def get_para(model):
        file = open(r"D:\my2\my_test\weights.txt", 'w')
        for v in model.trainable_variables:
            file.write(str(v.name) + '\n')
            file.write(str(v.shape) + '\n')
            file.write(str(v.numpy()) + '\n')
        file.close()


    #  评估误差
    def evaluate_error(predicted_stock_price, real_stock_price):
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import math
        # calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
        mse = mean_squared_error(predicted_stock_price, real_stock_price)
        # calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
        rmse = math.sqrt(mean_squared_error(predicted_stock_price, real_stock_price))
        # calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
        mae = mean_absolute_error(predicted_stock_price, real_stock_price)
        print('均方误差: %.6f' % mse)
        print('均方根误差: %.6f' % rmse)
        print('平均绝对误差: %.6f' % mae)


    # 测试集输入模型进行画预测线
    def predict_curve(predicted_stock_price, sc, test_date, test_set, pd, plt):
        # 对预测数据还原---从（0，1）反归一化到原始范围
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        # print(len(predicted_stock_price))
        # print(predicted_stock_price[:5])
        # 对真实数据还原---从（0，1）反归一化到原始范围
        real_stock_price = sc.inverse_transform(test_set[60:])
        # print("uuuuuuu")
        # print(real_stock_price[:5])
        predicted_stock_price = pd.DataFrame(predicted_stock_price)
        real_stock_price = pd.DataFrame(real_stock_price)
        # 画出真实数据和预测数据的对比曲线
        predicted_stock_price.index = test_date['date']
        real_stock_price.index = test_date['date']
        # print(real_stock_price.info())
        plt.figure(figsize=(16, 9))
        plt.plot(real_stock_price, color='red', label='Stock Price')
        plt.plot(predicted_stock_price, color='blue', label='Predicted')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.savefig(r"D:\my2\my_test\simpleRNN.png")
        plt.show()
        return predicted_stock_price, real_stock_price


    # 绘制曲线
    def plt_curve(history, plt):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(r"D:\my2\my_test\simpleRNN_loss.png")
        plt.show()

    # 预测未来
    def predict_future(last, np, model, sc):
        # print("jjjjjj")
        # print(last)
        x_data = np.array([last])
        # print(x_data)
        x_data = np.reshape(x_data, (x_data.shape[0], 60, 1))
        # print(x_data[0][-5:])
        # print(x_data.shape)
        predicted_price = []
        for i in range(0, 5):
            predicted_stock_price = model.predict(x_data)
            # print(predicted_stock_price)
            n = np.insert(x_data[0], 60, predicted_stock_price[0], axis=0)
            # print(n[-5:])
            # print(n.shape)
            # print("uuuuuuu", i)
            # print(n[:5])
            nn = np.delete(n, 0, axis=0)
            # print(nn[:5])
            # print(nn.shape)
            x_data[0] = nn
            # print(x_data[0][-5:])
            # print(x_data.shape)

            predicted_stock_price = sc.inverse_transform(predicted_stock_price)
            # print(predicted_stock_price)
            predicted_price.append(predicted_stock_price[0][0])
            # print(predicted_stock_price)
        print(predicted_price)
        return predicted_price

    # 画预测未来图
    def predict_future_curve(predicted_price, plt):
        plt.figure(figsize=(16, 9))
        plt.plot(predicted_price, color='red', label='Stock Price')
        plt.legend()
        # plt.savefig(r"D:\my2\my_test\simpleRNN.png")
        plt.show()


    def stock_robot(p):
        from sklearn.preprocessing import MinMaxScaler
        # import keras
        # print(keras.__version__)
        # import tensorflow as tf
        # print(tf.__version__)
        df = pd.read_csv(p)
        print(df.shape)
        # df = df[::-1] 倒置
        # a = [1, 2, 3, 4, "a", "ad", 45, 56]
        # print(a[::-2])  # [1, 3, 'a', 45]
        # df = df.reset_index(drop=True)重置索引
        print(df.head())
        open_price = df.iloc[:, 2:3]
        print(open_price.head())
        train_set = open_price[:600].values
        test_set = open_price[600:].values
        print(test_set.shape)
        print("Train size: ", train_set.shape)
        dates = pd.to_datetime(df['date'])
        # print(dates)
        # plt.plot_date(dates, open_price, fmt='-')
        # plt.show()
        # plt.savefig(r"D:\my2\my_test\test1final.png")
        # MinMaxScalar用于缩放0到1范围内的每个值的。这是非常重要的一步，因为当特征处于相对相似的缩放比例时，神经网络和其他算法的收敛速度更快
        sc = MinMaxScaler()
        train_set_scaled = sc.fit_transform(train_set)
        print(train_set_scaled[:3])
        print(train_set_scaled.shape)
        x_train = []
        y_train = []
        # print(a[..., 1])  # 第2列元素
        for i in range(60, 100):
            x_train.append(train_set_scaled[i - 60:i, 0])
            y_train.append(train_set_scaled[i, 0])
        print(x_train[:3])
        print(y_train[:3])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        # print(x_train)
        print(x_train.shape[0])
        print(x_train.shape[1])
        # shape和reshape() 函数都是对于数组(array)进行操作的，对于list结构是不可以的  如(2,3)表示2行3列
        #(5,60,1) 5维60行X1列的数组。
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        print(x_train[:2])
        print(x_train.shape)
        # from tensorflow.python.client import device_lib
        # print(device_lib.list_local_devices())
        dd = "vv"
        if dd:
            reg = keras.Sequential()
            # units: 正整数，也叫隐藏层，表示的是每个lstm单元里面前馈神经网络的输出维度，每一个门的计算都有一个前馈网络层
            # return_sequences: 布尔值。是返回输出序列中的最后一个输出，还是全部序列，True的话返回全部序列，False返回最后一个输出，默认为False
            reg.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            # print(tf.test.is_gpu_available())
            # rate: 在 0 和 1 之间浮动。需要丢弃的输入比例。
            reg.add(keras.layers.Dropout(0.2))
            reg.add(keras.layers.LSTM(units=50, return_sequences=True))
            reg.add(keras.layers.Dropout(0.2))
            reg.add(keras.layers.LSTM(units=50, return_sequences=True))
            reg.add(keras.layers.Dropout(0.2))
            reg.add(keras.layers.LSTM(units=50))
            reg.add(keras.layers.Dropout(0.2))
            # units: 正整数，输出空间维度
            reg.add(keras.layers.Dense(units=1))
            # 意为均方误差，也称标准差，缩写为MSE，可以反映一个数据集的离散程度。
            # optimizer='adam'优化算法是梯度下降法
            reg.compile(optimizer='adam', loss='mean_squared_error')
            # batch_size: 整数或None。每次提度更新的样本数。如果未指定，默认为 32.
            # verbose: 0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。
            reg.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)
        print(len(open_price), len(test_set))
        t = open_price[len(open_price) - len(test_set) - 60:].values
        print(t[:4])
        print(t.shape)
        t = sc.transform(t)
        print(t[:4])
        x_test = []
        for i in range(60, 95):
            x_test.append(t[i - 60:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        print(x_test[:2])
        print(x_test.shape)
        # 为输入样本生成输出预测。计算逐批次进行。x: 输入数据，Numpy数组（或者如果模型有多个输入，则为 Numpy数组列表）。
        # batch_size: 整数。如未指定，默认为32。verbose: 日志显示模式，0或1。steps: 声明预测结束之前的总步数（批次样本）
        # 默认值 None。
        pre = reg.predict(x_test)
        pre = sc.inverse_transform(pre)
        # plt.plot(test_set, color='green')
        # plt.savefig(r"D:\my2\my_test\tes.png")
        # plt.plot(pre, color='red')
        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        # plt.title('股票')
        # plt.savefig(r"D:\my2\my_test\te.png")
        # plt.show()
        # data = [[-1, 2], [-4, 6], [0, 10], [1, 18]]
        # data = [[2], [6], [10], [18]]
        # sc.fit(data)
        # print(sc.fit_transform(data))
        # print(sc.data_max_)
        # print(sc.data_min_)
        # a = np.arange(20).reshape(4, 5)
        # ms = MinMaxScaler(feature_range=(-1, 1))
        # b = ms.fit_transform(a)
        # print('a:\n', a)
        # print('b:\n', b)
        # print(ms.data_max_)
        # print(ms.data_min_)

    def deep_learn(p):
        import pandas as pd
        # from utils import *
        # import time
        import numpy as np
        # from mxnet import nd, autograd, gluon
        # from mxnet.gluon import nn, rnn
        # import mxnet as mx
        import datetime
        # import seaborn as sns
        import matplotlib.pyplot as plt
        # from sklearn.decomposition import PCA
        # import math
        # from sklearn.preprocessing import MinMaxScaler
        # from sklearn.metrics import mean_squared_error
        # from sklearn.preprocessing import StandardScaler
        # import xgboost as xgb
        # from sklearn.metrics import accuracy_score
        # import warnings
        # warnings.filterwarnings("ignore")
        # context = mx.cpu();
        # model_ctx = mx.cpu()
        # mx.random.seed(1719)

        def parser(x):
            return datetime.datetime.strptime(x, '%Y-%m-%d')
        # date_parser: function, default None   parse_dates=[0]解析第一列？
        # 用于解析日期的函数，默认使用dateutil.parser.parser来做转换。Pandas尝试使用三种不同的方式解析，
        # 指定哪一行作为表头。默认设置为0（即第一行作为表头），如果没有表头的话，要修改参数，设置header=None
        dataset_ex_df = pd.read_csv(p, header=0, parse_dates=[0], date_parser=parser)
        l = ""
        if l:
            print(dataset_ex_df.head())
            print(dataset_ex_df[['date', 'close']].head(3))
            print('There are {} number of days in the dataset.'.format(dataset_ex_df.shape[0]))
            num_training_days = int(dataset_ex_df.shape[0] * .75)
            print('training: {}. test: {}.'.format(num_training_days, dataset_ex_df.shape[0] - num_training_days))
        l = ""
        if l:
            # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为801英寸等于2 .5cm, A4纸是21 * 30cm的纸张
            # frameon=True: 是否显示边框
            plt.figure(figsize=(14, 5), dpi=100, frameon=True)
            plt.plot(dataset_ex_df['date'], dataset_ex_df['close'], label='Goldman Sachs stock')
            # vlines(x=1, ymin=0.1, ymax=2,
            plt.vlines(datetime.date(2021, 4, 20), 3000, 3600, linestyles='--', colors='gray', label='Train/Test data cut-off')
            plt.xlabel('Date')
            plt.ylabel('USD')
            plt.title('Figure 2: Goldman Sachs stock price')
            plt.legend()
            plt.show()

        # 创建技术指标。
        def get_technical_indicators(dataset):
            da = dataset['close'].copy()
            dataset = dataset.copy()

            # Create 7 and 21 days Moving Average    .loc[row_indexer,col_indexer]
            dataset['ma7'] = da.rolling(window=7).mean()
            dataset['ma21'] = da.rolling(window=21).mean()
            # Create MACD
            dataset['26ema'] = pd.DataFrame.ewm(da, span=26).mean()
            dataset['12ema'] = pd.DataFrame.ewm(da, span=12).mean()
            dataset['MACD'] = (dataset['12ema'] - dataset['26ema'])
            # Create Bollinger Bands
            # dataset['20sd'] = pd.stats.moments.rolling_std(da, 20)
            # dataset['upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
            # dataset['lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)

            # # Create Exponential moving average
            dataset['ema'] = da.ewm(com=0.5).mean()
            # Create Momentum
            dataset['momentum'] = da - 1
            print(dataset)

            return dataset

        def plot_technical_indicators(dataset, last_days):
            plt.figure(figsize=(16, 10), dpi=100)
            shape_0 = dataset.shape[0]
            xmacd_ = shape_0 - last_days

            dataset = dataset.iloc[-last_days:, :]
            x_ = range(3, dataset.shape[0])
            x_ = list(dataset.index)

            # Plot first subplot  subplot(nrows, ncols, index, **kwargs)
            plt.subplot(2, 1, 1)
            plt.plot(dataset['ma7'], label='MA 7', color='g', linestyle='--')
            plt.plot(dataset['close'], label='Closing Price', color='b')
            plt.plot(dataset['ma21'], label='MA 21', color='r', linestyle='--')
            plt.plot(dataset['ema'], label='ema', color='grey', linestyle='dotted')
            # plt.plot(dataset['upper_band'], label='Upper Band', color='c')
            # plt.plot(dataset['lower_band'], label='Lower Band', color='c')
            # plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
            plt.title('Technical indicators for Goldman Sachs - last {} days.'.format(last_days))
            plt.ylabel('USD')
            plt.legend()
            # Plot second subplot
            plt.subplot(2, 1, 2)
            plt.title('MACD')
            plt.plot(dataset['MACD'], label='MACD', linestyle='-.')
            plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
            plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
            # plt.plot(dataset['momentum'], label='Momentum', color='b', linestyle='-')
            plt.legend()
            plt.show()

        l = ""
        if l:
            dataset_TI_df = get_technical_indicators(dataset_ex_df[['close']])
            # dataset_TI_df.head()
            plot_technical_indicators(dataset_TI_df, 100)
        l = "v"
        if l:
            data_FT = dataset_ex_df[['date', 'close']]
            # print(np.asarray(data_FT['close'].tolist()))
            close_fft = np.fft.fft(np.asarray(data_FT['close'].tolist()))
            fft_df = pd.DataFrame({'fft': close_fft})
            # abs()函数返回数字的绝对值。np.angle计算复数的辐角主值。
            fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
            fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
            l = ""
            if l:
                plt.figure(figsize=(14, 7), dpi=100)
                # 区别在于np.array（默认情况下）将会copy该对象，而np.asarray除非必要，否则不会copy该对象。
                print(fft_df)
                fft_list = np.asarray(fft_df['fft'].tolist())
                # print(fft_list)
                for num_ in [3, 6, 9, 100]:
                    print("fft_list---------------------", num_)
                    fft_list_m10 = np.copy(fft_list)
                    fft_list_m10[num_:-num_] = 0
                    print(fft_list_m10[0:12])
                    # plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
                # plt.plot(data_FT['close'], label='Real')
                # plt.xlabel('Days')
                # plt.ylabel('USD')
                # plt.title('Figure 3: Goldman Sachs (close) stock prices & Fourier transforms')
                # plt.legend()
                # plt.show()
            l = ""
            if l:
                from collections import deque
                items = deque(np.asarray(fft_df['absolute'].tolist()))
                # floor返回不大于输入参数的最大整数。（向下取整）
                items.rotate(int(np.floor(len(fft_df) / 2)))
                plt.figure(figsize=(10, 7), dpi=80)
                plt.stem(items)
                plt.title('Figure 4: Components of Fourier transforms')
                plt.show()
        l = "v"
        if l:
            # from statsmodels.tsa.arima_model import ARIMA
            from statsmodels.tsa.arima.model import ARIMA
            # from pandas import DataFrame
            series = data_FT['close']
            l = ""
            if l:
                model = ARIMA(series, order=(5, 1, 0))
                # disp：True会打印中间过程，我们直接设置False即可
                model_fit = model.fit()
                print(model_fit.summary())
                # from pandas.tools.plotting import autocorrelation_plot
                from pandas.plotting import autocorrelation_plot
                autocorrelation_plot(series)
                plt.figure(figsize=(10, 7), dpi=80)
                plt.show()

            # from pandas import read_csv
            # from pandas import datetime
            from sklearn.metrics import mean_squared_error
            X = series[550:730].values
            size = int(len(X) * 0.66)
            train, test = X[0:size], X[size:len(X)]
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model = ARIMA(history, order=(5, 1, 0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
            error = mean_squared_error(test, predictions)
            # MSE(均方误差)
            print('Test MSE: %.3f' % error)
            # Plot the predicted (from ARIMA) and real prices
            plt.figure(figsize=(12, 6), dpi=100)
            plt.plot(test, label='Real')
            plt.plot(predictions, color='red', label='Predicted')
            plt.xlabel('Days')
            plt.ylabel('USD')
            plt.title('Figure 5: ARIMA model on GS stock')
            plt.legend()
            plt.show()

    # stock_robot_test(r'D:\my2\my_test\history_A_stock_k_data.csv')
    # stock_robot_lstm(r'D:\my2\my_test\history_A_stock_k_data.csv')
    # stock_robot_simple_rnn(r'D:\my2\my_test\history_A_stock_k_data.csv')
    # test()
    # deep_learn(r'D:\my2\my_test\history_A_stock_k_data.csv')

