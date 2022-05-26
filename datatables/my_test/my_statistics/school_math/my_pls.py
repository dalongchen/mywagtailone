import pandas as pd
# import numpy as np
import sqlite3
your_db_path = r"D:\myzq\axzq\T0002\stock_load\thesis\数学建模\my\school_math.db"  # 数据库路径


def pls_regression(db_path):
    # from sklearn.cross_decomposition import PLSRegression
    # 导入train_test_split用于拆分训练集和测试集
    # from sklearn.model_selection import train_test_split
    with sqlite3.connect(db_path) as conn:
        tab = 'all_integration_other_volume_price'
        select_table = """select * from {}""".format(tab)
        # select_table = """select * from {} where 预测涨幅>0.01 order by 时间 asc""".format(tab)
        df = pd.read_sql(select_table, conn)
        # df = df.iloc[:, 2:-288]  # 去 序号,时间,
        df = df.iloc[:, 2:]  # 去 序号,时间,
        print(df)
        # 写入excel文件
        df.to_excel(r"D:\myzq\axzq\T0002\stock_load\thesis\数学建模\my\school3.xlsx")
        return ''
        df_open = df['开盘价']

    # df = load_data(db_path, f='df', tab='all_integration_other_volume_price')
    # # x_train, x_test, y_train, y_test = load_data(db_path, f='df', tab='all_integration_other_volume_price')
    # df = df.iloc[:-48, 2:]
    # dfx = df.iloc[:, 2:]
    # del dfx['收盘价']
    # del dfx['最高价']
    # del dfx['最低价']
    # del dfx['成交额']
    # # print(dfx)
    # dfy = df['收盘价']
    # # print(dfy)
    # # dfy = df['成交量']
    # x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.1, random_state=7)
    # # print(x_train)
    # # print(y_train)
    # # print(x_train.shape)
    # # print(y_train.shape)
    # # x_train2, y_train2 = np.array(x_train), np.array(y_train)
    # pls = PLSRegression(n_components=80)
    # # Fit data：拟合数据
    # pls.fit(x_train, y_train)
    # # Ypredict = pls.predict(x_train)
    # # print("Ypredict", Ypredict)
    # # Ypredict = Ypredict.flatten()
    # # print("Ypredict", Ypredict)
    # # 真实值与预测值的确定系数，越接近于1越好
    # R2Y = pls.score(x_train, y_train)
    # print("R2Y", R2Y)
    # # return ''
    # # 变量重要性分析，变量对y的影响程度排序，一般认为大于1是有影响的
    # df_vip = pd.DataFrame()
    # df_vip['X'] = x_train.columns
    # # print(df_vip)
    # df_vip['vip'] = _calculate_vips(pls)
    # # print(df_vip)
    # # vip = df_vip.sort_values(by='vip', ascending=True).tail(200)
    # # print(vip)
    # # VIP的可视化
    # # plt.figure(figsize=(16, 16))
    # # plt.barh(vip.X, vip.vip, height=0.5)
    # # plt.title('VIP')
    # # plt.figure(figsize=(15, 8))
    # # length = range(len(y_train))
    # # plt.plot(length, y_train, marker='o', label='target_y')
    # # plt.plot(length, Ypredict, marker='o', label='target_y_predict')
    # # plt.legend()
    # # df_vip['coef'] = pls.coef_.flatten()
    # # df_vip = df_vip.sort_values(by='vip', ascending=False).iloc[150:200]
    # # df_vip = df_vip.sort_values(by='vip', ascending=False).iloc[100:150]
    # # df_vip = df_vip.sort_values(by='vip', ascending=False).iloc[50:100]
    # df_vip = df_vip.sort_values(by='vip', ascending=False)
    # # df_vip = df_vip.sort_values(by='vip', ascending=False).iloc[50:80]
    # # df_vip = df_vip.sort_values(by='vip', ascending=False).iloc[0:50]
    # # df_vip = df_vip.sort_values(by='vip', ascending=False).round(4).head(80)
    # print(df_vip)
    # with sqlite3.connect(db_path) as conn:
    #     df_vip..to_sql('数字经济收盘价影响分析', con=conn, if_exists='replace', index=False)

pls_regression(your_db_path)
