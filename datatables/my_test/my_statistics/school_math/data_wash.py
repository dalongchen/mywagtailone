import pandas as pd
import sqlite3
your_db_path = r"D:\myzq\axzq\T0002\stock_load\thesis\数学建模\school_math.db"  # 数据库路径


#  标准化后的表格整合
def integration_table(db_path, tab_name1, tab_name2, tab_name3=''):
    with sqlite3.connect(db_path) as conn:
        if tab_name1 == 'data_5_dimensionless':
            sql_select = """select * from {}""".format(tab_name1)
            df = pd.read_sql(sql_select, conn)
            # 获取df对象标题列表
            cols = list(df.keys())
            # print(cols)
            # 调位置
            cols.insert(2, cols.pop(cols.index('成交量')))
            # print(cols)
            # 重组df对象排列顺序
            df = df.loc[:, cols]
            """如果不存在就创建表append"""
            df.to_sql(tab_name2, con=conn, if_exists='replace', index=False)
            # print(df)
        elif tab_name1 == 'domestic_indicator_dimensionless':
            sql_select = """select 时间 from {}""".format(tab_name2)
            df = pd.read_sql(sql_select, conn)
            df = df['时间'].str[0:10]
            # print(df)
            # df1 = df.iloc[:, 0:1]  # 访问第0-1列
            # df2 = df.iloc[:, 1:]  # 访问第36-69列对数据标准化
            # print(df.keys())
        elif tab_name1 == 'all_integration_other':
            ww = ''
            if ww:
                sql_select2 = """select * from {}""".format(tab_name2)
                df2 = pd.read_sql(sql_select2, conn)
                # del df2['序号']
                # print("ww", df2)
                sql_select3 = """select * from {}""".format(tab_name3)
                df3 = pd.read_sql(sql_select3, conn)
                # df1 = df.iloc[:, 0:1]  # 访问第0-1列
                df3 = df3.iloc[:, 1:]  # 访问第列对数据标准化
                # print("ww3", df3)
                # 把标准化数据更新回原表
                df1 = pd.concat([df2, df3], axis=1)
                # print(df1)
                # print(df.keys())
                """如果不存在就创建表append"""
                df1.to_sql(tab_name1, con=conn, if_exists='replace', index=False)
            sql_day = """select * from {}""".format(tab_name1)
            # sql_day = """select 序号,时间 from {}""".format(tab_name1)
            df_day = pd.read_sql(sql_day, conn)
            print(df_day.keys())
            # df_da = df_day['时间'].str[0:10]
            # sql_day2 = r"""select * from {}""".format(tab_name3)
            sql_day2 = r"""select * from {} where 时间>='{}'""".format(tab_name3, '2021-07-14 00:00:00')
            df_day2 = pd.read_sql(sql_day2, conn)
            # df_day2 = df_day2.iloc[:, 1:]  #  technical_indicator_dimensionless需要
            # print(df_day2.values)
            print(df_day2.keys())
            # print()
            # print(df_day.iloc[:, 2:].values)
            all_list = []
            for i in df_day.values:
                dd = i[2][0:10] + ' 00:00:00'
                for ii in df_day2.values:
                    # print(dd == ii[0])
                    # print(ii[1])
                    if dd == ii[0]:
                        if ww:
                            aa = list(i)[:9] + list(ii)[1:]
                        else:
                            aa = list(i) + list(ii)[1:]
                        all_list.append(aa)
                        # print(aa)
            # print(all_list)
            # 参数解释：data是要传入的数据，index是索引（不指定会自动产生自增长的索引），
            if ww:
                col2 = df_day.keys()
            else:
                col2 = list(df_day.keys()) + list(df_day2.iloc[:, 1:].keys())
            print(col2)
            dfr = pd.DataFrame(data=all_list, columns=col2)
            print(dfr)  # [5893 rows x 52 columns]
            # tab_ = 'all_integration_other'
            # dfr.to_sql(tab_, con=conn, if_exists='replace', index=False)
        else:
            print("表名有误")

        # df.to_sql(tab_name2, con=conn, if_exists='replace', index=False)
# table_name1 = "all_integration"
table_name1 = "all_integration_other"

# table_name1 = "domestic_indicator_dimensionless"
# table_name1 = "data_5_dimensionless"
table_name2 = "integration"
table_name3 = "exchange_rate_dimensionless"
# table_name3 = "other_dimensionless"
# table_name3 = "international_indicator_dimensionless"
# table_name3 = "technical_indicator_dimensionless"
# table_name3 = "domestic_indicator_dimensionless"
# integration_table(your_db_path, table_name1, table_name2, tab_name3=table_name3)


#
def lagrange(df):
    # import pandas as pd
    import math as ma
    # 导入函数
    # from scipy.interpolate import lagrange
    # import matplotlib.pyplot as plt
    # 读取文本中的数据，数据有四列，MD，K，和Por这里以Por的异常值为例进行插值处理
    # df = pd.read_table('d:/data1.txt')
    # Por列中<0.05的为异常值，处理为空值
    # df['Por'][df['Por'] < 0.05] = None
    # plt.show()
    # print(df.shape)
    # print(df.shape[1])
    # n为插值考虑的范围，为前后5个数
    n = 1
    df1 = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    # print(df1)
    for ii in range(df.shape[1]):  # 循环行
        # ii = int(ii)
        # print(ii)
        # 搜索第2列（Por列）的异常值并且插值
        for i in range(df.shape[0]):  # 循环行
            # pass
            # print(df.iat[i, ii])
            # ma.isnan()判断值是否为空值，#iat类似于iloc使用隐式索引访问某个元素
            if ma.isnan(df.iat[i, ii]):
                # print("ddd", df.iat[i, 2])
                # 插值算法考虑的数据interdf是一个Series从空值算起前后n个,取3个
                interdf = df.iloc[i - n:i + n + 1, ii]
                if len(interdf) != 3:
                    print("interdf", len(interdf))
                    print("interdf", i, ii)
                    # print("inter", interdf)
                # 删除掉里面的空值
                interdf = interdf[interdf.notnull()]  # 还有10个
                # print("interdf", interdf.mean())
                # print("interdf", interdf.values)
                # x为数据的索引，也可以为其他列的值
                # list_x = list(interdf)
                # print(list_x)
                # # y为Series里的值，x,y都转换成列表
                # list_y = list(interdf.values)
                # # f为利用拉格朗日法建立的函数关系y=f(x)
                # f = lagrange(list_x, list_y)
                # print("f", f(i))
                # 把插值结果加入到df中
                # df.iat[i, 2] = f(i)
                df.iat[i, ii] = interdf.mean()
    df = pd.concat([df1, df], axis=1)
    # print(df)
    return df


# pandas自动建表并读入数据.db_path：数据库路径, path：文件路径, table_name表名
def pandas_create_table(db_path, path="", table_name="", df="", sheet_name=''):
    """使用字典填充：第1列缺失值用11代替，第2列缺失值用22代替print(data.fillna({1:11,2:22}))"""
    if path:
        df = pd.read_excel(path, engine="openpyxl", sheet_name=[sheet_name])
        # df = pd.DataFrame(df)
        # order_dict = pd.read_excel(r'C:\Users\sss\Desktop\test.xlsx', header=0, usecols=[2, 3]
        # names = ["Name", "Number"], sheet_name = ["Sheet1", "Sheet2"], skiprows = range(1, 10), skipfooter = 4)
        # for sheet_name, df in df.items():
        #     print(sheet_name)
        df = df[sheet_name]
        # df = df[:246]  #  其他板块信息
        # df = df.drop(labels=range(0, 2), axis=0)  # axis=0 表示按行删除，删除索引值是第1行至第3行的正行数据
        # del df["指数代码"]
        # del df["ARBR"]
        # df = df.dropna()
        # print(df.isnull())
        # df = df.fillna({'道琼斯工业平均指数': df['道琼斯工业平均指数'].mean(), '纳斯达克综合指数': df['纳斯达克综合指数'].mean()})  # 　第一列缺失值用该列均值代替，同理第2列
        # df = lagrange(df)
        # print()

    print(df)

    # data = df.values
    # print("获取到所有的值:\n{}".format(df.items()))
    # print("获取到所有的值:\n{}".format())
    # data_frame = pd.DataFrame(df)
    with sqlite3.connect(db_path) as conn:
        """如果不存在就创建表"""
        df.to_sql(table_name, con=conn, if_exists='replace', index=False)
        # cur = conn.cursor()
        # aa = df.keys()
        # aa = tuple(df.keys())
        # sql_del = """delete from {} where 成交量=0.0""".format(table_name)  # 删除19空
        # cur.execute(sql_del)
        # cur.close()


# 转换excel数据入数据库
def my_statistics_reed_excel(db_path):
    path = r"D:\myzq\axzq\T0002\stock_load\thesis\数学建模\202205数学建模竞赛预赛题目\B题附件\附表.xlsx"
    table_name = "other"
    sheet_name = "其他板块信息"
    # table_name = "international_indicator"
    # sheet_name = "国际市场指标"
    # table_name = "domestic_indicator"
    # sheet_name = "国内市场指标"
    # table_name = "technical_indicator"
    # sheet_name = "技术指标"
    # table_name = "data_5"
    # sheet_name = "数字经济版块信息"
    pandas_create_table(db_path, path=path, table_name=table_name, sheet_name=sheet_name)
# my_statistics_reed_excel(your_db_path)


# 数据清洗 删除19空列。入数据表 drop_null_column
def drop_null_column(db_path):
    """read_sql() 函数既可以读取整张数据表，又可以执行SQL语句，其语法格式如下：
    pandas.read_sql(sql，con，index_col=None，coerce_float=True，parmes=None，parse_dates=None，columns=None，chunksize=None）
常用参数的含义如下：
（1）sql：表示被执行的SQL语句
（2）con：接收数据库连接，表示数据库的连接信息
（3）index_col：默认为None，如果传入一个列表，则表示为层次化索引
（4）coerce_float：将非法字符串、非数字对象的值转换为浮点数类型
（5）params：传递给执行方法的参数列表，如params={‘name’:‘values’}
（6）columns：接收list表示读取数据的列名，默认为None"""
    with sqlite3.connect(db_path) as conn:
        select_table = """select * from my_statistics"""
        df = pd.read_sql(select_table, conn)
        del df['NATION']  # 删除19个空列，改变原始数据
        del df['HEART_RATE']
        del df['GLU_2H']
        del df['GSP']
        del df['UPR_24']
        del df['BUN']
        del df['UCR']
        del df['CP']
        del df['INS']
        del df['ESR']
        del df['LP_A']
        del df['PL']
        del df['FIBRIN']
        del df['ALB_CR']
        del df['LPS']
        del df['CA199']
        del df['CRP']
        del df['M1_M2']
        del df['TH2']
        # df.drop('NATION', axis=1, inplace='True')  # 改变原始数据
        # print(df)
        # pandas自动建表并读入数据db_path：数据库路径, path：文件路径, table_name表名
        pandas_create_table(db_path, path="", table_name="drop_null_column", df=df)

# drop_null_column(your_db_path)


# 数据统计 平均值等.入数据表 statistics_medicine
def statistics_medicine(db_path):
    with sqlite3.connect(db_path) as conn:
        select_table = """select * from drop_null_column"""
        df = pd.read_sql(select_table, conn)
        sum1 = round(df.sum(axis=0), 3)  # axis 0为列，1为行
        mean1 = round(df.mean(axis=0), 3)
        var1 = round(df.var(axis=0), 3)
        std1 = round(df.std(axis=0), 3)
        df2 = pd.DataFrame(columns=['name', 'sum', 'mean', 'var', 'std'])
        df2["name"] = list(df.keys())
        df2["sum"] = list(sum1)
        df2["mean"] = list(mean1)
        df2["var"] = list(var1)
        df2["std"] = list(std1)
        # age_mean = round(df['AGE'].mean(), 2)  # 统计HEIGHT列的平均值
        # print(df2)
        # pandas自动建表并读入数据db_path：数据库路径, path：文件路径, table_name表名
        pandas_create_table(db_path, path="", table_name="statistics_medicine", df=df2)

# statistics_medicine(your_db_path)


# 数据清洗 用均值填充空值，入数据表 mean_fill_null
def mean_fill_null(db_path):
    with sqlite3.connect(db_path) as conn:
        select_drop_null_column = """select * from drop_null_column"""
        df = pd.read_sql(select_drop_null_column, conn)
        # 用均值填充空值
        df['AGE'] = df['AGE'].fillna(round(df['AGE'].mean(axis=0), 3))
        df['HEIGHT'] = df['HEIGHT'].fillna(round(df['HEIGHT'].mean(axis=0), 3))
        df['WEIGHT'] = df['WEIGHT'].fillna(round(df['WEIGHT'].mean(axis=0), 3))
        df['BP_HIGH'] = df['BP_HIGH'].fillna(round(df['BP_HIGH'].mean(axis=0), 3))
        df['BP_LOW'] = df['BP_LOW'].fillna(round(df['BP_LOW'].mean(axis=0), 3))
        df['BMI'] = df['BMI'].fillna(round(df['BMI'].mean(axis=0), 3))

        df['GLU'] = df['GLU'].fillna(round(df['GLU'].mean(axis=0), 3))
        df['HBA1C'] = df['HBA1C'].fillna(round(df['HBA1C'].mean(axis=0), 3))
        df['TG'] = df['TG'].fillna(round(df['TG'].mean(axis=0), 3))
        df['TC'] = df['TC'].fillna(round(df['TC'].mean(axis=0), 3))
        df['HDL_C'] = df['HDL_C'].fillna(round(df['HDL_C'].mean(axis=0), 3))

        df['LDL_C'] = df['LDL_C'].fillna(round(df['LDL_C'].mean(axis=0), 3))
        df['FBG'] = df['FBG'].fillna(round(df['FBG'].mean(axis=0), 3))
        df['BU'] = df['BU'].fillna(round(df['BU'].mean(axis=0), 3))
        df['SCR'] = df['SCR'].fillna(round(df['SCR'].mean(axis=0), 3))
        df['SUA'] = df['SUA'].fillna(round(df['SUA'].mean(axis=0), 3))

        df['HB'] = df['HB'].fillna(round(df['HB'].mean(axis=0), 3))
        df['PCV'] = df['PCV'].fillna(round(df['PCV'].mean(axis=0), 3))
        df['PLT'] = df['PLT'].fillna(round(df['PLT'].mean(axis=0), 3))
        df['TBILI'] = df['TBILI'].fillna(round(df['TBILI'].mean(axis=0), 3))
        df['DBILI'] = df['DBILI'].fillna(round(df['DBILI'].mean(axis=0), 3))

        df['TP'] = df['TP'].fillna(round(df['TP'].mean(axis=0), 3))
        df['ALB'] = df['ALB'].fillna(round(df['ALB'].mean(axis=0), 3))
        df['LDH_L'] = df['LDH_L'].fillna(round(df['LDH_L'].mean(axis=0), 3))
        df['ALT'] = df['ALT'].fillna(round(df['ALT'].mean(axis=0), 3))
        df['AST'] = df['AST'].fillna(round(df['AST'].mean(axis=0), 3))

        df['GGT'] = df['GGT'].fillna(round(df['GGT'].mean(axis=0), 3))
        df['ALP'] = df['ALP'].fillna(round(df['ALP'].mean(axis=0), 3))
        df['PT'] = df['PT'].fillna(round(df['PT'].mean(axis=0), 3))
        df['PTA'] = df['PTA'].fillna(round(df['PTA'].mean(axis=0), 3))
        df['APTT'] = df['APTT'].fillna(round(df['APTT'].mean(axis=0), 3))

        df['IBILI'] = df['IBILI'].fillna(round(df['IBILI'].mean(axis=0), 3))
        df['GLO'] = df['GLO'].fillna(round(df['GLO'].mean(axis=0), 3))
        # print(df['HEIGHT'])
        # pandas自动建表并读入数据.db_path：数据库路径, path：文件路径, table_name表名
        pandas_create_table(db_path, path="", table_name="mean_fill_null", df=df)

# mean_fill_null(your_db_path)


# 数据无量纲化,采用数据标准化方式，入数据表 dimensionless
def dimensionless(db_path, tab_name1, tab_name2):
    from sklearn.preprocessing import StandardScaler  # 数据标准化
    scale = StandardScaler()  # 实例化，数据标准化
    with sqlite3.connect(db_path) as conn:
        sql_select = """select * from {}""".format(tab_name1)
        df = pd.read_sql(sql_select, conn)
        df1 = df.iloc[:, 0:1]  # 访问第0-1列
        df2 = df.iloc[:, 1:]  # 访问第36-69列对数据标准化
        # print(df1)
        # print(df2.keys())
        # 第36-69列对数据标准化
        scale = scale.fit(df2)  # fit，本质是生成均值和方差
        # print("拟合后的均值为：", scale.mean_)
        # print("拟合后的方差：", scale.var_)
        result_ = scale.transform(df2)
        # print(result_)
        # print(result_.shape)
        # print(result_.mean())
        # print(result_.std())
        columns = df2.keys()
        num_df = pd.DataFrame(data=result_, columns=columns)
        # print(num_df)
        # 把标准化数据更新回原表
        df = pd.concat([df1, num_df], axis=1)
        # print(result)
        # pandas自动建表并读入数据.db_path：数据库路径, path：文件路径, table_name表名
        # pandas_create_table(db_path, path="", table_name=tab_name2, df=result)
        """如果不存在就创建表"""
        df.to_sql(tab_name2, con=conn, if_exists='replace', index=False)
# tab_nam1 = "other"
# tab_nam2 = "other_dimensionless"
# tab_nam1 = "exchange_rate"
# tab_nam2 = "exchange_rate_dimensionless"
# tab_nam1 = "international_indicator"
# tab_nam2 = "international_indicator_dimensionless"
# tab_nam1 = "domestic_indicator"
# tab_nam2 = "domestic_indicator_dimensionless"
# tab_nam1 = "technical_indicator"
# tab_nam2 = "technical_indicator_dimensionless"
# my_statistics_reed_excel(your_db_path)
# dimensionless(your_db_path, tab_nam1, tab_nam2)


#
def before_96(db_path, tab_name1, tab_name2):
    # import math as ma
    with sqlite3.connect(db_path) as conn:
        sql_select = """select * from {}""".format(tab_name1)
        df = pd.read_sql(sql_select, conn)
        df_volume = df['成交量']
        # print(df_volume.shape[0])
        # df1 = df.iloc[:, 0:1]  # 访问第0-1列
        # df2 = df.iloc[:, 1:]  # 访问第36-69列对数据标准化
        # Por列中<0.05的为异常值，处理为空值
        # df['Por'][df['Por'] < 0.05] = None
        # n为插值考虑的范围，为前后5个数
        n = 97
        # n = 49
        two_list = []
        # print(df_volume)
        # ll = list(df_volume.values)
        # print(ll)
        for ii in range(df_volume.shape[0]):  # 循环行
            aa = df_volume.iloc[ii + 1:ii + n]
            two_list.append(list(aa))
            # print(ii)
            # if ii > 5:
            #     break
            # print(aa)
            # if ii % n == 0:
            #     print(ii)
            #     print(df_volume.iloc[1:ii])
            #     two_list.append(df_volume.iloc[1:ii])
        # print(two_list[-5:])
        # print(two_list)
        # print("len", len(two_list))
        two_list_df = pd.DataFrame(data=two_list)
        # print(two_list_df)
        # 把标准化数据更新回原表
        df_all = pd.concat([df, two_list_df], axis=1)
        # print(df_all)
        """如果不存在就创建表"""
        # df_all.to_sql(tab_name2, con=conn, if_exists='replace', index=False)


# table_name1, table_name2 = 'all_integration_other', 'all_integration_other_volume'
# before_96(your_db_path, table_name1, table_name2)


def volume_price(db_path, tab_name1, tab_name2):
    # import math as ma
    with sqlite3.connect(db_path) as conn:
        sql_select = """select * from {}""".format(tab_name1)
        df = pd.read_sql(sql_select, conn)
        # df_volume = df['成交量']
        # df2 = df.iloc[:, 1:]  # 访问第36-69列对数据标准化
        df_volume_price = df.iloc[:, 3:9]  # 访问第0-1列
        # print(df_volume_price.shape)
        # print(df_volume_price.shape[0])
        # Por列中<0.05的为异常值，处理为空值
        # df['Por'][df['Por'] < 0.05] = None
        # n为插值考虑的范围，为前后5个数
        n = 49
        # n = 49
        two_list = []
        # print(df_volume_price.values[0:3])
        # ll = list(df_volume.values)
        # print(ll)
        # dd = [[1, 2], [3, 4], [5, 6]]
        # ss = []
        # for gg in dd:
        #     ss += [i for i in gg]
        # print(ss)
        for ii in range(df_volume_price.shape[0]):  # 循环行
            # aa = df_volume_price.values[0:3]
            aa = df_volume_price.values[ii + 1:ii + n]
            ss = []
            for gg in aa:  # 两维变1维
                ss += [i for i in gg]
            # print(ii)
            two_list.append(ss)
            # print(ss)
            # print(len(ss))
            # if ii > 0:
            #     break
            # if ii % n == 0:
            #     print(ii)
            #     print(df_volume.iloc[1:ii])
            #     two_list.append(df_volume.iloc[1:ii])
        print(two_list[0])
        print(two_list[1])
        print("len", len(two_list))
        two_list_df = pd.DataFrame(data=two_list)
        print(two_list_df)
        # 把标准化数据更新回原表
        df_all = pd.concat([df, two_list_df], axis=1)
        print(df_all)
        # """如果不存在就创建表"""
        # df_all.to_sql(tab_name2, con=conn, if_exists='replace', index=False)

#
# table_name1, table_name2 = 'all_integration_other', 'all_integration_other_volume_price'
# volume_price(your_db_path, table_name1, table_name2)


# 加载元数据 , f如果有输出总体
def load_data(db_path, f="", tab=''):
    with sqlite3.connect(db_path) as conn:
        select_table = """select * from {}""".format(tab)
        df = pd.read_sql(select_table, conn)
        # print(df)
        x_ = df.iloc[:, 4:]  # 访问第2-69列
        y_ = df["成交量"]
        # print(x_)
        # print(y_)
        if not f:
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
            x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.2, random_state=7)
            return x_train, x_test, y_train, y_test
        if f:
            return x_, y_

# load_data(your_db_path)


def pca_analysis(db_path):
    # 加载元数据 f="hh"输出总体
    x_train, y_train = load_data(db_path, f="hh", tab="all_integration_other_volume_price")
    end_price = x_train['收盘价']
    # print(end_price)
    del x_train['收盘价']
    del x_train['最高价']
    del x_train['最低价']
    del x_train['成交额']
    x_train = x_train[:-48]  # 去空行
    y_train = y_train[:-48]  # 去空行
    end_price = end_price[:-48]  # 去空行
    # print(x_train)
    # x_train, x_test, y_train, y_test = load_data(db_path)
    from sklearn.decomposition import PCA
    """
    class sklearn.decomposition.PCA(
        n_components=None,     # 指定保留的特征数量（降维以后的维数）
        copy=True,                    # 训练过程中是否影响原始样本数据，也影响到最后降维数据的返回方式
        whiten=False,                # 是否对降维后的数据提供白化处理。（白化就是每一维的特征做一个标准差归一化处理，除以一个标准偏差）
        svd_solver=’auto’,          #  SVD分解的计算方法
        tol=0.0,                         # 计算奇异值的公差
        iterated_power=’auto’,    # 采用svd_solver采用randomized方式的迭代次数。
        random_state=None)      # 同来创建随机数的实例，比如随机种

    n_components：代表返回的主成分的个数,也就是你想把数据降到几维
    n_components=2 代表返回前2个主成分
    0 < n_components < 1代表满足最低的主成分方差累计贡献率
    n_components=0.98，指返回满足主成分方差累计贡献率达到98%的主成分
    n_components=None，返回所有主成分
    n_components=‘mle’，将自动选取主成分个数n，使得满足所要求的方差百分比
    whiten: 判断是否进行白化。所谓白化，就是对降维后的数据的每个特征进行归一化，让方差都为1.对于PCA降维本身来说，
    一般不需要白化。如果你PCA降维后有后续的数据处理动作，可以考虑白化。默认值是False，即不进行白化。
    svd_solver：str类型，str {‘auto’, ‘full’, ‘arpack’, ‘randomized’}
    意义：定奇异值分解 SVD 的方法。
    svd_solver=auto：PCA 类自动选择下述三种算法权衡。
    svd_solver=‘full’:传统意义上的 SVD，使用了 scipy 库对应的实现。
    svd_solver=‘arpack’:直接使用 scipy 库的 sparse SVD 实现，和 randomized 的适用场景类似。
    svd_solver=‘randomized’:适用于数据量大，数据维度多同时主成分数目比例又较低的 PCA 降维。

    copy : bool (default True)，False：返回降维数据使用：fit_transform(X) True：返回降维数据使用：fit(X).transform(X)
    whiten : bool, optional (default False) 降维后的数据进行白化处理（标准差归一化处理）。
    tol : float >= 0, optional (default .0) 公差：在svd_solver == ‘arpack’计算奇异值方法中需要使用的公差。
    iterated_power : int >= 0, or ‘auto’, (default ‘auto’)使用svd_solver == ‘randomized’计算SVD需要用到的幂法迭代次数。
    random_state : int, RandomState instance or None, optional (default None)
     如果为int，则随机数生成器使用的种子；如果为random state实例，则随机数生成器为randomstate；如果为none，
     则随机数生成器为np.random使用的randomstate实例。当svd_solver=='arpack'或'randomized'时使用。
    提示：尽管上面参数显得有点复杂，并涉及数学计算的概念，但大部分情况下，除了第一个参数，其他参数我们采用默认参数即可。
    """
    model = PCA(
        n_components=0.99,  # 指定保留的特征数量（降维以后的维数）
        copy=True,  # 训练过程中是否影响原始样本数据，也影响到最后降维数据的返回方式
        whiten=True,  # 是否对降维后的数据提供白化处理。（白化就是每一维的特征做一个标准差归一化处理，除以一个标准偏差）
        svd_solver='auto',  # SVD分解的计算方法
        tol=0.0,  # 计算奇异值的公差
        iterated_power='auto',  # 采用svd_solver采用randomized方式的迭代次数。
        random_state=None
    )
    model.fit(x_train)  # fit(self, X，Y=None) #模型训练，由于PCA是无监督学习，所以Y=None，没有标签。
    X_new = model.fit_transform(x_train)
    # print('降维后的数据:', X_new)
    # ratio = model.explained_variance_ratio_
    # print('保留主成分的方差贡献率:', ratio)
    df = pd.DataFrame(X_new)
    # 把标准化数据更新回原表
    df_all = pd.concat([y_train, end_price], axis=1)
    print(df_all)
    df_all2 = pd.concat([df_all, df], axis=1)
    print(df_all2)
    with sqlite3.connect(db_path) as conn:
        # """如果不存在就创建表"""
        df_all2.to_sql('pac_all_integration_other_volume_price', con=conn, if_exists='replace', index=False)
    """（某个事件x发生概率）P(x|θ) 是参数 θ(概率) 下 x（样本） 出现的可能性，同时也是 x 出现时参数为 θ 的似然。
    P(x|θ) 越大，就说明参数如果为 θ，你的观测数据（或事件） x 就越可能出现。那你既然已经观测到数据 x 了，
    那么越可能让这个 x 出现的参数 θ，就越是一个靠谱的模型参数。这个参数的靠谱程度，就是似然。
    概率用于在已知一些参数的情况下，预测接下来的观测所得到的结果p；
    而似然性则是用于在已知某些观测所得到的结果时，对有关事物的性质的参数进行估计。
    """
    # Maxcomponent = model.components_
    # print('返回具有最大方差的成分:', Maxcomponent)
    # score = model.score(x_train)
    # print('所有样本的log似然平均值:', score)
    # print('奇异值:', model.singular_values_)  # 事件特征值中那些可以表征事务特征的值，把平庸不太能表征的元素去掉
    # print('噪声协方差:', model.noise_variance_)
    """inverse_transform(self, X)#将降维后的数据转换成原始数据，但可能不会完全一样
    2. explained_variance_：它代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分。
    3. explained_variance_ratio_：它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。（主成分方差贡献率）
    4. singular_values_：返回所被选主成分的奇异值。
    """
# pca_analysis(your_db_path)
