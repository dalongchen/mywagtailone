import pandas as pd
import sqlite3
your_db_path = r"D:\myzq\axzq\T0002\stock_load\thesis\statistics\my_statis\my_statistics.db"  # 数据库路径


# pandas自动建表并读入数据.
# db_path：数据库路径, path：文件路径, table_name表名
def pandas_create_table(db_path, path="", table_name="", df=""):
    if path:
        df = pd.read_excel(path, engine="openpyxl")
        # data = df.values
        # print("获取到所有的值:\n{}".format(tuple(df.keys())))
        # print("获取到所有的值:\n{}".format(data))
    # data_frame = pd.DataFrame(df)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        # aa = df.keys()
        # aa = tuple(df.keys())
        """如果不存在就创建表"""
        create_sql = """CREATE TABLE IF NOT EXISTS {}{}""".format(table_name, tuple(df.keys()))
        cur.execute(create_sql)
        df.to_sql(table_name, con=conn, if_exists='replace', index=False)
        cur.close()


# 转换excel数据入数据库
# def my_statistics():
#     db_path = r"D:\myzq\axzq\T0002\stock_load\thesis\statistics\my_statistics.db"
#     path = r"D:\myzq\axzq\T0002\stock_load\thesis\statistics\数据集1.xlsx"
#     table_name = "my_statistics2"
#     tools.pandas_create_table(db_path, path, table_name)
# my_statistics()


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
def dimensionless(db_path):
    from sklearn.preprocessing import StandardScaler  # 数据标准化
    scale = StandardScaler()  # 实例化，数据标准化
    with sqlite3.connect(db_path) as conn:
        select_drop_null_column = """select
          Case_ID,
          label,
          SEX,
          MARITAL_STATUS,
          HYPERTENTION,

          HYPERLIPIDEMIA,
          A_S,
          CEREBRAL_APOPLEXTY,
          CAROTID_ARTERY_STENOSIS,
          FLD,

          CIRRHOSIS,
          CLD,
          PANCREATIC_DISEASE,
          BILIARY_TRACT_DISEASE,
          NEPHROPATHY,

          RENAL_FALIURE,
          NERVOUS_SYSTEM_DISEASE,
          CHD,
          MI,
          CHF,

          ARRHYTHMIAS,
          RESPIRATORY_SYSTEM_DISEASE,
          LEADDP,
          HEMATONOSIS,
          RHEUMATIC_IMMUNITY,

          PREGNANT,
          ENDOCRINE_DISEASE,
          MEN,
          PCOS,
          DIGESTIVE_CARCINOMA,

          UROLOGIC_NEOPLASMS,
          GYNECOLGICAL_TUMOR,
          BREAST_TUMOR,
          LUNG_TUMOR,
          INTRACRANIAL_TUMOR,

          OTHER_TUMOR,

          AGE,
          HEIGHT,
          WEIGHT,
          BP_HIGH,
          BP_LOW,

          BMI,
          GLU,
          HBA1C,
          TG,
          TC,

          HDL_C,
          LDL_C,
          FBG,
          BU,
          SCR,

          SUA,
          HB,
          PCV,
          PLT,
          TBILI,

          DBILI,
          TP,
          ALB,
          LDH_L,
          ALT,

          AST,
          GGT,
          ALP,
          PT,
          PTA,

          APTT,
          IBILI,
          GLO
          from mean_fill_null"""
        df = pd.read_sql(select_drop_null_column, conn)
        df1 = df.iloc[:, 0:36]  # 访问第0-35列
        df2 = df.iloc[:, 36:]  # 访问第36-69列对数据标准化
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
        result = pd.concat([df1, num_df], axis=1)
        # print(result)
        # pandas自动建表并读入数据.db_path：数据库路径, path：文件路径, table_name表名
        pandas_create_table(db_path, path="", table_name="dimensionless", df=result)

# dimensionless(your_db_path)


def null_sum(db_path):
    with sqlite3.connect(db_path) as conn:
        select_table = """select * from my_statistics"""
        df = pd.read_sql(select_table, conn)
        dd = df['TH2']
        isnull_sum = dd.isnull().sum()
        isnull_sum2 = (dd == " ").sum()
        isnull_sum3 = (dd == "").sum()
        # isnull_sum = df['NATION'].isnull().sum()
        # aa = (df['NATION'] == 0).astype(int).sum(axis=0)
        print(isnull_sum, isnull_sum2, isnull_sum3, "isnull:", isnull_sum+isnull_sum2+isnull_sum3)

        # del df['NATION']  # 删除19个空列，改变原始数据
        # del df['HEART_RATE']
        # del df['GLU_2H'] = 2319
        # del df['GSP']
        # del df['UPR_24']
        # del df['BUN']
        # del df['UCR']
        # del df['CP']

        # del df['INS']
        # del df['ESR']

        # del df['LP_A']
        # del df['PL']

        # del df['FIBRIN']
        # del df['ALB_CR']

        # del df['LPS']
        # del df['CA199']

        # del df['CRP']
        # del df['M1_M2']
        # del df['TH2']
        # pandas_create_table(db_path, path="", table_name="drop_null_column", df=df)
# null_sum(your_db_path)
