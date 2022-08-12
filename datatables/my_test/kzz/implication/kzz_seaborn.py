import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import global_variable as gl_v

# matplotlib不支持显示中文的 显示中文需要一行代码设置字体
mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）


class KzzSeaBorn(object):

    @staticmethod
    def sea_bs_mc(tab_b, tab_s):
        with sqlite3.connect(gl_v.db_path) as conn:
            df_b = pd.read_sql(r"select date,close,bs_option_kzz, general_mc, optimal_mc from '{}'".format(tab_b), conn)
            df_s = pd.read_sql(r"select date,close from '{}' where date>='2020-11-13'".format(tab_s), conn)
            # data = pd.read_sql(r"select log_ from '{}' ORDER BY date DESC".format(tab), conn)
            # print(df.head())
            df_b['s_close'] = df_s['close']
            sns.lineplot(x="date", y="close", data=df_b)
            sns.lineplot(x="date", y="bs_option_kzz", data=df_b)
            sns.lineplot(x="date", y="general_mc", data=df_b)
            sns.lineplot(x="date", y="optimal_mc", data=df_b)
            sns.lineplot(x="date", y="s_close", data=df_b)
            plt.title('Title using Matplotlib Function')
            plt.show()

    @staticmethod
    def sea_volatility(tab_b, tab_s):
        with sqlite3.connect(gl_v.db_path) as conn:
            aaa = """select date,close,bs_option_kzz,general_mc,optimal_mc,implication_volatility from '{}'""".format(tab_b)
            df_b = pd.read_sql(aaa, conn, parse_dates=["date"])
            df_b = df_b.set_index('date')
            # print(df_b.head())
            bbb = """select close, y_v_log from '{}' where date>='2020-11-13'""".format(tab_s)
            df_s = pd.read_sql(bbb, conn)
            df_b['s_close'] = df_s['close'].astype('float')
            df_b['y_v_log'] = df_s['y_v_log']
            # ccc = df_b.iloc[:, 1:].astype('float')
            # df_b.iloc[:, 1:] = (ccc - ccc.min()) / (ccc.max() - ccc.min())
            # print(df_b.columns)
            plt.figure(figsize=(15, 8))
            # sns.lineplot(x="date", y="s_close", data=df_b)
            # sns.lineplot(x="date", y="close", data=df_b)
            # sns.lineplot(x="date", y="bs_option_kzz", data=df_b)
            # sns.lineplot(x="date", y="general_mc", data=df_b)
            # sns.lineplot(x="date", y="optimal_mc", data=df_b)
            # sns.lineplot(x="date", y="y_v_log", data=df_b)
            # sns.lineplot(data=df_b['general_mc'])
            # sns.lineplot(df_b['date'], df_b.iloc[:, -1:])
            sns.lineplot(data=df_b)
            # sns.lineplot(data=df_b[['s_close']])

            # sns.lineplot(data=df_b['implication_volatility'].astype('float'))
            # sns.relplot(kind="line", data=df_b['implication_volatility'].astype('float'))
            plt.title('Title using Matplotlib ')
            plt.show()

    @staticmethod  # bs下隐含波动率,历史波动率和价格
    def implication_v_log_close(df_b):
        # sns.lineplot(data=df_b['y_v_log'])
        # plt.title('大参可转债历史波动率 ')
        # plt.title('柳药可转债历史波动率 ')
        # sns.lineplot(data=df_b['implication_volatility'])
        # plt.title('大参可转债隐含波动率 ')
        # sns.lineplot(data=df_b[['implication_volatility', 'y_v_log']])
        # plt.title('大参隐含波动率、历史波动率对比图')
        sns.lineplot(data=df_b[['implication_volatility', 'y_v_log', 'close']])
        plt.title('大参隐含波动率、历史波动率和转债收盘价对比图')
        plt.show()

    @staticmethod  # 蒙特卡洛下隐含波动率,历史波动率、bs隐含波动率和价格
    def mc_implication_v_log_close(df_b, name):
        if name == '大参':
            # sns.lineplot(data=df_b[['optimal_mc_vol', 'y_v_log']])
            # plt.title('{}蒙特卡洛隐含波动率、历史波动率对比图'.format(name))
            # sns.lineplot(data=df_b[['optimal_mc_vol', 'implication_volatility', 'y_v_log']])
            # plt.title('{}蒙特卡洛隐含波动率、bs隐含波动率,历史波动率'.format(name))
            sns.lineplot(data=df_b[['optimal_mc_vol', 'implication_volatility', 'y_v_log', 'close']])
            plt.title('{}隐含波动率、bs隐含波动率、历史波动率和转债收盘价'.format(name))
        if name == '柳药':
            # sns.lineplot(data=df_b[['optimal_mc_vol', 'y_v_log']])
            # plt.title('{}蒙特卡洛隐含波动率、历史波动率对比图'.format(name))
            # sns.lineplot(data=df_b[['optimal_mc_vol', 'implication_volatility', 'y_v_log']])
            # plt.title('{}蒙特卡洛隐含波动率、bs隐含波动率,历史波动率'.format(name))
            sns.lineplot(data=df_b[['optimal_mc_vol', 'implication_volatility', 'y_v_log', 'close']])
            plt.title('{}隐含波动率、bs隐含波动率、历史波动率和转债收盘价'.format(name))
        plt.show()

    @staticmethod  # 债券价格和bs计算的债券价值
    def bound_close_bs_value(df_b):
        sns.lineplot(data=df_b[['bs_option_kzz', 'close']])
        # plt.title('大参实际债券价格和bs算法价值')
        plt.title('柳药实际债券价格和bs算法价值')
        plt.show()

    @staticmethod  # 普通蒙特卡洛价值
    def bound_close_mc_value(df_b):
        sns.lineplot(data=df_b[['general_mc', 'bs_option_kzz', 'close']])
        # plt.title('大参蒙特卡洛价值和bs算法价值')
        plt.title('柳药实际债券价格和bs算法价值')
        plt.show()

    @staticmethod  # 最优停时蒙特卡洛价值
    def optimal_close_mc_value(df_b):
        sns.lineplot(data=df_b[['optimal_mc', 'general_mc', 'bs_option_kzz', 'close']])
        # plt.title('大参最优停时蒙特卡洛价值')
        plt.title('柳药最优停时蒙特卡洛价值')
        plt.show()

    @staticmethod  # 是否归一化
    def sea_history_implication_volatility(tab_b, tab_s, f_normal):
        with sqlite3.connect(gl_v.db_path) as conn:
            sql_b = """select date,close,bs_option_kzz,general_mc,optimal_mc,implication_volatility from '{}'""".format(
                tab_b)
            df_b = pd.read_sql(sql_b, conn, parse_dates=["date"])
            date_1 = df_b['date'].head(1).apply(lambda x: x.strftime('%Y-%m-%d'))[0]
            print(date_1)
            sql_s = """select close, y_v_log from '{}' where date>='{}'""".format(tab_s, date_1)
            df_s = pd.read_sql(sql_s, conn)
            df_b['s_close'] = df_s['close']
            df_b['y_v_log'] = df_s['y_v_log']
            df_b = df_b.set_index('date')
            df_b = df_b.astype('float')
            if f_normal == 'y':   # 是否归一化
                df_b = (df_b - df_b.min()) / (df_b.max() - df_b.min())  # 归一化
            plt.figure(figsize=(15, 8))

            dd = "implication_v_log_close"
            if dd == "implication_v_log_close":  # 隐含波动率,历史波动率和债券价格
                KzzSeaBorn.implication_v_log_close(df_b)
            if dd == "bound_close_bs_value":  # 债券价格和bs计算的债券价值
                KzzSeaBorn.bound_close_bs_value(df_b)
            if dd == "bound_close_mc_value":  # 普通蒙特卡洛价值
                KzzSeaBorn.bound_close_mc_value(df_b)
            if dd == "optimal_close_mc_value":  # 普通蒙特卡洛价值
                KzzSeaBorn.optimal_close_mc_value(df_b)

    @staticmethod  # 蒙特卡洛下隐含波动率 f_normal：是否归一化
    def mc_history_implication_volatility(tab_b, tab_s, f_normal, name):
        with sqlite3.connect(gl_v.db_path) as conn:
            sql_b = """select date,close,implication_volatility,optimal_mc_vol from '{}'""".format(
                tab_b)
            df_b = pd.read_sql(sql_b, conn, parse_dates=["date"])
            date_1 = df_b['date'].head(1).apply(lambda x: x.strftime('%Y-%m-%d'))[0]
            # print(date_1)
            sql_s = """select close, y_v_log from '{}' where date>='{}'""".format(tab_s, date_1)
            df_s = pd.read_sql(sql_s, conn)
            df_b['s_close'] = df_s['close']
            df_b['y_v_log'] = df_s['y_v_log']
            df_b = df_b.set_index('date')
            df_b = df_b.astype('float')
            if f_normal == 'y':  # 是否归一化
                df_b = (df_b - df_b.min()) / (df_b.max() - df_b.min())  # 归一化
            plt.figure(figsize=(15, 8))

            dd = "mc_implication_v_log_close"
            if dd == "mc_implication_v_log_close":  # 隐含波动率,历史波动率和债券价格
                KzzSeaBorn.mc_implication_v_log_close(df_b, name)

    @staticmethod  # 画收益率直方图，检验是否正态分布。
    def histogram_or_is_normal_dis(tab_b, tab_s):
        import scipy.stats as scs
        with sqlite3.connect(gl_v.db_path) as conn:
            se_sql = r"select date,log_ from '{}'".format(tab_s)
            df_b = pd.read_sql(se_sql, conn, parse_dates=["date"])
            # df_b = df_b.set_index('date')

            df_b = df_b['log_'][-500:] / 100
            print('Norm test p-value %14.3f' % scs.normaltest(df_b)[1])
            # v = np.random.normal(size=300)
            # print(scs.normaltest(v))
            norm = scs.norm.rvs(loc=0, scale=1, size=1000)  # rvs表示生成指定分布的分布函数
            print(scs.normaltest(norm))

            plt.figure(figsize=(15, 8))

            plt.hist(df_b, bins=40, edgecolor='k')
            plt.title('Title using Matplotlib ')
            plt.show()


if __name__ == '__main__':
    ksb = KzzSeaBorn()
    # ksb.sea_bs_mc(tab_b=gl_v.tab_b, tab_s=gl_v.tab_s)
    # ksb.sea_volatility(tab_b=gl_v.tab_b, tab_s=gl_v.tab_s)

    # ksb.sea_history_implication_volatility(tab_b=gl_v.da_sen_b, tab_s=gl_v.da_sen_s, f_normal='y')  # 大参
    # ksb.sea_history_implication_volatility(tab_b=gl_v.liu_yao_b, tab_s=gl_v.liu_yao_s, f_normal='')  # 柳药

    # 蒙特卡洛下隐含波动率
    # ksb.mc_history_implication_volatility(tab_b=gl_v.da_sen_b, tab_s=gl_v.da_sen_s, f_normal='y', name='大参')  # 大参
    ksb.mc_history_implication_volatility(tab_b=gl_v.liu_yao_b, tab_s=gl_v.liu_yao_s, f_normal='y', name='柳药')

    # ksb.histogram_or_is_normal_dis(tab_b=gl_v.liu_yao_b, tab_s=gl_v.liu_yao_s)  # 柳药


