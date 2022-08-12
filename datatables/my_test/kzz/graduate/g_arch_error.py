import pandas as pd
import sqlite3
import global_variable as gl_v


class KzzStockArch(object):
    def __init__(self):
        pass

    def adf_test(self, tab):
        with sqlite3.connect(gl_v.db_path) as conn:
            data = pd.read_sql(r"select log_ from '{}'".format(tab), conn)
            # data = pd.read_sql(r"select log_ from '{}' ORDER BY date DESC".format(tab), conn)
            data1 = data.dropna()
            data2 = data1.iloc[:122*2]

            dd = "ljungbox"  # 时间序列平稳性单位根检验
            if dd == "adf":
                self.KzzStockArch.get_adfuller(data)

            if dd == "ljungbox":  # ljungbox检验
                self.get_ljungbox(data2)

            dd = ""  #
            if dd == "garch":
                import arch
                """赤池信息准则（Akaike Information Criterion，AIC）和贝叶斯信息准则（Bayesian Information Criterion，BIC）
                选择最佳模型时，通常选择AIC最小的模型.const coef:10.2779 y截距，即b值。std err，反映系数的准确度，越低，
                准确度越高. P>|t|:p值，判断回归参数是否有统计学意义.
                Confidence Interval：置信区间，表示我们的系数可能范围（可能性为 95%）
                在模型中，增加多个变量，即使事实上无关的变量，也会小幅度条R平方的值，当时其是无意义，所有我们调整了下，
                降低R平方的值。用r square的时候，不断添加变量能让模型的效果提升，而这种提升是虚假的。利用adjusted r square，
                能对添加的非显著变量给出惩罚，也就是说随意添加一个变量不一定能让模型拟合度上升
                标准误 （Standard Error, SE）"""
                # am = arch.arch_model(data, vol='GARCH')
                # am = arch.arch_model(data, mean='AR', vol='GARCH', dist='gaussian',)
                # am = arch.arch_model(data, p=1, q=1, o=0, power=2.0, vol='Garch', dist='StudentsT')
                am = arch.arch_model(data, vol='Garch', dist='StudentsT')
                am = am.fit()
                # am = am.fit(update_freq=0, disp='off')
                # print(am.summary())
                # pre = tempmodel.forecast(horizon=pst, start=lag - 1, method='simulation').mean.iloc[lag - 1]
                # print(pre)
                # train = data.diff(1).dropna()
                # -0.056757
                # data = data['log_']-data['log_'].mean()
                # print("kk", )
                train = data[:-10]
                test = data[-10:]
                am = arch.arch_model(train, mean='AR', lags=1, vol='ARCH', )
                # am = arch.arch_model(train, mean='AR', lags=1, vol='GARCH',)
                res = am.fit()
                print(res.summary())

    @staticmethod
    def get_ljungbox(data):  # ljungbox检验
        from statsmodels.stats.diagnostic import acorr_ljungbox
        """
                    acorr_ljungbox(x, lags=None, boxpierce=False, model_df=0, period=None,return_df=None, auto_lag=False)
                    x: 观测序列
                    lags为延迟阶数，整数/列表，如果是一个整数，返回延迟1阶～延迟该指定整数阶的检验结果；如果是一个列表，
                    仅返回列表中指定的延迟阶数的检验结果；如果为None，默认值为min((nobs // 2 - 2), 40)，
                    nobs就是观测序列中样本个数。
                    boxpierce: 布尔类型，如果为True，不仅返回QLB统计量检验结果还返回QBP统计量检验结果；默认为False，
                    仅返回QLB统计量检验结果。
                    model_df: degree of freedom，即模型的自由度，默认值为0，一般检验单个序列默认值即可；
                    当检验模型拟合后的残差序列时，残差序列中观测值的个数不能表示自由度
                    （个人理解就是里面随机变量的个数，比如方差中无偏估计的因为已知均值，结合前n-1项就能算出第n项，
                    故自由度为样本个数减1）
                    如ARMA拟合后的残差检验时模型自由度为p+q，计算完Q统计量查找P值时，卡方分布的自由度应为lags-p-q，
                    此时指定model_df=p+q。
                    period: 季节性序列周期大小，帮助确定最大延迟阶数
                    return_df: 是否以DataFrame格式返回结果，默认False以元组方式返回结果
                    auto_lag: 根据自相关性自动计算最佳最大延迟阶数，指定时报错了，pass

                    返回值：
                    lbvalue: QLB检验统计量（Ljung-Box检验）
                    pvalue: QLB检验统计量下对应的P值（Ljung-Box检验）
                    bpvalue: QBP检验统计量，boxpierce为False时不反回（Box-Pierce检验）
                    bppvalue: QBP检验统计量下对应的P值，boxpierce为False时不反回（Box-Pierce检验）

                    第1行为Ljung-Box统计量,第2行为p值
                    如果p<0.05，拒绝原假设，说明原始序列存在相关性
                    如果p>=0.05，接收原假设，说明原始序列独立，纯随机
                    若是白噪声数据，则该数据没有价值提取，每一个P值都小于0.05或等于0，说明该数据不是白噪声数据，数据有价值，
                    可以继续分析。反之如果大于0.05，则说明是白噪声序列，是纯随机性序列。
                    """
        # ljungbox_result = acorr_ljungbox(data, lags=20)  # 返回统计量和p值，lags为检验的延迟数
        # ljungbox_result = acorr_ljungbox(data,  lags=[1, 2, 3, 6, 12, 24], return_df=True)
        ljungbox_result = acorr_ljungbox(data, return_df=True, boxpierce=True,)  # 返回统计量和p值，lags为检验的延迟数
        # ljungbox_result = acorr_ljungbox(data, return_df=True, boxpierce=True, auto_lag=True)  # 返回统计量和p值，lags为检验的延迟数
        print(ljungbox_result)

    @staticmethod  # 时间序列平稳性单位根检验
    def get_adfuller(data):
        from statsmodels.tsa.stattools import adfuller
        """
                    (-0.04391111656553232, 0.9547464774274733, 10, 22,
                    {'1%': -3.769732625845229, '5%': -3.005425537190083, '10%': -2.6425009917355373}, 291.54354258641223)
                    第一个是adt检验的结果，简称为T值，表示t统计量。
                    第二个简称为p值，表示t统计量对应的概率值。
                    第三个表示延迟。
                    第四个表示测试的次数。
                    第五个是配合第一个一起看的，是在99%，95%，90%置信区间下的临界的ADF检验的值。
                    第一点，1%、%5、%10不同程度拒绝原假设的统计值和ADF Test result的比较，
                    ADF Test result同时小于1%、5%、10%即说明非常好地拒绝该假设。
                    本数据中，adf结果为-0.04391111656553232，大于三个level的统计值，接收假设，即存在单位根。
                    第二点，p值要求小于给定的显著水平，p值要小于0.05，等于0是最好的。本数据中，
                    P-value 为 0.9547464774274733,大于三个level，接受假设，即存在单位根。
                    ADF检验的原假设是存在单位根，只要这个统计值是小于1%水平下的数字就可以极显著的拒绝原假设，
                    认为数据平稳。注意，ADF值一般是负的，也有正的，但是它只有小于1%水平下的才能认为是及其显著的拒绝原假设。
                    对于ADF结果在1% 以上 5%以下的结果，也不能说不平稳，关键看检验要求是什么样子的。
                    adfuller(
                        x,
                        maxlag=None,
                        regression="c",
                        autolag="AIC",
                        store=False,
                        regresults=False,
                    )
                    (-11.450545565916247, 5.8728395141112274e-21, 1, 242,
                    {'1%': -3.457664132155201, '5%': -2.8735585105960224, '10%': -2.5731749894132916}, 1150.6699808503472)
                    """
        result = adfuller(data.dropna(), regresults=False, )
        # print(result)
        print("\nresult is\n{}".format(result))
        result_fromat = pd.Series(result[0:4], index=['Test Statistic', 'p-value', 'Lags Used',
                                                      'Number of Observations Used'])
        for k, v in result[4].items():
            result_fromat['Critical Value (%s)' % k] = v
        result_fromat['The maximized information criterion if autolag is not None.'] = result[5]
        print("\nresult_fromat is\n{}".format(result_fromat))
        print("\n\n===== adfuller()的回归模型系数 =====")
        [t, p, c, r] = adfuller(x=data, regression='ctt', regresults=True)
        print("r.resols.summary() is")
        print(r.resols.summary())
        print("\nr.resols.params are")
        print(r.resols.params)


ksa = KzzStockArch()
# ksa.adf_test(tab='sh.000001')
# ksa.adf_test(tab='603233')  # 大参
ksa.adf_test(tab='603368')  # 柳药
# ksa.adf_test(tab='sz.002624')