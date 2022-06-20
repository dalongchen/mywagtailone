import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


class AmericanOptionsLSMC(object):
    def __init__(self, S0, strike, T, M, r, sigma, simulations):
        try:
            self.S0 = float(S0)
            self.strike = float(strike)
            assert T > 0
            self.T = float(T)
            assert M > 0
            self.M = int(M)
            assert r >= 0
            self.r = float(r)
            assert sigma > 0
            self.sigma = float(sigma)
            assert simulations > 0
            self.simulations = int(simulations)
        except ValueError:
            print('Error passing Put Options parameters')

        if S0 < 0 or strike < 0 or T <= 0 or r < 0 or sigma < 0:
            raise ValueError('Error: Negative inputs not allowed')

        self.time_unit = self.T / float(self.M)
        self.discount = np.exp(-self.r * self.time_unit)

    # bs期权的定价计算 ．
    def bsm_call_value(S_0, K, T, r, sigma):
        from math import log, sqrt, exp
        from scipy import stats
        """stats.norm.cdf(α,均值,方差)：累积概率密度函数, 相当于已知正态分布函数曲线和x值，求函数x点左侧积分
    使用实例a=st.norm.cdf(0,loc=0,scale=1)
    math.exp()它接受一个数字并以指数格式返回该数字(如果数字为x ，则返回e ** x"""
        S_0 = float(S_0)
        d_1 = (log(S_0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d_2 = (log(S_0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        C_0 = (S_0 * stats.norm.cdf(d_1, 0.0, 1.0) - K * exp(-r * T) * stats.norm.cdf(d_2, 0.0, 1.0))
        return C_0

    def bs_father():
        """C—期权初始合理价格
        L—期权交割价格
        S—所交易金融资产现价
        T—期权有效期
        r—连续复利计无风险利率
        σ2—年度化方差"""
        S_0 = 100.0  # 股票或指数初始的价格;
        K = 105  # 行权价格,L—期权交割价格
        T = 1.0  # 期权的到期年限(距离到期日时间间隔)
        r = 0.05  # 无风险利率
        sigma = 0.2  # 波动率(收益标准差)
        # 到期期权价值
        import time
        tis1 = time.perf_counter()
        print(bsm_call_value(S_0, K, T, r, sigma))
        tis2 = time.perf_counter()
        print("run time: ", tis2 - tis1)

    # bs_father()

    # 欧式看涨期权的定价公式Black-Scholes-Merton(1973)
    def balck_scholes_merton():
        from time import time
        from math import exp, sqrt, log
        from random import gauss, seed  # 随机生成正态分布,返回具有高斯分布的隨機浮點數
        import matplotlib.pyplot as plt

        seed(2000)
        # 计算的一些初始值
        S_0 = 100.0  # 股票或指数初始的价格;不是可转债价格
        K = 105  # 行权价格
        T = 1.0  # 期权的到期年限(距离到期日时间间隔)
        r = 0.05  # 无风险利率
        sigma = 0.2  # 波动率(收益标准差)
        M = 50  # number of time steps
        # M = 50  # number of time steps
        dt = T / M  # 一年的1/50
        I = 20000  # number of simulation

        start = time()
        S = []  #
        for i in range(I):
            path = []  # 时间间隔上的模拟路径
            for t in range(M + 1):
                if t == 0:
                    path.append(S_0)
                else:
                    z = gauss(0.0, 1.0)
                    # print("z", z)
                    S_t = path[t - 1] * exp((r - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * z)
                    path.append(S_t)
                    # break
            S.append(path)
            # print(path[-1])
            # print(path[-1] - K)
            # print(max(path[-1] - K, 0))
            # print("path", path)
            # break
        # 计算期权现值,max(path[-1] - K, 0)==path[-1] - K<0 则取0
        C_0 = exp(-r * T) * sum([max(path[-1] - K, 0) for path in S]) / I
        total_time = time() - start
        print('European Option value %.6f' % C_0)
        print('total time is %.6f seconds' % total_time)

        # plt.figure(figsize=(10, 7))
        # plt.grid(True)
        # plt.xlabel('Time step')
        # plt.ylabel('index level')
        # for i in range(30):
        #     plt.plot(S[i])
        # plt.show()

    # balck_scholes_merton()

    # 欧式看涨期权的定价公式Black-Scholes-Merton(1973),向量化实现
    def balck_scholes_merton_numpy():
        import numpy as np
        from time import time

        S_0 = 100.0  # 股票或指数初始的价格;
        K = 105  # 行权价格
        T = 1.0  # 期权的到期年限(距离到期日时间间隔)
        # T = 252  # Number of trading days一年
        r = 0.05  # 无风险利率
        sigma = 0.2  # 波动率(收益标准差)
        M = 50  # number of time steps
        dt = T / M  # time enterval
        I = 20000  # number of simulation

        # 20000条模拟路径，每条路径５０个时间步数
        S = np.zeros((M + 1, I))
        S[0] = S_0
        np.random.seed(2000)
        start = time()
        for t in range(1, M + 1):
            z = np.random.standard_normal(I)
            S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
        C_0 = np.exp(-r * T) * np.sum(np.maximum(S[-1] - K, 0)) / I  # np.maximum(S[-1] - K, 0)期权最少为0，不能负数
        end = time()

        # 估值结果
        print('total time is %.6f seconds' % (end - start))
        print('European Option Value %.6f' % C_0)
        ss = sum(S[-1] < K)  # 在两万次模拟中超过一万次到期期权内在价值为０
        print('%.6f' % ss)  # 10798.000000
        # 查看潜在价格分布的几个“分位数”，以了解非常高或非常低回报的可能性。使用numpy的“percentile”函数来计算
        # 5 % 和95 % 的分位数：
        print("5% quantile =", np.percentile(S, 5))
        print("95% quantile =", np.percentile(S, 95))
        d = ''
        if d:
            # 前２０条模拟路径
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 7))
            plt.grid(True)
            plt.xlabel('Time step')
            plt.ylabel('index level')
            for i in range(20):
                plt.plot(S.T[i])
            plt.show()

            # 到期时所有模拟指数水平的频率直方图
            plt.hist(S[-1], bins=50)
            plt.grid(True)
            plt.xlabel('index level')
            plt.ylabel('frequency')
            # Text(0, 0.5, 'frequency')
            plt.show()

            # 模拟期权到期日的内在价值,上面C_0是期权到期日的内在价值平均数
            plt.hist(np.maximum(S[-1] - K, 0), bins=50)
            plt.grid(True)
            plt.xlabel('option inner value')
            plt.ylabel('frequency')
            plt.show()

    # balck_scholes_merton_numpy()

    # 美式期权定价（最小二乘蒙特卡洛模拟法，LSM）,K行权价
    def gmb_mcs_amer(self, option='call'):
        import math
        from time import time
        strike = self.strike
        S0 = self.S0
        r = self.r
        sigma = self.sigma
        T = self.T
        # I = 50
        I = self.simulations
        M = self.M
        dt = T / M
        discount = math.exp(-r * dt)
        # df = math.exp(-r * dt)
        M += 1
        S = np.zeros((M, I))
        S[0] = S0
        start = time()
        np.random.seed(2000)  # 数值随便指定，指定了之后对应的数值唯一
        print(S.shape)
        # sn = np.random.standard_normal(size=[M, I])  # M行I列的二维数组
        # print(sn)
        for t in range(1, M):  # M+1的作用是S[t - 1]
            """np.concatenate((a1,a2,…), axis=0)，0表示以列拼接，能够一次完成多个数组的拼接。其中a1,a2,…是数组
            //取整除 - 返回商的整数部分（向下取整） 9//2=4, -9//2 = -5,对偶变量法？"""
            brownian = np.random.standard_normal(I // 2)
            sn = np.concatenate((brownian, -brownian))
            # sn = np.random.standard_normal(I)
            S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * sn)
            # S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * sn[t])
            if t == 1:
                print("aa", t, S[t])
        if option == 'call':
            # print("k:", K)
            h = np.maximum(S - strike, 0)
        else:
            h = np.maximum(strike - S, 0)
        V = np.copy(h)
        # print(V)
        # print(V.shape)
        # print(S)
        # print(S.shape)
        """range(start,end,step)注：start 指的是计数起始值，默认是 0；stop 指的是计数结束值，但不包括 stop ；step 是步长，
        默认为 1，不可以为 0 。range() 方法生成一段左闭右开的整数范围。在使用range()函数时，如果只有一个参数，就表示end，
        从0开始；如果有两个参数，就表示start和end；如果有三个参数，最后一个就表示步长。它接收的参数必须是整数，可以是负数，
        但不能是浮点数等其它类型。
    """
        # for t in range(M - 2, 0, -1):  # 49-1, 这里最大49是为了后面V[t + 1]
        for t in range(M - 2, 0, -1):  # 49-1, 这里最大49是为了后面V[t + 1]
            # print(t)
            """np.polyfit(x,y,deg=1) x，y就可以理解为x和y坐标了，deg就是阶数，阶数是自变量的最高次方.
            制定deg参数为2的话即返回抛物线的参数[L,k,b]=Lx**2+kx**1+b"""
            reg = np.polyfit(S[t], V[t + 1] * discount, 5)
            # reg = np.polyfit(S[t], V[t+1] * dt, 7)
            # reg = np.polyfit(S[t-1], V[t] * dt, 7)
            # reg = np.polyfit(S[t], V[t - 1] * dt, 7)  # np.polyfit(S[t-1], V[t] * dt, 7)
            # print(reg)
            # print(np.poly1d(reg))
            continuation_value = np.polyval(reg, S[t])  # 上面是依据x拟合出系数，这里是依据x算出y. 计算当前时刻持有价值
            # C = np.polyval(reg, S[t])
            # if t == M-2:
            if t == 1:
                # print("aa", t, V[t])
                # print("aa2", t, V[t]*dt)
                # print("aa3", t, S[t])
                from sklearn.metrics import r2_score
                R_square = r2_score(V[t + 1] * dt, continuation_value)  # 第一个为实际y，第二为预测y
                # R_square = r2_score(V[t - 1] * dt, C)  # 第一个为实际y，第二为预测y
                print(t, 'R_square: {:.2f}'.format(R_square))  # 得到R平方接近1，该模型拟合度较强，可以用来预测
                # plt.plot(S[t], V[t - 1] * dt, '*', label='original values')
                # plt.plot(S[t], C, 'x', label='polyfit values')
                # # plt.plot(S[t], V[t - 1] * dt)
                # # plt.plot(S[t], V[t - 1] * dt, color="red", linestyle="-", linewidth=2.0, label='label1')
                # plt.show()
                # plt.savefig('p1.png')
            """如果c中t时刻（49）持有价值》h中t时刻立即变现价值，则取t+1时刻（50）期权值折现，否则取当前期权值"""
            V[t] = np.where(continuation_value > h[t], V[t + 1] * discount, h[t])  # 对应元素比较，如果true，取V[t + 1] * df中对应元素
        # print(V)
        # print(V[2])
        C0 = discount * np.mean(V[1])  # 折现到v0时刻
        # C0 = np.mean(V[1]*discount)
        end = time()
        print('total time is %.6f seconds' % (end - start))
        print(C0)
        return C0

    # 考虑下修。欧式看涨期权的定价公式Black-Scholes-Merton(1973),向量化实现
    def down_balck_scholes_merton_numpy():
        import numpy as np
        import pandas as pd
        from time import time
        interest = [0.2, 0.3, 0.5, 1, 2, 3]  # 当期利息
        # S_0 = 75.0  # 股票或指数初始的价格;
        S_0 = 100.0  # 股票或指数初始的价格;
        K = 105  # 行权价格
        """T = (datetime.date(2013,9,21) - datetime.date(2013,9,3)).days / 365.0"""
        T = 1.0  # 期权的到期年限(距离到期日时间间隔)
        # T = 252  # Number of trading days一年
        r = 0.05  # 无风险利率
        # sigma = 1  # 波动率(收益标准差)
        sigma = 0.2  # 波动率(收益标准差)
        M = 250  # number of time steps
        # M = 50  # number of time steps
        """(T-t) 给了我们年化的到期时间。 例如，对于 30 天选项，这将是 30/365=0.082."""
        dt = T / M  # 实际就是t2-t1
        # I = 20000  # number of simulation
        I = 999  # number of simulation
        # I = 10000  # number of simulation
        # I = 20000  # number of simulation
        """（1）下修条件格式如下：在可转债存续期间，当本公司股票在任意连续X个交易日中有Y个交易日的收盘价低于当期转股价格的Z%时，
        公司董事会有权提出转股价格向下修正方案并提交本公司股东大会表决。例如，航信转债的下修条件为：在可转债存续期间，
        当本公司股票在任意连续20个交易日中有10个交易日的收盘价低于当期转股价格的90%时，
        公司董事会有权提出转股价格向下修正方案并提交本公司股东大会表决，
        上述方案须经出席会议的股东所持表决权的三分之二以上通过方可实施。
        （2）下修后的转股价格一般要求不低于股东大会前20个交易日股票交易均价和前1交易日的交易均价，并低于之前的转股价格。
        有的公司还会要求转股价格不得低于最近每股净资产和股票面值。
        （3）发生下修后，其他条款应根据下修后的价格进行重新计算，回售条件的累计时间将重计。
        （7）考虑回售和下修的不确定性，设置下修概率q，假设只有发生回售才能发生下修，
        并假设触发条件时有20%的概率回售，50%的概率下修，30%的概率不下修且不回售。
        回售条款的目的是保护投资者的权益。即当一段时间内股票价格过低时，投资者可以将可转债回售给发行人。
        回售的触发期限较短，一般为最后两年，而且要求较为苛刻，投资者通常一年只有一次回售的机会。
        回售条件一般格式如下：在最后两个计息年度任何连续X个交易日的股票收盘价格低于当期转股价格的Y%时，触发回售。
        例如，航信转债的回售条件为：在最后两个计息年度任何连续30个交易日的股票收盘价格低于当期转股价格的70%时，
        可转债持有人有权将其持有的可转债全部或部分按债券面值加当期应计利息的价格回售给发行人。
        """
        blow = 0.7  # 触发回售条件
        # blow = 0.9
        down = 1.05 * 0.333 + 1.1 * 0.333 + 1.15 * 0.333  # 下修价格下限上浮幅度期望值
        # down = 1.1

        # 20000条模拟路径，每条路径５０个时间步数
        # S = np.zeros((I, int(M + 1)))
        add = 3  # 增加几行
        S = np.zeros((M + add, I))
        print(S.shape)
        S[0] = S_0
        S[-1] = K
        # print(S[-2] - S[-1])
        # print(S[-1])
        # print(S)
        np.random.seed(2000)
        # aaa = np.random.random()
        # print("aaa:", aaa)
        start = time()
        """z = [ 1.73673761  1.89791391 -2.10677342 ...  0.29239831 -0.81623889 0.02075496]"""
        n = 15  # 连续15天
        n2 = 30  # 连续15天
        redeem_i = []  # 保存以及赎回路径
        for t in range(M):
            # print(t)
            z = np.random.standard_normal(I)
            S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
            if t >= n:
                # gg = S[t-n:t]  # 取15(n)行数据，也是15(n)天,15*I
                for i in range(I):  # 循环列I=10000
                    # print("i:", i)
                    if sum(S[t - n:t, i] < S[-1, i] * blow) == n:  # 连续少于行权价0.7的天数切好=15，下修
                        # print("1gg[t, i]", S[t, i], i, t)
                        if i not in redeem_i:
                            S[-1, i] = min(down * max(S[t - 21:t - 1, i].mean(), S[t, i]), S[-1, i]) * 0.8 + S[-1, i] * 0.2
                            if t == M - 1:
                                S[-add, i] = np.exp(-r * (t / 245)) * max((100 / S[-(add + 1), i]) * S[-1, i], 108)
                    if t >= n2:
                        if sum(S[t - n2:t, i] >= S[-1, i] * 1.3) >= n:  # 连续大于行权价1.3的天数大于=15,赎回
                            # print("1gg[t, i]", S[t, i], i, t)
                            if i not in redeem_i:
                                S[-add, i] = np.exp(-r * (t / 245)) * ((100 / S[-1, i]) * S[t, i] + interest[4])
                                redeem_i.append(i)
                                S[-add + 1, i] = t
                if t == M - 1:
                    print("4n_num", S[0:-add])
        C_0 = S[-add].mean()  # np.maximum(S[-1] - K, 0)期权最少为0，不能负数
        # C_0 = np.exp(-r * T) * np.sum(np.maximum(S[-3] - S[-1], 0)) / I  # np.maximum(S[-1] - K, 0)期权最少为0，不能负数
        end = time()

        # 估值结果
        print('total time is %.6f seconds' % (end - start))
        print('European Option Value %.6f' % C_0)
        # ss = sum(S[-1] < K)  # 在两万次模拟中超过一万次到期期权内在价值为０
        le = len(redeem_i)
        print('赎回次数%.6f,占比：' % le, le / I)  # 10798.000000
        # 查看潜在价格分布的几个“分位数”，以了解非常高或非常低回报的可能性。使用numpy的“percentile”函数来计算
        # 5 % 和95 % 的分位数：
        print("5% quantile =", np.percentile(S[0:-add], 5))
        print("95% quantile =", np.percentile(S[0:-add], 95))
        d = ''
        if d:
            import sqlite3
            db_path = r"D:\myzq\axzq\T0002\stock_load\thesis\kzz\monte.db"  # 数据库路径
            with sqlite3.connect(db_path) as conn:
                df = pd.DataFrame(S)
                # print(df)
                df.to_sql('monte', con=conn, if_exists='replace', index=False)
        dd = ''
        if dd:
            # 前２０条模拟路径
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 7))
            plt.grid(True)
            plt.xlabel('Time step')
            plt.ylabel('index level')
            for i in range(20):
                plt.plot(S.T[i])
            plt.show()

            # 到期时所有模拟指数水平的频率直方图
            plt.hist(S[-1], bins=50)
            plt.grid(True)
            plt.xlabel('index level')
            plt.ylabel('frequency')
            # Text(0, 0.5, 'frequency')
            plt.show()

            # 模拟期权到期日的内在价值,上面C_0是期权到期日的内在价值平均数
            plt.hist(np.maximum(S[-1] - K, 0), bins=50)
            plt.grid(True)
            plt.xlabel('option inner value')
            plt.ylabel('frequency')
            plt.show()


            # down_balck_scholes_merton_numpy()


# american = AmericanOptionsLSMC(100., 120., 1., 50, 0.03, 0.35, 10000)
# american = AmericanOptionsLSMC(100., 120., 1., 25, 0.03, 0.35, 26)
# american.gmb_mcs_amer()
# american.balck_scholes_merton()
# american.gmb_mcs_amer(105, option='call')
"""call是看涨期权，put是看跌期权，最简单的理解下，未来股价看涨，你应该买call，未来股价看跌你应该买put。"""
# gmb_mcs_amer(105, option='call')  # 1.4620120629034186
# gmb_mcs_amer(110, option='put')  # 9.918961211536654


class AmericanOptionsLSMC2(AmericanOptionsLSMC):
    def __init__(self, S0, strike, T, M, r, sigma, simulations):
        super().__init__(S0, strike, T, M, r, sigma, simulations)
        # self.current_d = current_d

    def get_price(self):
        # br = np.random.standard_normal(6 * 245)*self.S0
        # br = np.around(abs(br), 2)
        # br = np.zeros(6 * 245)
        br = np.ones(6 * 245, dtype=int)*self.S0
        # br = self.S0
        print("br", br.shape)
        print("br", br)
        # return ""

    def gmb_mcs_amer(self):
        from time import time
        import sqlite3
        # import matplotlib.pyplot as plt
        dt = self.T / self.M
        discount = math.exp(-self.r * dt)
        # self.M += 1
        S = np.zeros((self.M+1, self.simulations))
        s_value = np.copy(S[0])
        s_strike = np.copy(S[0])
        s_strike[:] = self.strike
        # print(s_strike)
        continuation_s = np.copy(S)
        S[0] = self.S0
        start = time()
        np.random.seed(2000)  # 数值随便指定，指定了之后对应的数值唯一
        print(S.shape)
        # sn = np.random.standard_normal(size=[M, I])  # M行I列的二维数组
        arr = []
        for t in range(1, self.M+1):  # M+1的作用是S[t - 1]
            """np.concatenate((a1,a2,…), axis=0)，0表示以列拼接，能够一次完成多个数组的拼接。其中a1,a2,…是数组
            //取整除 - 返回商的整数部分（向下取整） 9//2=4, -9//2 = -5,对偶变量法？"""
            brownian = np.random.standard_normal(self.simulations // 2)
            sn = np.concatenate((brownian, -brownian))
            # sn = np.random.standard_normal(I)
            S[t] = S[t-1] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * math.sqrt(dt) * sn)
            # S[t] = S[t - 1] * np.exp((self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * math.sqrt(dt) * sn)
            # S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * sn[t])
            price_rate = S[t] / s_strike  # 现股价和转股价比率
            if (t >= 0) and t < 123:
                pass
                # print("current_d:", t)
            elif (t >= 699) and t < 700:
                pass
                # price_rate2 达到下修的标准，相对于转股价下跌的幅度, down_p 下修概率
                # self.get_s_down_strike(S, price_rate, s_strike, t, 0.5, 0.5)
                # print("price_rate", price_rate)
                # print("s_strike2", s_strike)
                # print("S[t]", S[t])
                # s_strike2 = np.copy(s_strike)
                # down_p = 0.5
                # price_rate2 = 0.6
                # s_strike = np.where(price_rate < price_rate2, np.maximum(S[t]*1.1*down_p + s_strike*(1 - down_p), S[t]), s_strike)
                # print("s_strike3", s_strike)

                # s_down = np.argwhere(price_rate < price_rate2)  # 找出a中小于0.3的所有元素的下标.下修转股价
                # if len(s_down):
                #     # print("s_down", s_down)
                #     uu = S[t][s_down] * 1.1 * down_p + s_strike[s_down] * (1 - down_p)  # 下修转股价
                #     s_strike[s_down] = np.maximum(uu, S[t][s_down])
            elif (t >= 123) and t < 245*1:
                # pass
                # price_rate2 达到下修的标准，相对于转股价下跌的幅度, down_p 下修概率
                # self.get_s_down_strike(S, price_rate, s_strike, t, 0.5, 0.5)
                s_strike = self.get_s_down_strike(S, price_rate, s_strike, t, 0.5, 0.5)
                self.get_s_value(arr, price_rate, s_value, t)
                # s_down = np.argwhere(price_rate < 0.3)  # 找出a中小于0.3的所有元素的下标.下修转股价
                # # print("s_down", s_down)
                # if len(s_down):
                #     # print("s_down", s_down)
                #     uu = S[t][s_down] * 1.1 * 0.5 + s_strike[s_down] * 0.5  # 下修转股价
                #     s_strike[s_down] = np.maximum(uu, S[t][s_down])
                # s_true_index = np.argwhere(price_rate > 1.3)
                # if len(s_true_index):
                #     for ii in s_true_index:
                #         # print(ii, t, "ii")
                #         if ii[0] not in arr:
                #             # print("s", S[t], t)
                #             # print("s", S[t][ii[0]], t)
                #             s_value[ii[0]] = S[t][ii[0]] * math.exp(-self.r * (t / 245))
                #             arr.append(ii[0])
            elif (t >= 245*1) and t < 245*2:
                # pass
                # self.get_s_down_strike(S, price_rate, s_strike, t, 0.5, 0.5)
                s_strike = self.get_s_down_strike(S, price_rate, s_strike, t, 0.5, 0.5)
                self.get_s_value(arr, price_rate, s_value, t)
            elif (t >= 245*2) and t < 245*3:
                # pass
                # self.get_s_down_strike(S, price_rate, s_strike, t, 0.5, 0.5)
                s_strike = self.get_s_down_strike(S, price_rate, s_strike, t, 0.5, 0.5)
                self.get_s_value(arr, price_rate, s_value, t)
            elif (t >= 245*3) and t < 245*4:
                # pass
                # self.get_s_down_strike(S, price_rate, s_strike, t, 0.5, 0.5)
                s_strike = self.get_s_down_strike(S, price_rate, s_strike, t, 0.5, 0.5)
                self.get_s_value(arr, price_rate, s_value, t)
            elif (t >= 245*4) and t < 245*5:
                # pass
                # self.get_s_down_strike(S, price_rate, s_strike, t, 0.65, 0.9)
                s_strike = self.get_s_down_strike(S, price_rate, s_strike, t, 0.65, 0.9)
                self.get_s_value(arr, price_rate, s_value, t)
            elif (t >= 245*5) and t < 245*6:
                # pass
                # self.get_s_down_strike(S, price_rate, s_strike, t, 0.65, 0.9)
                s_strike = self.get_s_down_strike(S, price_rate, s_strike, t, 0.65, 0.9)
                self.get_s_value(arr, price_rate, s_value, t)
            if t == 1:
                pass
                # print("aa", t, S[t])
        # print("arr", arr)
        # print("s_value", s_value)
        # print("s_value2", s_value[arr])
        # print("s_strike", s_strike)
        S = np.delete(S, arr, axis=1)
        s_strike = np.delete(s_strike, arr)
        # print("s_strike", s_strike)
        # h = np.maximum(S - self.strike, 0)
        # print("h", h)
        h = np.maximum(S - s_strike, 0)
        # print("s", S[0:5])
        # print("h", h[0:5])
        print("h", h.shape)
        V = np.copy(h)
        """range(start,end,step)注：start 指的是计数起始值，默认是 0；stop 指的是计数结束值，但不包括 stop ；step 是步长，
        默认为 1，不可以为 0 。range() 方法生成一段左闭右开的整数范围。在使用range()函数时，如果只有一个参数，就表示end，
        从0开始；如果有两个参数，就表示start和end；如果有三个参数，最后一个就表示步长。它接收的参数必须是整数，可以是负数，
        但不能是浮点数等其它类型。
    """
        for t in range(self.M-1, 0, -1):  # 49-1, 这里最大49是为了后面V[t + 1]
            """np.polyfit(x,y,deg=1) x，y就可以理解为x和y坐标了，deg就是阶数，阶数是自变量的最高次方.
            制定deg参数为2的话即返回抛物线的参数[L,k,b]=Lx**2+kx**1+b"""
            reg = np.polyfit(S[t], V[t+1] * discount, 5)
            continuation_value = np.polyval(reg, S[t])  # 上面是依据x拟合出系数，这里是依据x算出y. 计算当前时刻持有价值
            # continuation_s[t] = continuation_value
            if t == self.M:
            # if t == 1:
                from sklearn.metrics import r2_score
                R_square = r2_score(V[t] * dt, continuation_value)  # 第一个为实际y，第二为预测y
                # R_square = r2_score(V[t - 1] * dt, C)  # 第一个为实际y，第二为预测y
                print(t, 'R_square: {:.2f}'.format(R_square))  # 得到R平方接近1，该模型拟合度较强，可以用来预测
            """如果c中t时刻（49）持有价值》h中t时刻立即变现价值，则取t+1时刻（50）期权值折现，否则取当前期权值"""
            V[t] = np.where(continuation_value > h[t], V[t+1] * discount, h[t])  # 对应元素比较，如果true，取V[t + 1] * df中对应元素
        # print("V[1]", V[1])
        V[1] += 110 * np.exp(-self.r * 6)  # 改折现
        # print("V[1]2", V[1])
        # print("s_value2", s_value)
        # print("s_value2", )
        # C0 = discount * np.mean(V[1])  # 折现到v0时刻
        C0 = discount * np.mean(list(V[1]) + list(s_value[arr]))  # 折现到v0时刻
        # C0 = np.mean(V[1]*discount)
        end = time()
        print(' %.6f seconds' % (end - start))
        print(C0)
        gg = ""
        if gg:
            db_path = r"D:\myzq\axzq\T0002\stock_load\thesis\kzz\monte.db"
            with sqlite3.connect(db_path) as conn:
                df = pd.DataFrame(np.round(S, 2))
                df2 = pd.DataFrame(np.round(h, 2))
                df3 = pd.DataFrame(np.round(continuation_s, 2))
                df4 = pd.DataFrame(np.round(V, 2))
                df.to_sql("s", con=conn, if_exists='replace', index=False)
                df2.to_sql("h", con=conn, if_exists='replace', index=False)
                df3.to_sql("continuation_s", con=conn, if_exists='replace', index=False)
                df4.to_sql("v", con=conn, if_exists='replace', index=False)
        # return C0

    # 下修 # price_rate2 达到下修的标准，相对于转股价下跌的幅度, down_p 下修概率
    def get_s_down_strike(self, S, price_rate, s_strike, t, price_rate2, down_p):
        s_strike = np.where(price_rate < price_rate2, np.maximum(S[t] * 1.1 * down_p + s_strike * (1 - down_p), S[t]),
                            s_strike)
        return s_strike

        # s_down = np.argwhere(price_rate < price_rate2)  # 找出a中小于0.3的所有元素的下标.下修转股价
        # # print("s_down", s_down)
        # if len(s_down):
        #     # print("s_down", s_down)
        #     uu = S[t][s_down] * 1.1 * down_p + s_strike[s_down] * (1-down_p)  # 下修转股价
        #     s_strike[s_down] = np.maximum(uu, S[t][s_down])

    # 强赎
    def get_s_value(self, arr, price_rate, s_value, t):
        s_true_index = np.argwhere(price_rate > 1.3)
        if len(s_true_index):
            for ii in s_true_index:
                # print(ii, t, "ii")
                if ii[0] not in arr:
                    # print("s", S[t], t)
                    # print("s", S[t][ii[0]], t)
                    s_value[ii[0]] = price_rate[ii[0]]*100 * math.exp(-self.r * (t / 245))
                    arr.append(ii[0])


# american2 = AmericanOptionsLSMC2(100., 120., 1., 6*245, 0.03, 0.35, 10)
# american2 = AmericanOptionsLSMC2(100., 120., 1., 5, 0.03, 0.35, 26)
# american2.gmb_mcs_amer()
# american2.get_price()
"""call是看涨期权，put是看跌期权，最简单的理解下，未来股价看涨，你应该买call，未来股价看跌你应该买put。"""
# gmb_mcs_amer(105, option='call')  # 1.4620120629034186
# gmb_mcs_amer(110, option='put')  # 9.918961211536654


# 获取可转债股价
class KzzStockPrice(object):
    def __init__(self):
        pass

    # 输入代码和时间段获取股票k线数据. adjustflag(1：后复权， 2：前复权，3：不复权）
    def history_k_data(self, code="sh.000001", start_date='2021-07-25', end_date='2021-07-31', frequency="d",
                       adjustflag="2", col="all"):
        import baostock as bs
        lg = bs.login()
        if (not code.startswith("sh.")) and (not code.startswith("sz.")):
            # print("ghjppp", code)
            if code.startswith("SH.") or code.startswith("SZ."):
                code = code.lower()
            else:
                # print("ghjppkkkkkkp", code)
                code = self.add_sh(code, big="baostock")
        # 显示登陆返回信息
        # print('login respond error_code:' + lg.error_code)
        # print('login respond  error_msg:' + lg.error_msg)
        # “分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
        # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
        # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
        if col == "all":
            str_col = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
        if col == "only_up_rate":
            str_col = "date,close,pctChg"
        # print("ghj", code)
        rs = bs.query_history_k_data_plus(code, str_col, start_date, end_date, frequency, adjustflag)
        # print('query_history_k_data_plus respond error_code:' + rs.error_code)
        # print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        bs.logout()
        if col == "all":
            result = pd.DataFrame(data_list, columns=rs.fields)
            # adjustflag	指数没有复权?	不复权、前复权、后复权
            # turn	换手率	精度：小数点后6位；单位：%
            # tradestatus	交易状态	1：正常交易 0：停牌
            # pctChg	涨跌幅（百分比）	精度：小数点后6位
            # peTTM	滚动市盈率	精度：小数点后6位
            # psTTM	滚动市销率	精度：小数点后6位
            # pcfNcfTTM	滚动市现率	精度：小数点后6位
            # pbMRQ	市净率	精度：小数点后6位
            # isST	是否ST	1是，0否
            # result.to_csv("D:\\history_A_stock_k_data.csv", index=False)
        if col == "only_up_rate":
            import sqlite3
            db_path = r"D:\myzq\axzq\T0002\stock_load\thesis\kzz\monte.db"  # 数据库路径
            result = pd.DataFrame(data_list, columns=rs.fields, dtype='float').round(3)
            # 对数收益率 = log(收盘价/前一个收盘价)
            result['log_'] = (np.log(result['close'] / result['close'].shift(periods=1, axis=0))*100).round(3)
            # print(result[1:])
            with sqlite3.connect(db_path) as conn:
                result[1:].to_sql(code, con=conn, if_exists='replace', index=False)
            # print(data_list[-5:])
            # return data_list

    # 静态函数和该类没有直接的交互，只是寄存在了该类的命名空间中
    @staticmethod
    def add_sh(code, big=""):  # big="baostock"加(sh. or sz.)code加(sh or sz) or (SZ or SH)
        if big == "":
            if code.startswith("0") or code.startswith("3") or code.startswith("2"):
                code = "sz" + code
            elif code.startswith("5") or code.startswith("6") or code.startswith("9"):
                code = "sh" + code
            else:
                print("err1", code)
        elif big == "baostock":
            if code.startswith("0") or code.startswith("3") or code.startswith("2"):
                code = "sz." + code
            elif code.startswith("5") or code.startswith("6") or code.startswith("9"):
                code = "sh." + code
            else:
                print("err2", code)
        else:
            if code.startswith("0") or code.startswith("3") or code.startswith("2"):
                code = "SZ" + code
            elif code.startswith("5") or code.startswith("6") or code.startswith("9"):
                code = "SH" + code
            else:
                print("err3", code)
        return code

# kzs = KzzStockPrice()
# kzs.history_k_data(start_date='2021-05-26', end_date='2022-05-27', col="only_up_rate")  # 取上证综指
# kzs.history_k_data(code="sh.603233", start_date='2021-05-26', end_date='2022-05-27', col="only_up_rate")
# kzs.history_k_data(code="sz.002624", start_date='2021-05-25', end_date='2022-05-27', col="only_up_rate")


# 波动率
class KzzStockArch(object):
    def __init__(self):
        pass

    # 画热力图
    @staticmethod
    def thermodynamicOrder(df, ar=4, ma=1):
        import itertools
        from statsmodels.tsa.arima.model import ARIMA
        import seaborn as sns
        results_aic = pd.DataFrame(index=['AR{}'.format(i) for i in range(0, ar + 1)],
                                   columns=['MA{}'.format(i) for i in range(0, ma + 1)])
        # print(results_aic)
        """itertools.product(*iterables[, repeat]).以元组的形式，根据输入的可遍历对象生成笛卡尔积，与嵌套的for循环类似。
        repeat指定重复生成序列的次数。
        a = (1, 2, 3)
        b = ('A', 'B', 'C')
        c = itertools.product(a,b)
        for elem in c:
            print elem

        (1, 'A')
        (1, 'B')
        (1, 'C')
        (2, 'A')
        (2, 'B')
        (2, 'C')
        (3, 'A')
        (3, 'B')
        (3, 'C')
        """
        for p, q in itertools.product(range(0, ar + 1), range(0, ma + 1)):
            # print(p, q)
            if p == 0 and q == 0:
                results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
                # print(results_aic)
                continue
            try:
                # results = arima_model.ARMA(df, (p, q)).fit()
                results = ARIMA(df, order=(p, 0, q)).fit()
                # 返回不同pq下的model的BIC值
                results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.aic
                print(results.aic)
            except:
                continue
        results_aic = results_aic[results_aic.columns].astype(float)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax = sns.heatmap(results_aic,
                         # mask=results_aic.isnull(),
                         ax=ax,
                         annot=True,  # 将数字显示在热力图上
                         fmt='.2f',
                         )
        ax.set_title('AIC')
        plt.show()

    # 画涨幅中位数图和正太分布对比
    @staticmethod
    def quantile_plot(x, **kwargs):
        import scipy.stats as stats
        """红色线条表示正态分布，蓝色线条表示样本数据，蓝色越接近红色参考线，说明越符合预期分布（这是是正态分布）
q-q 图是通过比较数据和正态分布的分位数是否相等来判断数据是不是符合正态分布
scipy.stats.probplot(x, sparams=(), dist='norm', fit=True, plot=None, rvalue=False)
x：array_like样本/响应数据
sparams：tuple, 可选参数
Distribution-specific形状参数(形状参数加上位置和比例)。
dist：分发或分发函数名称。对于正常概率图，默认值为‘norm’。
fit：如果为True(默认值)，则将least-squares回归(best-fit)行拟合到样本数据。
plot：如果给定，则绘制分位数。如果给出并且scipy.stats.fit为真，也绘制最小二乘拟合。
"""
        res = stats.probplot(x, fit=True, plot=plt)
        _slope, _int, _r = res[-1]
        ax = plt.gca()
        ax.get_lines()[0].set_marker('s')
        ax.get_lines()[0].set_markerfacecolor('r')
        ax.get_lines()[0].set_markersize(13.0)
        ax.get_children()[-2].set_fontsize(22.)
        txkw = dict(size=14, fontweight='demi', color='r')
        """相关性=R-squared误差取值范围为0到1，这个值越接近1说明模型的拟合度越好。"""
        r2_tx = "r^2 = {:.2%}\nslope = {:.4f}".format(_r, _slope)
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        ax.text(0.5 * xmax, .8 * ymin, r2_tx, **txkw)
        plt.show()
        # plt.savefig('figname.png')
        return

    def adf_test(self, tab):
        import sqlite3
        db_path = r"D:\myzq\axzq\T0002\stock_load\thesis\kzz\monte.db"  # 数据库路径
        with sqlite3.connect(db_path) as conn:
            # data = pd.read_sql("select close from 'sh.603233'", conn)
            # data = pd.read_sql("select pctChg from 'sh.603233'", conn)
            data = pd.read_sql(r"select log_ from '{}' ORDER BY date DESC".format(tab), conn)
            # print(data.head())
            dd = ""  # 计算年华对数和普通波动率
            if dd == "variance":
                print("方差：", data.var())
                print("标准差: ", data.std())
                print(data.shape[0])
                print(data.shape[0]**0.5)
                print("年华波动率: ", data.std()*(data.shape[0]**0.5))

            dd = ""  #
            if dd == "garch":
                import arch
                """赤池信息准则（Akaike Information Criterion，AIC）和贝叶斯信息准则（Bayesian Information Criterion，BIC）
                选择最佳模型时，通常选择AIC最小的模型.const coef:10.2779 y截距，即b值。std err，反映系数的准确度，越低，准确度越高。
                P>|t|:p值，判断回归参数是否有统计学意义.
                Confidence Interval：置信区间，表示我们的系数可能范围（可能性为 95%）
                在模型中，增加多个变量，即使事实上无关的变量，也会小幅度条R平方的值，当时其是无意义，所有我们调整了下，降低R平方的值。
                简单地说就是，用r square的时候，不断添加变量能让模型的效果提升，而这种提升是虚假的。利用adjusted r square，
                能对添加的非显著变量给出惩罚，也就是说随意添加一个变量不一定能让模型拟合度上升
                标准误 （Standard Error, SE）"""
                # am = arch.arch_model(data, vol='GARCH')
                # am = arch.arch_model(data, mean='AR', vol='GARCH', dist='gaussian',)
                # am = arch.arch_model(data, p=1, q=1, o=0, power=2.0, vol='Garch', dist='StudentsT')
                # am = arch.arch_model(data, vol='Garch', dist='StudentsT')
                # am = am.fit()
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
                am = arch.arch_model(train, mean='AR', lags=1, vol='ARCH',)
                # am = arch.arch_model(train, mean='AR', lags=1, vol='GARCH',)
                res = am.fit()
                print(res.summary())

            dd = "ljungbox"  # ljungbox检验
            if dd == "ljungbox":
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
                ljungbox_result = acorr_ljungbox(data['log_'], return_df=True, boxpierce=True,)  # 返回统计量和p值，lags为检验的延迟数
                # ljungbox_result = acorr_ljungbox(data, return_df=True, auto_lag=True)  # 返回统计量和p值，lags为检验的延迟数
                print(ljungbox_result)

                # import statsmodels.api as sm
                # # 举一个非白噪声例子结束
                # data = sm.datasets.sunspots.load_pandas().data
                # data = data.set_index('YEAR')
                # res = acorr_ljungbox(data.SUNACTIVITY, lags=[6, 12, 24], boxpierce=True, return_df=True)
                # print(res)
                # data.plot(figsize=(12, 4))
                # plt.show()

            dd = ""  # arima
            if dd == "arima":
                from statsmodels.tsa.arima.model import ARIMA
                """三个阶参数 (p,d,q) 指定：
                AR(p)：说明数据的增长/下降模式
                I (d): 增长/下降的变化率被考虑在内
                MA (q)：考虑时间点之间的噪声 """
                model = ARIMA(data, order=(2, 0, 0)).fit()
                print(model.summary())    # 生成一份模型报告返回预测结果， 标准误差， 和置信区间
                # print(model.forecast(5))   # 为未来5天进行预测，

            dd = ""  # arima参数
            if dd == "arima_p_q":
                from pmdarima.arima import auto_arima
                """
                auto_arima可以帮助我们自动确定 A R I M A ( p , d , q ) ( P , D , Q ) m
                data_low：训练集
                start_p：p参数迭代的初始值
                max_p：p参数迭代的最大值
                seasonal：季节性
                trace：平滑
                stepwise：显示运行过程,stepwise参数，默认值就是True，表示用stepwise algorithm来选择最佳的参数组合，
                会比计算所有的参数组合要快很多，而且几乎不会过拟合，当然也有可能忽略了最优的组合参数。
                所以如果你想让模型自动计算所有的参数组合，然后选择最优的，可以将stepwise设为False
                information_criterion='aic',
                test='adf',       # use adftest to find optimal 'd'
                m=1,              # frequency of series
                d=None,           # let model determine 'd'
                （1）趋势参数
                      p pp：趋势自回归阶数。
                      d dd：趋势差分阶数。
                      q qq：趋势移动平均阶数。
                （2）季节性参数
                      P PP：季节性自回归阶数。
                      D DD：季节性差分阶数。
                      Q QQ：季节性移动平均阶数。
                      m mm：单个季节期间的时间步数。

                与非季节性模型的区别在于，季节性模型都是以m为固定周期来做计算的，比如D就是季节性差分，
                是用当前值减去上一个季节周期的值，P和Q和非季节性的p,q的区别也是在于前者是以季节窗口为单位，而后者是连续时间的。
                上节介绍的auto arima的代码中，seasonal参数设为了false，构建季节性模型的时候，把该参数置为True，
                然后对应的P，D，Q,m参数即可，代码如下：
                # Seasonal - fit stepwise auto-ARIMA
                smodel = pm.auto_arima(data, start_p=1, start_q=1,test='adf',
                                         max_p=3, max_q=3, m=12,
                                         start_P=0, seasonal=True,
                                         d=None, D=1, trace=True,
                                         error_action='ignore',
                                         suppress_warnings=True,
                                         stepwise=True)
                 无季节模型参数
              auto_arima(df.value, start_p=1, start_q=1,
                              information_criterion='aic',
                              test='adf',       # use adftest to find optimal 'd'
                              max_p=3, max_q=3, # maximum p and q
                              m=1,              # frequency of series
                              d=None,           # let model determine 'd'
                              seasonal=False,   # No Seasonality
                              start_P=0,
                              D=0,
                              trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)"""
                # model1 = auto_arima(data, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True,
                #                     d=None, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True,
                #                     )
                model1 = auto_arima(data, start_p=1, start_q=1,
                                    information_criterion='aic',
                                    test='adf',  # use adftest to find optimal 'd'
                                    max_p=3, max_q=3,  # maximum p and q
                                    m=1,  # # 季节性周期长度，当m=1时则不考虑季节性
                                    d=None,  # let model determine 'd'
                                    seasonal=False,  # No Seasonality
                                    start_P=0, D=0, trace=True,
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=True  # stepwise为False则不进行完全组合遍历
                                    )
                model1.fit(data)
                print(model1.summary())  # 生成一份模型报告返回预测结果， 标准误差， 和置信区间
                print(model1.predict(n_periods=12))   # 为未来5天进行预测，
                # import joblib
                # joblib.dump(model1, r'D:\myzq\axzq\T0002\stock_load\thesis\kzz\model_save.pkl')
                # clf = joblib.load("train_model.m")  # 模型从本地调回
                # 所以可以建立ARIMA 模型，ARIMA(3,1,0)

            dd = ""  # 画热力图
            if dd == "thermodynamicOrder":
                self.thermodynamicOrder(data)
                # self.thermodynamicOrder(data['log_'])

            dd = ""  # 计算自相关系数, 偏自相关系数,白噪声检验:Ljung-Box检验
            if dd == "acf":
                """检验residual是否是白噪声，如果是则无法预测.说白了就是因为平稳序列具有比较好的性质，具有时间平移不变性，
                具体的说法是弱平稳时间序列均值为常数，协方差只与lag有关。其容易建立模型来预测未来，不平稳的不好搞。"""
                from statsmodels.tsa import stattools
                """计算自相关系数，这里设置滞后项为5期,默认是40期滞后.nlags指定了10，
                所以就会最多对比当前期与10期之前的时间序列数据，这里10期就是10个交易日。
                时间序列最常用来剔除周期性因素的方法当属差分了，它主要是对等周期间隔的数据进行线性求减.
                自相关图和偏自相关图中浅蓝色条形的边界为 |2|/根号t ，T为序列的长度。由于随机扰动的存在，
                自相关系数并不严格等于0，我们期望在95%的置信度下，相关系数均在 |2|/根号t之间。
                如果一个序列中有较多自相关系数的值在边界之外，那么该序列很可能不是白噪声序列。
                图中自相关系数均在边界之内，为白噪声序列"""
                # acf = stattools.acf(data, nlags=5, fft=False)
                # 计算偏自相关系数
                # pacf = stattools.pacf(data, nlags=5)
                # print(f'自相关系数为：{acf};\n偏自相关系数为：{pacf}')
                from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
                """自相关图, 偏自相关图横坐标表示延迟阶数，纵坐标表示自相关系数"""
                # plot_acf(data, lags=15)
                # plot_pacf(data, lags=15)
                # plt.tight_layout()
                # plt.show()
                order = (1, 0)
                # model = stattools.ARMA(data, order).fit()
                import statsmodels.api as sm  # 统计相关的库
                model = sm.tsa.ARMA(data, order).fit()
                # print(model.fittedvalues)
                at = data['log_'] - model.fittedvalues
                # print(at.head())
                at2 = np.square(at)
                plot_acf(at, lags=95)
                plot_pacf(at, lags=95)
                plt.tight_layout()
                plt.show()
                m = 95  # 我们检验25个自相关系数
                acf, q, p = sm.tsa.acf(at2, nlags=m, qstat=True)  # 计算自相关系数 及p-value
                out = np.c_[range(1, 96), acf[1:], q, p]
                output = pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
                output = output.set_index('lag')
                # print(list(output["P-value"]))
                print(output[output["P-value"] < 0.05])

            dd = ""  # 画收益图
            if dd == "plt":
                import seaborn as sns
                # rs = (np.log(df / df.shift(1))).dropna()
                data.plot(figsize=(12, 5))
                # plt.title('指数日对数收益率', size=15)
                sns.displot(data, color='blue')  # 密度图
                plt.show()

            dd = ""
            if dd == "plt2":  # 画涨幅中位数图和正太分布对比
                self.quantile_plot(data['log_'])

            dd = ""  # 时间序列平稳性单位根检验
            if dd == "adf":
                from statsmodels.tsa.stattools import adfuller
                """
                (-0.04391111656553232, 0.9547464774274733, 10, 22, {'1%': -3.769732625845229, '5%': -3.005425537190083, '10%': -2.6425009917355373}, 291.54354258641223)
                第一个是adt检验的结果，简称为T值，表示t统计量。
                第二个简称为p值，表示t统计量对应的概率值。
                第三个表示延迟。
                第四个表示测试的次数。
                第五个是配合第一个一起看的，是在99%，95%，90%置信区间下的临界的ADF检验的值。
                第一点，1%、%5、%10不同程度拒绝原假设的统计值和ADF Test result的比较，ADF Test result同时小于1%、5%、10%即说明非常好地拒绝该假设。本数据中，adf结果为-0.04391111656553232，大于三个level的统计值，接收假设，即存在单位根。
                第二点，p值要求小于给定的显著水平，p值要小于0.05，等于0是最好的。本数据中，P-value 为 0.9547464774274733,大于三个level，接受假设，即存在单位根。
                ADF检验的原假设是存在单位根，只要这个统计值是小于1%水平下的数字就可以极显著的拒绝原假设，认为数据平稳。注意，ADF值一般是负的，也有正的，但是它只有小于1%水平下的才能认为是及其显著的拒绝原假设。
                对于ADF结果在1% 以上 5%以下的结果，也不能说不平稳，关键看检验要求是什么样子的。
                adfuller(
                    x,
                    maxlag=None,
                    regression="c",
                    autolag="AIC",
                    store=False,
                    regresults=False,
                )
                (-11.450545565916247, 5.8728395141112274e-21, 1, 242, {'1%': -3.457664132155201, '5%': -2.8735585105960224, '10%': -2.5731749894132916}, 1150.6699808503472)
                """
                result = adfuller(data, regresults=False,)
                # print(result)
                print("\nresult is\n{}".format(result))
                result_fromat = pd.Series(result[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])
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


# ksa = KzzStockArch()
# ksa.adf_test(tab='sh.000001')
# ksa.adf_test(tab='sh.603233')
# ksa.adf_test(tab='sz.002624')


# 获取可转债行情价并计算债券价值,隐含波动率
class KzzPrice(object):
    def __init__(self):
        pass

    @staticmethod  # 输入代码和时间段获取k线数据.
    def ak_bond_price(code):
        import akshare as ak
        d = 'bond_zh_hs_cov_daily'
        if d == "bond_zh_hs_cov_daily":
            bond__daily = ak.bond_zh_hs_cov_daily(symbol=code)
            print(bond__daily)
            # 对数收益率 = log(收盘价/前一个收盘价)
            bond__daily['log_'] = (np.log(bond__daily['close'] / bond__daily['close'].shift(periods=1, axis=0)) * 100).round(3)
            bond__daily['up_change'] = ((bond__daily['close'] - bond__daily['close'].shift(periods=1, axis=0))/bond__daily['close'].shift(periods=1, axis=0) * 100).round(3)
            bond__daily['bound_value'] = ""
            bond__daily['interval'] = ""
            bond__daily['implication_option'] = ""
            bond__daily['implication_volatility'] = ""
            # print(result[1:])
        d = 'sq_lite'
        if d == "sq_lite":
            import sqlite3
            db_path = r"D:\myzq\axzq\T0002\stock_load\thesis\kzz\monte.db"  # 数据库路径
            with sqlite3.connect(db_path) as conn:
                bond__daily.to_sql(code, con=conn, if_exists='replace', index=True)
                # bond__daily[1:].to_sql(code, con=conn, if_exists='replace', index=True)
                # conn.close()

    # 静态函数和该类没有直接的交互，只是寄存在了该类的命名空间中
    @staticmethod
    def bond_value(code, start, end, interest='', aaa=''):
        d = 'bond_value'
        if d == "bond_value":
            import sqlite3
            import datetime
            import time
            import math
            # interest = interest[::-1]
            # print(interest)
            db_path = r"D:\myzq\axzq\T0002\stock_load\thesis\kzz\monte.db"  # 数据库路径
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                data = pd.read_sql(r"select date,close from '{}'".format(code), conn, parse_dates=['date'])
                # data = pd.read_sql(r"select date from '{}' ORDER BY date DESC".format(code), conn)
                # print(data.head())
                # print(data['date'][0])
                # d1 = datetime.datetime(data[0])  # 第一个日期
                d2 = datetime.datetime.strptime(end, '%Y-%m-%d')
                time_start1 = time.clock()  # 记录开始时间
                ii = -1
                for d1 in data['date']:
                    interval = (d2 - d1).days/365  # 两日期差距
                    ii += 1
                    # print("interval:", data['close'][ii])
                    d = 'down'
                    if d == "down":
                        y_int = int(interval)  # 向下取整
                        c = 0
                        if y_int == 4:
                            c = 1
                        elif y_int == 3:
                            c = 2
                        elif y_int == 2:
                            c = 3
                        elif y_int == 1:
                            c = 4
                        elif y_int == 0:
                            c = 5
                        """
                        interval=5.几时, y_int=5  c=0
                             a=y_int-i  interval - a      b = i + c  interest[b]     aaa[b]
                                                                     interest[6]=110
                        i=5>>a=5-5=0>>  interval - 0=5. >>b = 5 + 0  interest[5]=2 > aaa[5]=3.327
                        i=4>>a=5-4=1>>  interval - 1=4. >>b = 4 + 0  interest[4]=1.8 > aaa[4]=3.283
                        i=3>>a=5-3=2>>  interval - 2=3. >>b = 3 + 0  interest[3]=1.5 > aaa[3]=3.11
                        i=2>>a=5-2=3>>  interval - 3=2. >>b = 2 + 0  interest[2]=1.0 > aaa[2]=2.937
                        i=1>>a=5-1=4>>  interval - 4=1. >>b = 1 + 0  interest[1]=0.6 > aaa[1]=2.697
                        i=0>>a=5-0=5>>  interval - 5=0. >>b = 0 + 0  interest[0]=0.3 > aaa[0]=2.456

                        interval=4.几时, y_int=4  c=1
                             a=y_int-i  interval - a      b = i + c interest[b]       aaa[b]
                        i=4>>a=4-4=0>>  interval - 0=4. >>b = 4 + 1 interest[5]=110 > aaa[5]=3.327
                        i=3>>a=4-3=1>>  interval - 1=3. >>b = 3 + 1 interest[4]=1.8 > aaa[4]=3.283
                        i=2>>a=4-2=2>>  interval - 2=2. >>b = 2 + 1 interest[3]=1.5 > aaa[3]=3.11
                        i=1>>a=4-1=3>>  interval - 3=1. >>b = 1 + 1 interest[2]=1.0 > aaa[2]=2.937
                        i=0>>a=4-0=4>>  interval - 4=0. >>b = 0 + 1 interest[1]=0.6 > aaa[1]=2.697

                        interval=3.几时, y_int=3  c=2
                             a=y_int-i  interval - a      b = i + c interest[b]         aaa[b]
                        i=3>>a=3-3=0>>  interval - 0=3. >>b = 3 + 2 interest[5]=110 > aaa[5]=3.327
                        i=2>>a=3-2=1>>  interval - 1=2. >>b = 2 + 2 interest[4]=1.8 > aaa[4]=3.283
                        i=1>>a=3-1=2>>  interval - 2=1. >>b = 1 + 2 interest[3]=1.5 > aaa[3]=3.11
                        i=0>>a=3-0=3>>  interval - 3=0. >>b = 0 + 2 interest[2]=1.0 > aaa[2]=2.937

                        interval=2.几时, y_int=2  c=3
                             a=y_int-i  interval - a      b = i + c interest[b]       aaa[b]
                        i=2>>a=2-2=0>>  interval - 0=2. >>b = 2 + 3 interest[5]=110 > aaa[5]=3.327
                        i=1>>a=2-1=1>>  interval - 1=1. >>b = 1 + 3 interest[4]=1.8 > aaa[4]=3.283
                        i=0>>a=2-0=2>>  interval - 2=0. >>b = 0 + 3 interest[3]=1.5 > aaa[3]=3.11

                        interval=1.几时, y_int=1  c=4
                             a=y_int-i  interval - a      b = i + c interest[b]       aaa[b]
                        i=1>>a=1-1=0>>  interval - 0=1. >>b = 1 + 4 interest[5]=110 > aaa[5]=3.327
                        i=0>>a=1-0=1>>  interval - 1=0. >>b = 0 + 4 interest[4]=1.8 > aaa[4]=3.283

                        interval=0.几时, y_int=0  c=5
                             a=y_int-i  interval - a      b = i + c interest[b]       aaa[b]
                        i=0>>a=0-0=0>>  interval - 0=0. >>b = 0 + 5 interest[5]=110 > aaa[5]=3.327
                        """
                        sum_value = 0
                        for i in range(y_int, -1, -1):
                            a = y_int-i
                            b = i + c
                            # print(a)
                            # print(a, b, interest[b], 1+aaa[b]/100, interval - a)
                            if i == y_int:
                                le = len(interest)
                                if le == 6:  # 包括最后一年利息
                                    sum_value += ((interest[b]-(interest[b]-100)*0.2)/((1 + aaa[b]/100) ** (interval - a)))
                                elif le == 7:
                                    seven = interest[b+1]-(interest[b+1]-100)*0.2+interest[b]*0.8
                                    sum_value += (seven/((1 + aaa[b]/100) ** (interval - a)))
                            else:
                                sum_value += ((interest[b]*0.8) / ((1 + aaa[b]/100) ** (interval - a)))
                        # print("sum_value:", sum_value)
                        d = ''
                        if d == "sq_lite":
                            sql_update = r"update {} set bound_value=?,interval=?,implication_option=? where date=?".format(code)
                            # print("sum_value:", round(sum_value, 3), d1, sql_update)
                            ar = (round(sum_value, 3), round(interval, 6), round(data['close'][ii]-sum_value, 3), str(d1))
                            cur.execute(sql_update, ar)
                time_end1 = time.clock()  # 记录结束时间
                time_sum1 = time_end1 - time_start1  # 计算的时间差为程序的执行时间，单位为秒/s
                print("time_sum1", time_sum1)
                cur.close()

kzs = KzzPrice()
# kzs.ak_bond_price(code="sh113605")  #
interest1 = [0.30, 0.60, 1.00, 1.50, 1.80, 110]  # 包含最后一期利息
# interest=[0.30, 0.60, 1.00, 1.50, 1.80, 2.0, 110]
kzs.bond_value(code="sh113605", start="2020-10-22", end="2026-10-22", interest=interest1,
               aaa=[2.456, 2.697, 2.937, 3.11, 3.283, 3.327])




