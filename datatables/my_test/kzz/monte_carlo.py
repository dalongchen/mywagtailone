import numpy as np
import pandas as pd
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
        # import matplotlib.pyplot as plt
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


american2 = AmericanOptionsLSMC2(100., 120., 1., 6*245, 0.03, 0.35, 10)
# american2 = AmericanOptionsLSMC2(100., 120., 1., 5, 0.03, 0.35, 26)
# american2.get_price()
american2.gmb_mcs_amer()
"""call是看涨期权，put是看跌期权，最简单的理解下，未来股价看涨，你应该买call，未来股价看跌你应该买put。"""
# gmb_mcs_amer(105, option='call')  # 1.4620120629034186
# gmb_mcs_amer(110, option='put')  # 9.918961211536654

