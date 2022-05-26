import numpy as np


class AmericanOptionsLSMC(object):
    """ Class for American put options pricing using LSMC
    S0 : float : initial stock/index level
    strike : float : strike price
    T : float : time to maturity (in year fractions)
    M : int : grid or granularity for time (in number of total points)
    r : float : constant risk-free short rate
    div :    float : dividend yield
    sigma :  float : volatility factor in diffusion term
    simulations: int: need to be even number
    """

    def __init__(self, S0, strike, T, M, r, div, sigma, simulations):
        try:
            self.S0 = float(S0)
            self.strike = float(strike)
            assert T > 0
            self.T = float(T)
            assert M > 0
            self.M = int(M)
            assert r >= 0
            self.r = float(r)
            assert div >= 0
            self.div = float(div)
            assert sigma > 0
            self.sigma = float(sigma)
            assert simulations > 0
            self.simulations = int(simulations)
        except ValueError:
            print('Error passing Put Options parameters')

        if S0 < 0 or strike < 0 or T <= 0 or r < 0 or div < 0 or sigma < 0:
            raise ValueError('Error: Negative inputs not allowed')

        self.time_unit = self.T / float(self.M)
        self.discount = np.exp(-self.r * self.time_unit)

    @property
    def MCprice_matrix(self, seed=123):
        """ Returns stock price path matrix with rows: time, columns: stock price path
        print(l.method_with_property) # 加了@property后，可以用调用属性的形式来调用方法,后面不需要加（）。
        print(l.method_without_property())  #没有加@property , 必须使用正常的调用方法的形式，即在后面加()"""
        np.random.seed(seed)
        # row 0 for S0, row 1 to M+1 for MC stock price hence in totall M+1 rows
        MCprice_matrix = np.zeros((self.M + 1, self.simulations), dtype=np.float64)
        MCprice_matrix[0, :] = self.S0
        for t in range(1, self.M + 1):
            # //取整除 - 返回商的整数部分（向下取整） 9//2=4, -9//2 = -5
            brownian = np.random.standard_normal(self.simulations // 2)
            """np.concatenate((a1,a2,…), axis=0)，0表示以列拼接，能够一次完成多个数组的拼接。其中a1,a2,…是数组"""
            brownian = np.concatenate((brownian, -brownian))
            # if t == 1:
            #     print("brownian", t, brownian)
            # GBM solution
            MCprice_matrix[t, :] = (MCprice_matrix[t - 1, :]
                                    * np.exp((self.r - self.sigma ** 2 / 2.) * self.time_unit
                                             + self.sigma * brownian * np.sqrt(self.time_unit)))
        return MCprice_matrix

    @property
    def MCpayoff(self):
        """Returns the exercise value of American Put Option"""
        # payoff = np.maximum(self.strike - self.MCprice_matrix, np.zeros((self.M + 1, self.simulations), dtype=np.float64))
        # print("aa", np.zeros((self.M + 1, self.simulations)).shape)  # (51, 10000)
        # print("aa", self.MCprice_matrix-self.strike)  # (51, 10000)
        # print("aa", (self.MCprice_matrix-self.strike).shape)  # (51, 10000)
        payoff = np.maximum(self.MCprice_matrix-self.strike, np.zeros((self.M + 1, self.simulations), dtype=np.float64))
        # print("aa", payoff)  # (51, 10000)
        return payoff

    @property
    def value_vector(self):
        # print("aa", self.MCpayoff[-1, :].shape)
        value_matrix = np.zeros_like(self.MCpayoff)  # 像这个矩阵MCpayoff，值为0
        # print("aa2", value_matrix)
        # last row's cash flow is already determined to be the exercise value as there is no continuation value
        value_matrix[-1, :] = self.MCpayoff[-1, :]
        # print("aa2", value_matrix[-1, :])
        # print("aa3", value_matrix[48 + 1, :])
        # going backward -1 at a time
        # value matrix column t is the cash flow of put option a time t regardless of when it is exercise
        for t in range(self.M - 1, 0, -1):
            # print("aa2", t, value_matrix)
            regression = np.polynomial.polynomial.polyfit(self.MCprice_matrix[t, :],
                                                          value_matrix[t + 1, :] * self.discount, 5)
            continuation_value = np.polyval(regression[::-1], self.MCprice_matrix[t, :])
            value_matrix[t, :] = np.where(self.MCpayoff[t, :] > continuation_value, self.MCpayoff[t, :],
                                          value_matrix[t + 1, :] * self.discount)

        return value_matrix[1, :] * self.discount

    @property
    def price(self):
        # print("rtyrt")
        return np.sum(self.value_vector) / float(self.simulations)

    @property
    def delta(self):
        diff = self.S0 * 0.01
        myCall_1 = AmericanOptionsLSMC(self.S0 + diff,
                                       self.strike, self.T, self.M,
                                       self.r, self.div, self.sigma, self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.S0 - diff,
                                       self.strike, self.T, self.M,
                                       self.r, self.div, self.sigma, self.simulations)
        return (myCall_1.price - myCall_2.price) / float(2. * diff)

    @property
    def gamma(self):
        diff = self.S0 * 0.01
        myCall_1 = AmericanOptionsLSMC(self.S0 + diff,
                                       self.strike, self.T, self.M,
                                       self.r, self.div, self.sigma, self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.S0 - diff,
                                       self.strike, self.T, self.M,
                                       self.r, self.div, self.sigma, self.simulations)
        return (myCall_1.delta - myCall_2.delta) / float(2. * diff)

    @property
    def vega(self):
        diff = self.sigma * 0.01
        myCall_1 = AmericanOptionsLSMC(self.S0,
                                       self.strike, self.T, self.M,
                                       self.r, self.div, self.sigma + diff,
                                       self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.S0,
                                       self.strike, self.T, self.M,
                                       self.r, self.div, self.sigma - diff,
                                       self.simulations)
        return (myCall_1.price - myCall_2.price) / float(2. * diff)

    @property
    def rho(self):
        diff = self.r * 0.01
        if (self.r - diff) < 0:
            myCall_1 = AmericanOptionsLSMC(self.S0,
                                           self.strike, self.T, self.M,
                                           self.r + diff, self.div, self.sigma,
                                           self.simulations)
            myCall_2 = AmericanOptionsLSMC(self.S0,
                                           self.strike, self.T, self.M,
                                           self.r, self.div, self.sigma,
                                           self.simulations)
            return (myCall_1.price - myCall_2.price) / float(diff)
        else:
            myCall_1 = AmericanOptionsLSMC(self.S0,
                                           self.strike, self.T, self.M,
                                           self.r + diff, self.div, self.sigma,
                                           self.simulations)
            myCall_2 = AmericanOptionsLSMC(self.S0,
                                           self.strike, self.T, self.M,
                                           self.r - diff, self.div, self.sigma,
                                           self.simulations)
            return (myCall_1.price - myCall_2.price) / float(2. * diff)

    @property
    def theta(self):
        diff = 1 / 252.
        myCall_1 = AmericanOptionsLSMC(self.S0,
                                       self.strike, self.T + diff, self.M,
                                       self.r, self.div, self.sigma,
                                       self.simulations)
        myCall_2 = AmericanOptionsLSMC(self.S0,
                                       self.strike, self.T - diff, self.M,
                                       self.r, self.div, self.sigma,
                                       self.simulations)
        return (myCall_2.price - myCall_1.price) / float(2. * diff)


AmericanPUT = AmericanOptionsLSMC(100., 120., 5., 50, 0.03, 0.06, 0.25, 10000)
# AmericanPUT = AmericanOptionsLSMC(36., 40., 1., 50, 0.06, 0.06, 0.2, 10000)
print('Price: ', AmericanPUT.price)
# print('Delta: ', AmericanPUT.delta)
# print('Gamma: ', AmericanPUT.gamma)
# print('Vega:  ', AmericanPUT.vega)
# print('Rho:   ', AmericanPUT.rho)
# print('Theta: ', AmericanPUT.theta)


# Price: 4.473117701771221
# Delta: -0.7112251324731934
# Gamma: 0.12615233203125087
# Vega: 12.196835824506369
# Rho: -10.0335229852333
# Theta: -1.8271728267244622


# def prices():
#     for S0 in (36., 38., 40., 42., 44.):  # initial stock price values
#         for vol in (0.2, 0.4):  # volatility values
#             for T in (1.0, 2.0):  # times-to-maturity
#                 AmericanPUT = AmericanOptionsLSMC(S0, 40., T, 50, 0.06, 0.06, vol, 1500)
#                 print("Initial price: %4.1f, Sigma: %4.2f, Expire: %2.1f --> Option Value %8.3f" % (
#                     S0, vol, T, AmericanPUT.price))


# from time import time
#
# t0 = time()
# optionValues = prices()  # calculate all values
# t1 = time()
# d1 = t1 - t0
# print("Duration in Seconds %6.3f" % d1)


"""
Initial price: 36.0, Sigma: 0.20, Expire: 1.0 --> Option Value    4.439
Initial price: 36.0, Sigma: 0.20, Expire: 2.0 --> Option Value    4.779
Initial price: 36.0, Sigma: 0.40, Expire: 1.0 --> Option Value    7.135
Initial price: 36.0, Sigma: 0.40, Expire: 2.0 --> Option Value    8.459
Initial price: 38.0, Sigma: 0.20, Expire: 1.0 --> Option Value    3.225
Initial price: 38.0, Sigma: 0.20, Expire: 2.0 --> Option Value    3.726
Initial price: 38.0, Sigma: 0.40, Expire: 1.0 --> Option Value    6.134
Initial price: 38.0, Sigma: 0.40, Expire: 2.0 --> Option Value    7.666
Initial price: 40.0, Sigma: 0.20, Expire: 1.0 --> Option Value    2.296
Initial price: 40.0, Sigma: 0.20, Expire: 2.0 --> Option Value    2.808
Initial price: 40.0, Sigma: 0.40, Expire: 1.0 --> Option Value    5.201
Initial price: 40.0, Sigma: 0.40, Expire: 2.0 --> Option Value    6.815
Initial price: 42.0, Sigma: 0.20, Expire: 1.0 --> Option Value    1.589
Initial price: 42.0, Sigma: 0.20, Expire: 2.0 --> Option Value    2.145
Initial price: 42.0, Sigma: 0.40, Expire: 1.0 --> Option Value    4.484
Initial price: 42.0, Sigma: 0.40, Expire: 2.0 --> Option Value    6.123
Initial price: 44.0, Sigma: 0.20, Expire: 1.0 --> Option Value    1.088
Initial price: 44.0, Sigma: 0.20, Expire: 2.0 --> Option Value    1.646
Initial price: 44.0, Sigma: 0.40, Expire: 1.0 --> Option Value    3.838
Initial price: 44.0, Sigma: 0.40, Expire: 2.0 --> Option Value    5.438
Duration in Seconds 43.709
"""
