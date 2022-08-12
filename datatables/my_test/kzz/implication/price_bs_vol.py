import numpy as np
import pandas as pd
import sqlite3
from scipy.stats import norm
import global_variable as gl_v


# 获取可转债行情价并计算债券价值,隐含波动率,bs期权价值
class KzzPriceOption(object):

    def __init__(self):
        pass

    @staticmethod  # 输入债券代码和时间段获取债券k线数据.
    def ak_bond_price(code, save=''):
        d = 'bond_zh_hs_cov_daily'
        if d == "bond_zh_hs_cov_daily":
            import akshare as ak
            bond__daily = ak.bond_zh_hs_cov_daily(symbol=code)
            bond__daily.index = bond__daily.index.strftime('%Y-%m-%d')
            clo = bond__daily['close']
            clo_1 = clo.shift(periods=1, axis=0)
            # 对数收益率 = log(收盘价/前一个收盘价)
            bond__daily['log_'] = (np.log(clo/clo_1) * 100).round(3)
            bond__daily['up_change'] = ((clo - clo_1)/clo_1 * 100).round(3)
            bond__daily['bound_value'] = ""
            bond__daily['interval'] = ""
            bond__daily['implication_option'] = ""
            bond__daily['implication_volatility'] = ""
            bond__daily['stock_close'] = ""
        if save == "y":
            with sqlite3.connect(gl_v.db_path) as conn:
                bond__daily.to_sql(code, con=conn, if_exists='replace', index=True)

    @staticmethod  # 计算债券价值father,end为债券最后退市时间
    def bond_value(code, end, interest='', r_y='', stock_code='', save=''):
        import datetime
        import time
        with sqlite3.connect(gl_v.db_path) as conn:
            cur = conn.cursor()
            # 获取债券收盘价
            # data_b = pd.read_sql(r"select date,close from '{}'".format(code), conn)
            data_b = pd.read_sql(r"select date,close from '{}'".format(code), conn, parse_dates=['date'])
            # 获取正股收盘价
            se_sql = r"select date,close from '{}'".format(stock_code)
            # data_s = pd.read_sql(se_sql, conn)
            data_s = pd.read_sql(se_sql, conn, parse_dates=['date'])
            # print(data_b.head())
            d_end = datetime.datetime.strptime(end, '%Y-%m-%d')
            time_start1 = time.clock()  # 记录开始时间
            ii = -1
            for d_bound in data_b['date']:
                arr1 = data_s[data_s['date'] == d_bound]  # 获取正股收盘价
                st_close = arr1['close'].values
                if st_close:
                    stock_close = st_close[0]
                else:
                    stock_close = ''
                    print("error", d_bound)
                ii += 1
                interval = (d_end - d_bound).days/365  # 两日期差距
                sum_value = KzzPriceOption.get_net_bound_value(interest, interval, r_y)  # 计算债值
                if save == "y":
                    # sql_update = r"update {} set bound_value=?,interval=?,option=? where date=?".format(code)
                    sql_update = r"update {} set bound_value=?,interval=?,implication_option=?,stock_close=? where date=?".format(code)
                    ar = (round(sum_value, 3), round(interval, 6), round(data_b['close'][ii]-sum_value, 3), stock_close, str(d_bound)[:10])
                    cur.execute(sql_update, ar)

            time_end1 = time.clock()  # 记录结束时间
            time_sum1 = time_end1 - time_start1  # 计算的时间差为程序的执行时间，单位为秒/s
            print("time_sum1", time_sum1)
            cur.close()

    @staticmethod  # 计算债值son
    def get_net_bound_value(interest, interval, r_y):
        y_int = int(interval)  # 向下取整
        c = 0  # 可转债剩余时间取对应利息，年
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
        sum_value = 0  # 循环计算债值
        for i in range(y_int, -1, -1):  # 5,4,3,2,1,0
            a = y_int - i  # 0,1,2,3,4,5
            b = i + c  # 5+0 or 5+1 or 5+2 or 5+3 or 5+4 or 5+5
            # 4+0 or 4+1 or 4+2 or 4+3 or 4+4 or 4+5
            inter_b = interest[b]
            in_a = interval - a  # 5., 4., 3., 2., 1., 0.
            if i == y_int:
                le = len(interest)
                if le == 6:  # 包括最后一年利息
                    sum_value += ((inter_b - (inter_b - 100) * 0.2) / ((1 + r_y[b] / 100) ** in_a))
                    # sum_value += ((interest[b]-(interest[b]-100)*0.2)/((1 + aaa[b]/100) ** (interval - a)))
                elif le == 7:  # not包括最后一年利息
                    seven = interest[b + 1] - (interest[b + 1] - 100) * 0.2 + inter_b * 0.8
                    sum_value += (seven / ((1 + r_y[b] / 100) ** in_a))
            else:
                sum_value += ((inter_b * 0.8) / ((1 + r_y[b] / 100) ** in_a))
        return sum_value

    @staticmethod  # B S M期权定价公式
    def bsm_option(S, K, T, r, volatility):
        """
        S:初始价格  标的证券的当前价格
        K:执行价格  行权价为
        T:到期时间
        r:无风险利率
        volatility: 年波动率
        stats.norm.cdf(α,均值,方差)：累积概率密度函数"""
        vol_y = volatility * np.sqrt(T)  # 年波动率=每个交易日的波动率*\sqrt{365}?
        d1 = (np.log(S / K) + (r + 0.5 * volatility ** 2) * T) / vol_y
        d2 = d1 - vol_y
        return ((S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))*100/K).round(3)

    # 二分法计算隐含波动率
    def implied_volatility(self, P, S, K, T, r):
        sigma_min = 0.04
        sigma_max = 1.5
        sigma_mid = (sigma_min + sigma_max) / 2
        call_min = self.bsm_option(S, K, T, r, sigma_min)
        call_max = self.bsm_option(S, K, T, r, sigma_max)
        call_mid = self.bsm_option(S, K, T, r, sigma_mid)
        diff = P - call_mid  # 隐含期权价值p
        # print(call_min, call_mid, call_max, P, diff)
        iii = 3
        if (P < call_min) or (P > call_max):
            print('error, the price of option is beyond the limit')
            sigma_mid = np.nan
            return sigma_mid
        else:
            while abs(diff) > 0.01:
                if diff > 0:
                    sigma_min = sigma_mid
                else:
                    sigma_max = sigma_mid
                sigma_mid = (sigma_max + sigma_min) / 2
                call_mid = self.bsm_option(S, K, T, r, sigma_mid)
                diff = P - call_mid
                iii += 1
                if iii > 15:
                    print('超次：', iii, sigma_min, sigma_mid, sigma_max, call_mid, diff)
                    if iii > 20:
                        print(iii, 'error, the price of option')
                        sigma_mid = np.nan
                        return sigma_mid
        print('次数：', iii)
        return sigma_mid

    @gl_v.time_show  # 隐含波动率
    def bond_implied_volatility(self, bound_code="", k='', r='', save=''):
        with sqlite3.connect(gl_v.db_path) as conn:
            cur = conn.cursor()
            se_sql = r"select date,implication_option,stock_close,interval from '{}'".format(bound_code)
            data = pd.read_sql(se_sql, conn, parse_dates=['date'])
            # data = data.astype('float', errors='ignore')
            # data = pd.read_sql(r"select date from '{}' ORDER BY date DESC".format(code), conn)
            # print(data.head(2))
            sql_update = r"update {} set implication_volatility=? where date=?".format(bound_code)
            """C: 期权定价
            S:当前股价
            K:合约交割价格
            T:合约时长
            r:连续复利
            N:正态分布累计概率
            d1,d2:正态分布概率累计起始点"""
            for row in data.itertuples():
                da = getattr(row, 'date')._date_repr
                k = self.day_k(k, da, bound_code)  # 判断行权价
                v = self.implied_volatility(float(getattr(row, 'implication_option')), float(getattr(row, 'stock_close')),
                                            k, float(getattr(row, 'interval')), r)
                # print(da, k, v)
                if save == "y":
                    # cur.execute(sql_update, (round(v, 4), da))
                    cur.execute(sql_update, (round(v, 4), da + ' 00:00:00'))  # 大参
            cur.close()

    @staticmethod  # 历史波动率
    def history_volatility(s_code):
        with sqlite3.connect(gl_v.db_path) as conn:
            cur = conn.cursor()
            se_sql = r"select date,log_ from '{}'".format(s_code)
            # se_sql = r"select date,up_change,log_ from '{}'".format(s_code)
            data = pd.read_sql(se_sql, conn)
            # data = pd.read_sql(se_sql, conn, parse_dates=['date'])
            # data = data.astype('float', errors='ignore')
            # data = data.dropna()
            # dd = data.dropna().head(1189)
            # print(dd.values)

            """滚动计算年华波动率"""
            # data['y_v_change'] = (np.sqrt(244) * data['up_change'].rolling(122).std()).round(3)
            data['y_v_log'] = (np.sqrt(244) * (data['log_']/100).rolling(122).std()).round(6)
            data = data.loc[:, ['y_v_log', 'date']]
            # data = data.loc[:, ['y_v_change', 'y_v_log', 'date']]
            # print(data.values)
            """加列"""
            # sql2 = r"""alter table '{}' add y_v_change number(6)""".format(code)
            # sql2 = r"""alter table '{}' add y_v_log number(6)""".format(s_code)
            # cur.execute(sql2)
            """更新"""
            sql3 = "UPDATE '{}' SET y_v_log=(?) WHERE date=(?)".format(s_code)
            # sql3 = "UPDATE '{}' SET y_v_change=(?),y_v_log=(?) WHERE date=(?)".format(code)
            # print(sql3)
            # cur.executemany(sql3, data.values)
            cur.close()

    @staticmethod  # 判断行权价
    def day_k(k, r_day, bound_code):
        if bound_code == 'sh113605':
            if (r_day >= '2020-12-23') and (r_day < '2021-06-10'):
                k = 83.71
            elif (r_day >= '2021-06-10') and (r_day < '2021-10-27'):
                k = 69.09
            elif (r_day >= '2021-10-27') and (r_day < '2022-06-13'):
                k = 69.05
            elif r_day >= '2022-06-13':
                k = 57.13
            else:
                print(k)
        if bound_code == 'sh113563':
            if (r_day >= '2020-06-24') and (r_day < '2021-06-10'):
                k = 24.47
            elif (r_day >= '2021-06-10') and (r_day < '2022-06-17'):
                k = 23.87
            elif r_day >= '2022-06-17':
                k = 23.39
            else:
                print(k)
        return k

    @gl_v.time_show  # 计算可转债bs期权价值和bs方法下可转债价值
    def call_option_value(self, s_code, b_code, k, r=0.03):
        with sqlite3.connect(gl_v.db_path) as conn:
            cur = conn.cursor()
            b_sql = r"select date,stock_close,interval, bound_value from '{}'".format(b_code)
            b_df = pd.read_sql(b_sql, conn)
            # data = data.dropna()
            # print(b_df_f.head(3))
            """查询历史波动率"""
            da = b_df['date'].str[:10]
            v_sql = r"select y_v_log from '{}' where date in {} ".format(s_code, tuple(da.values))
            # print(v_sql)
            cu = cur.execute(v_sql)
            cu = cu.fetchall()
            """循环计算bs期权"""
            if len(cu) == da.shape[0]:
                b_df_f = b_df.loc[:, ['stock_close', 'interval', 'bound_value']].astype('float', errors='ignore')
                c_value = []
                for kk, vv in b_df_f.iterrows():
                    k = self.day_k(k, da[kk], b_code)  # 判断行权价
                    cc = self.bsm_option(vv['stock_close'], k, vv['interval'], r, cu[kk][0])  # 算的是一股的期权
                    # c_value.append((cc, (vv['bound_value']+cc).round(3), da[kk]))  # 柳药
                    c_value.append((cc, (vv['bound_value']+cc).round(3), da[kk]+' 00:00:00'))  # 大参
                """加列"""
                # sql2 = r"""alter table '{}' add bs_option number(6)""".format(b_code)
                # sql2 = r"""alter table '{}' add bs_option_kzz number(6)""".format(b_code)
                # cur.execute(sql2)
                """更新"""
                sql3 = "UPDATE '{}' SET bs_option=(?),bs_option_kzz=(?) WHERE date=(?)".format(b_code)
                # cur.executemany(sql3, c_value)
            else:
                print('期权计算错误')
            cur.close()

kzz_p_o = KzzPriceOption()
# kzz_p_o.ak_bond_price(code="sh113563", save='y')  # 柳药 # 输入债券代码和时间段获取k线数据.
f = ''  # 计算债券价值
if f == 'sh113605':
    interest1 = [0.30, 0.60, 1.00, 1.50, 1.80, 110]  # 包含最后一期利息
    # interest1=[0.30, 0.60, 1.00, 1.50, 1.80, 2.0, 110]  # not包括最后一年利息
    #  https://yield.chinabond.com.cn/cbweb-mn/yield_main?locale=zh_CN
    aaa1 = [2.6144, 2.9437, 3.2038, 3.7833, 3.9808, 4.0679]  # aa
    kzz_p_o.bond_value(code="sh113605", end="2026-10-22", interest=interest1, r_y=aaa1, stock_code='603233', save='')

if f == 'sh113563':
    interest1 = [0.30, 0.60, 1.00, 1.50, 1.80, 108]  # 包含最后一期利息
    # interest1=[0.30, 0.60, 1.00, 1.50, 1.80, 2.0, 110]  # not包括最后一年利息
    #  https://yield.chinabond.com.cn/cbweb-mn/yield_main?locale=zh_CN
    aaa1 = [2.6144, 2.9437, 3.2038, 3.7833, 3.9808, 4.0679]  # end为债券最后退市时间
    kzz_p_o.bond_value(code="sh113563", end="2026-01-16", interest=interest1, r_y=aaa1, stock_code='603368', save='')
# kzz_p_o.bond_implied_volatility(bound_code="sh113605", k=83.85, r=0.03, save='y')
# kzz_p_o.bond_implied_volatility(bound_code="sh113563", k=34.94, r=0.03, save='y')  # 柳药

kzz_p_o.history_volatility('603368')  # 柳药
# kzz_p_o.history_volatility('603233')  # 大参

# kzz_p_o.call_option_value('603233', 'sh113605', k=83.85)  # 大参
# kzz_p_o.call_option_value('603368', 'sh113563', k=34.94)  # 柳药

