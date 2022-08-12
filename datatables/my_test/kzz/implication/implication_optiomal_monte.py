import numpy as np
import time
import sqlite3
from global_variable import time_show, db_path


# 实证最优蒙特卡洛
class ImplicationOptionsMC(object):
    def __init__(self):
        pass

    @staticmethod  # 生成用于过程模拟的标准正态随机数,return: i行M列的二维数组
    def gen_sn(M, I, anti_paths=True):
        # M -= 1
        np.random.seed(1000)  # 数值随便指定，指定了之后对应的数值唯一
        """np.concatenate((a1,a2,…), axis=0)，0表示以列方向拼接
        //取整除 - 返回商的整数部分（向下取整） 9//2=4, -9//2 = -5,对偶变量法？
        """
        if anti_paths is True:  # i行M+1列的二维数组
            sn = np.random.standard_normal((int(I / 2), M))
            sn = np.concatenate((sn, -sn), axis=0)
        else:
            sn = np.random.standard_normal((I, M))
        return sn

    @staticmethod  # 模拟出各个路径股价=i行m+30列
    def _monte_carlo(s0, vol, r, m, sn):
        dt = 1 / m  # 单位时间
        arr_stock = np.exp((r - 0.5 * (vol ** 2)) * dt + vol * np.sqrt(dt) * sn)
        """np.cumprod,axis=1按行累乘,a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        print(np.cumprod(a, axis=1))
        [[  1   2   6]
        [  4  20 120]
        [  7  56 504]]
        print(np.cumprod(a, axis=0))  axis=0按列累乘
        [[  1   2   3]
         [  4  10  18]
         [ 28  80 162]]
        s2= s1*1.0079, s3 = s2*1.0479 >>s3=s1*1.0079*1.0479
        """
        arr_stock = np.cumprod(arr_stock, axis=1)  # 按行方向累乘
        arr_stock *= s0  # 模拟出各个路径股价
        # arr_stock *= s0[-1]  # 模拟出各个路径股价
        # arr_stock = np.concatenate((np.ones((arr_stock.shape[0], len(s0)))*s0, arr_stock), axis=1)
        return arr_stock

    # 下修 i>=31
    def _process_down(self, term_down, row, i, row_k, slice_arr, dis):
        # 根据条款分析是否触发，是则返回1，否则返回0
        is_down = self._is_down(slice_arr, term_down, row_k, dis)
        if is_down:
            mean_20 = self._slice_mc(row, i, term_down[3]).mean()  # put if inter,否则可能有空值平均的风险,i=0时切出空数组
            return is_down, row[i], mean_20
        return None, None, None

    # 赎回,当天不赎回  i>=31
    def _process_recall(self, term_recall, row, i, row_k, slice_arr):  # i>=31
        # 'recall': [5.5, 15, 30, 1.3 ], 赎回条款，一般半年后可以赎回=5.5年，15=30天内有15天大于转股价*1.3
        # 根据条款分析赎回是否触发，是则返回1，否则返回0
        is_recall = self._is_recall(slice_arr, term_recall, row_k)
        if is_recall:
            return is_recall, row[i+1]  # 触发的第二天
        return None, None

    # 回售  i>=31
    def _process_resell(self, term_resell, row, k2, y_day, r, m, term_coupon, slice_arr, arr_mc_k, key, is_include, len_s0):
        # 'resell': [2, 30, 30, 0.7, 100]  # 回售条款
        is_resell = self._is_resell(slice_arr, term_resell, arr_mc_k[key, k2])
        if is_resell:
            arr_mc_k_k2 = arr_mc_k[key, k2+1:]
            vv_ = self.is_include_coupon(is_include, k2, m, term_coupon, y_day, len_s0)  # 回售是否包含利息
            discount1 = 1 / ((1 + r) ** (1 / y_day))
            row_k2 = row[k2+1:]
            row_resell = np.zeros_like(row_k2)
            row_resell[-1] = np.max([(row_k2[-1] / arr_mc_k_k2[-1]) * 100, term_coupon[-1]])
            for t in range(row_k2.shape[0] - 2, -1, -1):
                if t == 0:  # 这里term_resell[-1]没有包含利息?
                    convert = np.max([(row_k2[0] / arr_mc_k_k2[0]) * 100, term_resell[-1] + vv_])
                else:
                    convert = (row_k2[t] / arr_mc_k_k2[t]) * 100
                convert1 = row_resell[t + 1] * discount1
                row_resell[t] = np.where(convert > convert1, convert, convert1)
            return is_resell, row_resell[0]
        return None, None

    @staticmethod  # 取30天或不足30# array=row、arrTime, point=i, length=30
    def _slice_mc(arr, i, length):
        return arr[int(max([0, i - length])):int(i)]

    @staticmethod  # 根据条款分析下修是否触发，是则返回1，否则返回0。
    def _is_down(arr, down_term, row_k, dis):
        # 'down': [5.5, 1, 10, 20, 15, 30, 0.85]  # 半年内不下调. 面值，净资产，20交易日，15交易日，30交易日
        # 哪些点的价格达到下修要求
        logic_price = arr[:] < down_term[-1] * dis * row_k
        # 利用numpy的广播原理，1*loglcTime*logicPrice 会形成类似（0.1.0.1..]的向量#其中1代表这一天既在赎回期内，
        # 价格也达到触发线，否则是0 这样再加到一起，就可以看符合赎回要求的天数达不达到条款要求的天数了recall_term[1]=15
        # tt = np.array([True, True, False])
        # tt2 = np.array([True, False, True])
        # print("tt---", 1*tt*tt2)  # [1 0 0]有一假则乘积为假
        return 1 if np.sum(1 * logic_price) >= down_term[4] else 0

    @staticmethod  # 根据条款分析赎回是否触发，是则返回1，否则返回0。
    def _is_recall(arr, recall_term, row_k):
        # 哪些点的价格达到赎回要求,recall_term='Recall': [5.5, 15, 30, 1.3],  赎回条款
        logic_price = arr[:] > recall_term[-1]*1.01*row_k
        # print("logicPrice---", logicPrice)
        # 利用numpy的广播原理，1*loglcTime*logicPrice 会形成类似（0.1.0.1..]的向量#其中1代表这一天既在赎回期内，
        # 价格也达到触发线，否则是0 这样再加到一起，就可以看符合赎回要求的天数达不达到条款要求的天数了recall_term[1]=15
        # tt = np.array([True, True, False])
        # tt2 = np.array([True, False, True])
        # print("tt---", 1*tt*tt2)  # [1 0 0]有一假则乘积为假
        return 1 if np.sum(1 * logic_price) >= recall_term[1] else 0

    @staticmethod  # __isResell是根据条款分析回售是否触发，是则返回1，否则返回0。
    def _is_resell(slice_arr, term_resell, row_k):  # 'Resell': [2, 30, 30, 0.7, 100]  # 回售条款
        logic_price = slice_arr[:] < term_resell[3] * row_k
        return 1 if np.sum(1 * logic_price) >= term_resell[1] else 0

    @staticmethod  # 回售是否包含利息
    def is_include_coupon(is_include, k2, m, term_coupon, y_day, len_s0):
        if is_include:  # 1包含利息, 0不包含
            vv_ = 0
        else:
            y_ = (m+len_s0-k2) / y_day  # y_<=1为最后一年, 距离结束时间
            if (y_ > 0) and (y_ <= 1):
                vv_ = term_coupon[-2] * (1 - y_)  # 剩最后一年=1.8
            elif (y_ > 1) and (y_ <= 2):
                vv_ = term_coupon[-3] * (2 - y_)  # 1.3
            else:
                print(k2, '回售利息 error')
        return vv_*0.8

    @staticmethod  # 画前20条模拟路径
    def get_plt(arr_mc, num):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        # matplotlib其实是不支持显示中文的 显示中文需要一行代码设置字体
        mpl.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
        plt.figure(figsize=(10, 7))
        plt.grid(True)
        plt.xlabel('Time step')
        plt.ylabel('index level')
        for i in range(num):
            plt.plot(arr_mc[i])
        plt.title('模拟{}条股价路径 '.format(num))
        plt.show()

    # @time_show  # 最优停时蒙特卡洛
    def analyse_optimal_mc(self, arr_mc, k, m, r, y_day, term, is_include, len_s0):
        """大参
              term = {
                  'coupon': [0.3, 0.5, 0.8, 1.3, 1.8, 110],  # 票息
                  'recall': [5.5, 15, 30, 1.3],  # 赎回条款，一般半年后可以赎回=5.5年，15=30天内有15天大于转股价*1.3
                  'resell': [2, 30, 30, 0.7, 100],  # 回售条款,最后2年,是否包含最后一期利息？
                  'down': [5.5, 1, 10, 20, 15, 30, 0.85]  # 半年内不下调. 面值，净资产，20交易日，15交易日，30交易日
              }
              """
        term_down = term['down']
        term_resell = term['resell']
        term_recall = term['recall']
        term_coupon = term['coupon']
        k_arr = np.zeros(shape=(arr_mc.shape[0], 3))
        arr_mc_k = np.full_like(arr_mc, k)  # 创建行权价k=25的矩阵i*(m+1)
        dis = 0.93  # 非回售期下修折扣
        for key, row in enumerate(arr_mc):  # arr_mc=i行m+1列
            row_k = k
            flag = ''
            start1 = 30
            if row.shape[0] > 244 * 5.5:
                start1 = int(row.shape[0]-244 * 5.5+1)
            """row.shape[0]-1表最后两天不分析是否下修，赎回，回售,直接到期赎回"""
            for k2 in range(start1, row.shape[0]-1):  # 循环的是路径i，同一路径下不会同时满足赎回和回售
                # if (m+30-k2) <= 244 * 5.5:  # 剩余时间少于5.5年才回售，赎回，下修
                if (row.shape[0] - k2) <= 244*2:
                    dis = 0.98  # 回售期下修折扣
                """ 回售，赎回，下修都是30内"""
                slice_arr = self._slice_mc(row, k2, term_down[5])
                """if转股价>净资产或面值，下修"""
                if (row_k > term_down[1]) and (row_k > term_down[2]):
                    # 是否下修,进入回售期？term_down, row, i, row_k, slice_arr, dis
                    is_down, down_value, mean_20 = self._process_down(term_down, row, k2, arr_mc_k[key, k2], slice_arr, dis)
                    if is_down:
                        # 'down': [5.5,1, 10, 20, 15, 30, 0.85 * row_k]  # 半年内不下调. 面值，净资产，20交易日，15交易日，30交易日
                        """下修后的转股价格一般要求不低于股东大会前20个交易日股票交易均价和前1交易日的交易均价，并低于之前的转股价格"""
                        row_k = min([max([term_down[1], term_down[2], mean_20, down_value]), row_k])
                        arr_mc_k[key, k2+1:] = row_k  # 触发当天不下调?
                # 是否赎回 当天不赎回
                is_recall, call_value = self._process_recall(term_recall, row, k2, arr_mc_k[key, k2], slice_arr)
                if is_recall:
                    """触发赎回则计算转股价值并折现为现值.第一个触发赎回就break，不再运行后面"""
                    k_arr[key, 0] = k2+1
                    k_arr[key, 1] = 100 * (call_value / arr_mc_k[key, k2+1])  # 触发第二天转股价值
                    k_arr[key, 2] = 1
                    flag += 'break'
                    break
                # 是否触及回售,剩余时间少于2年才回售期
                if (row.shape[0] - k2) <= 244*2:
                    is_resell, sell_value = self._process_resell(term_resell, row, k2, y_day, r, m, term_coupon,
                                                                 slice_arr, arr_mc_k, key, is_include, len_s0)
                    if is_resell:  # 回售考虑最优停时？
                        # print("回售", key, k2, sell_value, row_k, arr_mc_k[key, k2], row[k2])
                        k_arr[key, 0] = k2+1
                        k_arr[key, 1] = sell_value
                        k_arr[key, 2] = 2
                        flag += 'break'
                        break
            if flag == '':  # 如果为空说明没有发生赎回和回售
                # print(row[k2:], arr_mc_k[key, k2:])
                # 如果整个生命周期都没有触发附加条款，则计算到期价值(或转股或还债)
                kzz_v = 100 * (row[-1] / arr_mc_k[key, -1])  # 转股价值
                k_arr[key, 0] = row.shape[0]-1
                k_arr[key, 1] = np.max([kzz_v, term_coupon[-1]])  # 包含最后一期利息？
                k_arr[key, 2] = 3
        return k_arr, arr_mc_k

    @staticmethod  # 最优停时蒙特卡洛son,倒推计算最优价值
    def optimal_mc_son(i_k, arr_mc_k, arr_mc, r, y_day, len_s0):
        discount = 1 / ((1 + r) ** (1 / y_day))
        optimal_mc = np.zeros_like(arr_mc)
        for key, row in enumerate(arr_mc):  # arr_mc=i行m列:
            i_k_row = i_k[key]
            i = int(i_k_row[0])-len_s0  # 该行停止时位置
            row = row[:i + 1]  # 该行停止时长度集合
            optimal_mc[key, i] = i_k_row[1]  # 该行停止时价值
            mc_k_row = arr_mc_k[key, len_s0:i+len_s0+1]  # 该行行权价长度=row
            for t in range(row.shape[0] - 2, -1, -1):
                convert = (row[t] / mc_k_row[t]) * 100
                row_v_dis = optimal_mc[key, t + 1] * discount
                optimal_mc[key, t] = np.where(convert > row_v_dis, convert, row_v_dis)
        return (np.mean(optimal_mc[:, 0]) * discount).round(3)

    def kzz_mc(self, s0, k, v, r, m, i, y_day, term, is_include):
        r = np.log(1 + r)  # 0.029,计算年华利率
        # sn = self.gen_sn(m, i, anti_paths=False)  # return: i行m列的二维数组
        sn = self.gen_sn(m, i, anti_paths=True)
        arr_mc = self._monte_carlo(s0[-1], v, r, m, sn)  # 模拟出各个路径股价=i行m+1列
        # self.get_plt(arr_mc[:, len(s0)-1:], 20)  # 画20条模拟路径
        len_s0 = s0.shape[0]
        arr_mc_s0 = np.concatenate((np.ones((arr_mc.shape[0], len_s0)) * s0, arr_mc), axis=1)
        i_k, arr_mc_k = self.analyse_optimal_mc(arr_mc_s0, k, m, r, y_day, term, is_include, len_s0)
        # num = i_k[:, 2]
        # redeem, back, end = sum(num == 1), sum(num == 2), sum(num == 3)
        # time_start = time.clock()  # 记录开始时间
        son_v = self.optimal_mc_son(i_k, arr_mc_k, arr_mc, r, y_day, len_s0)
        # time_sum = time.clock() - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        # print("最优分析time:", time_sum, "/赎回：", redeem, ":", redeem / i, "/回售：", back, ":", back / i, "/到期：", end, ":", end / i)
        return son_v

    # 10*len数据库版
    def sql_monte_carlo_call_option_value(self, s_code, b_code, start_d):
        row_b, row_s = self.sql_get_s0(b_code, s_code)
        r = 0.03
        i = 1000
        y_day = 244
        if b_code == 'sh113605':
            k = 83.85  # 大参
            term = {  # 大参
                'coupon': [0.3, 0.5, 0.8, 1.3, 1.8, 110-10*0.2],  # 票息
                'recall': [5.5, 15, 30, 1.3],  # 赎回条款，一般半年后可以赎回=5.5年，15=30天内有15天大于转股价*1.3
                'resell': [2, 30, 30, 0.7, 100],  # 回售条款,最后2年,是否包含最后一期利息？
                'down': [5.5, 1, 7.7, 20, 15, 30, 0.85]  # 半年内不下调. 面值，净资产，20交易日，15交易日，30交易日
            }
        if b_code == 'sh113563':  # 柳药
            k = 34.94
            term = {  # 柳药
                'coupon': [0.3, 0.5, 0.8, 1.3, 1.8, 108-8*0.2],  # 票息
                'recall': [5.5, 15, 30, 1.3],  # 赎回条款，一般半年后可以赎回=5.5年，15=30天内有15天大于转股价*1.3
                'resell': [2, 30, 30, 0.7, 100],  # 回售条款,最后2年,是否包含最后一期利息？
                'down': [5.5, 1, 15.256, 20, 15, 30, 0.85]  # 半年内不下调. 面值，净资产，20交易日，15交易日，30交易日
            }
        self.sql_for_day_value_mc(b_code, i, k, r, row_b, row_s, term, y_day, start_d)

    def sql_for_day_value_mc(self, b_code, i, k, r, row_b, row_s, term, y_day, start_d):
        # len_b = len(row_b)
        len_b = row_b.__len__()
        # row_s = np.array(row_s)
        if len_b != row_s.shape[0]:
            print('股票和可转债数据数量不等')
        if len_b == row_s.shape[0]:
            t_total = 0
            for ii in range(len_b):
                r_b = row_b[ii]  # date,interval close
                r_day = r_b[0]
                if r_day >= start_d:
                    time_start = time.clock()  # 记录开始时间
                    # s_current = row_s[ii, 1]
                    r_s = row_s[max(0, ii - 28):ii + 1][:, 0]  # close切出前29个
                    k = self.day_k(k, r_day, b_code)  # 判断行权价
                    self.day_net(r_day, term, b_code)  # 判断净资产
                    """
                    r_s  股票29收盘价,
                    r_b[1] 债券剩余时间,
                    r_b[2] 债券收盘价
                    row_s[ii, 1]当前历史波动率
                    is_include = 1,回售包括利
                    """
                    # 二分法计算隐含波动率
                    sigma = self.two_implied_volatility(r_b[2], k, i, r, y_day, term, r_s, r_b[1], row_s[ii, 1], is_include=0)
                    dd = 'g'
                    if dd:
                        self.update_kzz_value_optimal_mc(b_code, round(sigma, 3), r_b[0])
                    time_sum = time.clock() - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
                    t_total += time_sum
                    print(r_day, ii, '/总时:', (t_total / 60), '/单程:', time_sum, '/vol值:', sigma, '/行权价:', k, '/净资产:', term['down'][2])

    @staticmethod  # 保存最优停估算值
    def update_kzz_value_optimal_mc(b_code, kzz_value, r_b0):
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            sql_u = "UPDATE '{}' SET optimal_mc_vol=(?) WHERE date=(?)".format(b_code)
            """批量更新"""
            # val_list = [(-5.1, '2020-11-13 00:00:00'), (1.6, '2020-11-16 00:00:00')]
            cur.executemany(sql_u, [(kzz_value, r_b0)])
            conn.commit()
            cur.close()

    @staticmethod  # 获取前29股票价格
    def sql_get_s0(b_code, s_code):
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            # sql_s_b = "select date,interval from '{}' where date>=?".format(b_code)
            # cur_b = cur.execute(sql_s_b, ('2022-06-13', ))
            sql_s_b = "select date,interval,close from '{}'".format(b_code)
            cur_b = cur.execute(sql_s_b)
            # row_b = cur_b.fetchmany(200)
            row_b = cur_b.fetchall()
            sql_s_s = "select close,y_v_log from '{}' where date>=?".format(s_code)
            cur_s = cur.execute(sql_s_s, (row_b[0][0][:10],))
            # row_s = cur_s.fetchmany(400)
            row_s = cur_s.fetchall()
            row_s = np.array(row_s)
            cur.close()
            return row_b, row_s

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
        if bound_code == 'sh113563':  # 柳药
            if (r_day >= '2020-06-24') and (r_day < '2021-06-10'):
                k = 24.47
            elif (r_day >= '2021-06-10') and (r_day < '2022-06-17'):
                k = 23.87
            elif r_day >= '2022-06-17':
                k = 23.39
            else:
                print(k)
        return k

    @staticmethod  # 判断净资产
    def day_net(r_day, term, bound_code):
        if bound_code == 'sh113605':
            if (r_day > '2020-12-31') and (r_day <= '2021-03-31'):
                term['down'][2] = 7.89
            elif (r_day > '2021-03-31') and (r_day <= '2021-06-30'):
                term['down'][2] = 8.12
            elif (r_day > '2021-06-30') and (r_day <= '2021-09-30'):
                term['down'][2] = 6.49
            elif (r_day > '2021-09-30') and (r_day <= '2021-12-31'):
                term['down'][2] = 6.73
            elif (r_day > '2021-12-31') and (r_day <= '2022-03-31'):
                term['down'][2] = 6.69
            elif (r_day > '2022-03-31') and (r_day <= '2022-06-30'):
                term['down'][2] = 7.17
            # term['down'][2] *= 10
        if bound_code == 'sh113563':  # 柳药
            """
            22-03-31 21-12-31 21-09-30 21-06-30 21-03-31 20-12-31 20-09-30 20-06-30 20-03-31 19-12-31
            15.2561	14.6463	  14.4226	13.9551	14.2193	 13.6372	13.2473	12.6585	17.5927 16.87"""
            if (r_day > '2019-12-31') and (r_day <= '2020-03-31'):
                term['down'][2] = 16.87
            elif (r_day > '2020-03-31') and (r_day <= '2020-06-30'):
                term['down'][2] = 17.5927
            elif (r_day > '2020-06-30') and (r_day <= '2020-09-30'):
                term['down'][2] = 12.6585
            elif (r_day > '2020-09-30') and (r_day <= '2020-12-31'):
                term['down'][2] = 13.2473
            elif (r_day > '2020-12-31') and (r_day <= '2021-03-31'):
                term['down'][2] = 13.6372
            elif (r_day > '2021-03-31') and (r_day <= '2021-06-30'):
                term['down'][2] = 14.2193
            elif (r_day > '2021-06-30') and (r_day <= '2021-09-30'):
                term['down'][2] = 13.9551
            elif (r_day > '2021-09-30') and (r_day <= '2021-12-31'):
                term['down'][2] = 14.4226
            elif (r_day > '2021-12-31') and (r_day <= '2022-03-31'):
                term['down'][2] = 14.6463
            elif (r_day > '2022-03-31') and (r_day <= '2022-06-30'):
                term['down'][2] = 15.2561

    # 二分法计算隐含波动率
    def two_implied_volatility(self, p, k, i, r, y_day, term, r_s_29, r_b_int, s_current, is_include):
        sigma_min = s_current*0.4
        sigma_max = s_current*2.1
        """
        p: 债券收盘价
        r_s_29 股票收盘价,
        r_b_int 债券剩余时间,
        is_include = 1,包括利
        """
        call_min = self.kzz_mc(r_s_29, k, sigma_min, r, int(244 * float(r_b_int)), i, y_day, term, is_include)
        call_max = self.kzz_mc(r_s_29, k, sigma_max, r, int(244 * float(r_b_int)), i, y_day, term, is_include)
        if p < call_min:
            sigma_min *= 0.3
            call_min = self.kzz_mc(r_s_29, k, sigma_min, r, int(244 * float(r_b_int)), i, y_day, term, is_include)
            print('error0.4:', p, call_min, sigma_min)
        if p > call_max:
            sigma_max *= 1.7
            call_max = self.kzz_mc(r_s_29, k, sigma_max, r, int(244 * float(r_b_int)), i, y_day, term, is_include)
            print('error1.5:', p, call_max, sigma_max)

        if (p < call_min) or (p > call_max):
            print('error, the price of option is beyond the limit')
            sigma_mid = np.nan
            return sigma_mid
        else:
            sigma_mid = (sigma_min + sigma_max) / 2
            call_mid = self.kzz_mc(r_s_29, k, sigma_mid, r, int(244 * float(r_b_int)), i, y_day, term, is_include)
            diff = p - call_mid  # 隐含期权价值p
            if p >= 10:
                a_dif = 0.1  # 相差一毛
            else:
                a_dif = p*0.005  # 5/200=5*0.005=0.025  1/200=0.005
            iii = 3
            while abs(diff) > a_dif:
                if diff > 0:
                    sigma_min = sigma_mid
                else:
                    sigma_max = sigma_mid
                sigma_mid = (sigma_max + sigma_min) / 2
                call_mid = self.kzz_mc(r_s_29, k, sigma_mid, r, int(244 * float(r_b_int)), i, y_day, term, is_include)
                diff = p - call_mid
                iii += 1
                """0.2092132568359375 0.20922431945800782 0.20923538208007814"""
                if (sigma_max-sigma_mid < 0.01) and (sigma_mid-sigma_min < 0.01):
                    print('超次：', iii, sigma_min, sigma_mid, sigma_max, call_mid, diff, p)
                    return sigma_mid
                if iii > 20:
                    print('error', iii, sigma_min, sigma_mid, sigma_max, call_mid, diff, p)
                    sigma_mid = np.nan
                    return sigma_mid
            print('计算次数：', iii)
            return sigma_mid

implication = ImplicationOptionsMC()
implication.sql_monte_carlo_call_option_value(s_code='603233', b_code='sh113605', start_d='2020-11-13')  # 大参  10*len数据库版
# implication.sql_monte_carlo_call_option_value(s_code='603368', b_code='sh113563', start_d='2021-02-08')  # 柳药




"""
1000
2020-02-24 0 /总时: 8.642066058720905 /单程: 518.5239635232543 /债值: 0.6120532531738282 /行权价: 34.94 /净资产: 16.87
2020-02-24 0 /总时: 30.773411182549868 /单程: 1846.404670952992 /vol值: 0.6119555664062499 /行权价: 34.94 /净资产: 16.87
2000
 2020-02-24 0 /总时: 66.44592621601447 /单程: 3986.755572960868 /vol值: 0.6102551269531249 /行权价: 34.94 /净资产: 16.87
 500
 2020-02-24 0 /总时: 9.284551208076236 /单程: 557.0730724845741 /vol值: 0.6045465087890624 /行权价: 34.94 /净资产: 16.87
 98.82 137.014 155.545 125.74"""
