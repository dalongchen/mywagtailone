import numpy as np
import sqlite3
db_path = r"D:\myzq\axzq\T0002\stock_load\thesis\kzz\monte.db"  # 数据库路径


# 普通+最优停时蒙特卡洛
class KzzOptionsMC(object):
    def __init__(self):
        pass

    @staticmethod  # 生成用于过程模拟的标准正态随机数,return: i行M列的二维数组
    def gen_sn(M, I, anti_paths=True):
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

    @staticmethod  # 模拟出各个路径股价=i行m+1列
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
        return arr_stock

    # 下修
    def _process_down(self, term_down, row, i, arr_time, y_day, row_k, m, term_resell):
        if ((m-i) / y_day) >= term_resell[0]:  # 如果现今大于2年，为未回售期
            dis = 0.93
        else:
            dis = 0.98
        # 'resell': [2, 30, 30, 0.7, 103]  # 回售条款
        # 'down': [5.5,1, 10, 20, 15, 30, 0.85]  # 半年内不下调. 面值，净资产，20交易日，15交易日，30交易日
        if row[i] < term_down[-1] * dis * row_k:  # 股价<0.85可能下修,0.85*57.13=48.56
            # 前30天或不足30， term["down"][5]=30
            slice_arr = self._slice_mc(row, i, term_down[5])
            slice_arr_time = self._slice_mc(arr_time, i, term_down[5])
            if slice_arr.shape[0] > term_down[4]:  # slice_arr数组长度大于15才判断是否下修
                # 根据条款分析是否触发，是则返回1，否则返回0
                is_down = self._is_down(slice_arr, slice_arr_time, term_down, row_k, dis)
                if is_down:
                    mean_20 = self._slice_mc(row, i, term_down[3]).mean()  # put if inter,否则可能有空值平均的风险,i=0时切出空数组
                    return is_down, row[i], mean_20
        return None, None, None

    # 赎回
    def _process_recall(self, term_recall, row, i, arr_time, y_day, row_k):
        if row[i] > term_recall[-1]*1.01*row_k:  # 股价大于1.30可能赎回,term["Recall"][-1]=1.30*57.13=74.15
            # 价大于1.30的前30天或不足30， term["Recall"][2]=30
            # 'recall': [5.5, 15, 30, 1.3 ], 赎回条款，一般半年后可以赎回=5.5年，15=30天内有15天大于转股价*1.3
            slice_arr = self._slice_mc(row, i, term_recall[2])
            slice_arr_time = self._slice_mc(arr_time, i, term_recall[2])
            if slice_arr.shape[0] > term_recall[1]:  # slice_arr数组长度大于15才判断是否赎回
                # 根据条款分析赎回是否触发，是则返回1，否则返回0
                is_recall = self._is_recall(slice_arr, slice_arr_time, term_recall, row_k)
                if is_recall:
                    return is_recall, (i+1) / y_day, row[i]  # 剩余m天数不包括今天,so，i+1才是折现到今天s0处
        return None, None, None

    # 回售
    def _process_resell(self, term_resell, row, i, arr_time, y_day, row_k, r, end_time, term_coupon):
        if row[i] < term_resell[3]*0.98*row_k:  # 'resell': [2, 30, 30, 0.7, 103]  # 回售条款
            slice_arr = self._slice_mc(row, i, term_resell[2])
            slice_arr_time = self._slice_mc(arr_time, i, term_resell[2])
            if slice_arr.shape[0] >= term_resell[1]:  # slice_arr数组长度大于30才判断是否回售
                is_resell = self._is_resell(slice_arr, slice_arr_time, term_resell, row_k)
                if is_resell:
                    # y_aa = 1 / ((1 + r) ** ((i+1) / y_day))  # 剩余m天数不包括今天,so，i+1才是折现到今天s0处
                    dis_resell = term_resell[-1] / ((1 + r) ** ((i+1) / y_day))  # 回售折现到s0处
                    end_v = term_coupon[-1] / ((1 + r) ** end_time)  # 持有到终结折现到s0处, 包含最后一期利息？
                    if dis_resell > end_v:  # 回售价值大于持有到终结价值
                        kzz_v = 100 * (row[i] / row_k)  # 转股价值
                        return is_resell, np.max([kzz_v, term_resell[-1]])  # 不折现
        return None, None

    @staticmethod  # 取30天或不足30# array=row、arrTime, point=i, length=30
    def _slice_mc(arr, i, length):
        return arr[int(max([0, i - length])):int(i)]

    @staticmethod  # 根据条款分析下修是否触发，是则返回1，否则返回0。
    def _is_down(arr, slice_arr_time, down_term, row_k, dis):
        # 'down': [5.5, 1, 10, 20, 15, 30, 0.85]  # 半年内不下调. 面值，净资产，20交易日，15交易日，30交易日
        logic_time = np.array(slice_arr_time[:]) < down_term[0]
        # 哪些点的价格达到下修要求
        logic_price = arr[:] < down_term[-1] * dis * row_k
        # 利用numpy的广播原理，1*loglcTime*logicPrice 会形成类似（0.1.0.1..]的向量#其中1代表这一天既在赎回期内，
        # 价格也达到触发线，否则是0 这样再加到一起，就可以看符合赎回要求的天数达不达到条款要求的天数了recall_term[1]=15
        # tt = np.array([True, True, False])
        # tt2 = np.array([True, False, True])
        # print("tt---", 1*tt*tt2)  # [1 0 0]有一假则乘积为假
        return 1 if np.sum(1 * logic_time * logic_price) >= down_term[4] else 0

    @staticmethod  # 根据条款分析赎回是否触发，是则返回1，否则返回0。
    def _is_recall(arr, slice_arr_time, recall_term, row_k):
        # 哪些点在赎回期之内?,recall_term='Recall': [5.5, 15, 30, 1.3],  # 赎回条款
        logic_time = np.array(slice_arr_time[:]) < recall_term[0]
        # 哪些点的价格达到赎回要求
        logic_price = arr[:] > recall_term[-1] * row_k
        # print("logicPrice---", logicPrice)
        # 利用numpy的广播原理，1*loglcTime*logicPrice 会形成类似（0.1.0.1..]的向量#其中1代表这一天既在赎回期内，
        # 价格也达到触发线，否则是0 这样再加到一起，就可以看符合赎回要求的天数达不达到条款要求的天数了recall_term[1]=15
        # tt = np.array([True, True, False])
        # tt2 = np.array([True, False, True])
        # print("tt---", 1*tt*tt2)  # [1 0 0]有一假则乘积为假
        return 1 if np.sum(1 * logic_time * logic_price) >= recall_term[1] else 0

    @staticmethod  # __isResell是根据条款分析回售是否触发，是则返回1，否则返回0。
    def _is_resell(slice_arr, slice_arr_time, term_resell, row_k):  # 'Resell': [2, 30, 30, 0.7, 103]  # 回售条款
        logic_time = np.array(slice_arr_time[:]) < term_resell[0]  # 最后2年才能回售,true,未到最后2年为false
        logic_price = slice_arr[:] < term_resell[3] * row_k
        return 1 if np.sum(1 * logic_time * logic_price) >= term_resell[1] else 0

    @staticmethod  # 转股价值并折现
    def _cash_flow(row_end_time, row_value, row_k, r):
        kzz_now_v = 100 * (row_value / row_k)  # 转股价值
        dis_kzz_v = kzz_now_v / ((1 + r) ** row_end_time)  # 转股价值折现
        return dis_kzz_v

    @staticmethod  # 回售是否包含利息
    def is_include_coupon(is_include, k2, m, term_coupon, y_day):
        if is_include:  # 1包含利息, 0不包含
            vv_ = 0
        else:
            y_ = (m - k2) / y_day  # y_<=1为最后一年
            if (y_ >= 0) and (y_ < 1):
                vv_ = term_coupon[-2] * (1 - y_)
            elif (y_ >= 1) and (y_ < 2):
                vv_ = term_coupon[-3] * (2 - y_)
            else:
                print(k2, 'time error')
        return vv_

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
                print("error")
        if bound_code == 'sh113563':
            if (r_day >= '2020-06-24') and (r_day < '2021-06-10'):
                k = 24.47
            elif (r_day >= '2021-06-10') and (r_day < '2022-06-17'):
                k = 23.87
            elif r_day >= '2022-06-17':
                k = 23.39
            else:
                print("error")
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
        if bound_code == 'sh113563':
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

    @staticmethod  # 前1000条模拟路径
    def get_plt(arr_mc):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        # matplotlib其实是不支持显示中文的 显示中文需要一行代码设置字体
        mpl.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
        plt.figure(figsize=(10, 7))
        plt.grid(True)
        plt.xlabel('Time step')
        plt.ylabel('index level')
        for i in range(1000):
            plt.plot(arr_mc[i])
        plt.title('大参前1000路径 ')
        plt.show()

    # 普通蒙特卡洛
    def analyse_general_mc(self, arr_mc, arr_time, k, m, r, term_coupon, term_down, term_recall, term_resell, y_day,
                           is_include):
        end_time = (m + 1) / y_day  # 剩余m天数不包括今天,so，m+1才是折现到今天s0处
        i_value = []
        for key, row in enumerate(arr_mc):  # arr_mc=i行m列
            row_k = k
            flag = ''
            for k2, v2 in enumerate(row):  # 循环的是i，说明i发生不会同时满足赎回和回售
                """if转股价>净资产或面值，下修"""
                if (row_k > term_down[1]) and (row_k > term_down[2]):
                    # 是否下修,进入回售期？     term_down, row, i, arr_time, y_day, row_k, m, term_resell
                    is_down, down_value, mean_20 = self._process_down(term_down, row, k2, arr_time, y_day, row_k, m, term_resell)
                    if is_down:
                        # 'down': [5.5,1, 10, 20, 15, 30, 0.85 * row_k]  # 半年内不下调. 面值，净资产，20交易日，15交易日，30交易日
                        convert_p2 = max([term_down[1], term_down[2], mean_20, down_value])
                        """下修后的转股价格一般要求不低于股东大会前20个交易日股票交易均价和前1交易日的交易均价，并低于之前的转股价格"""
                        row_k = min([convert_p2, row_k])
                # 是否赎回
                is_recall, call_end_time, call_value = self._process_recall(term_recall, row, k2, arr_time, y_day, row_k)
                if is_recall:
                    """触发赎回则计算转股价值并折现为现值.第一个触发赎回就break，不再运行后面"""
                    i_value.append(self._cash_flow(call_end_time, call_value, row_k, r))
                    flag += 'break'
                    break
                # 是否触及回售
                is_resell, sell_value = self._process_resell(term_resell, row, k2, arr_time, y_day, row_k, r,
                                                             end_time, term_coupon)
                if is_resell:
                    print("回售", key, k2, sell_value)
                    vv_ = self.is_include_coupon(is_include, k2, m, term_coupon, y_day)  # 回售是否包含利息
                    i_value.append((sell_value+vv_) / ((1 + r) ** ((k2+1) / y_day)))  # 折现
                    flag += 'break'
                    break
            if flag == '':  # 如果为空说明没有发生赎回和回售
                # 如果整个生命周期都没有触发附加条款，则计算到期价值(或转股或还债)
                kzz_v = 100 * (row[-1] / row_k)  # 转股价值
                i_value.append(np.max([kzz_v, term_coupon[-1]]) / ((1 + r) ** end_time))  # 折现,包含最后一期利息？
        return np.mean(i_value).round(3)

    # 最优停时蒙特卡洛
    def analyse_optimal_mc(self, arr_mc, arr_time, k, m, r, term_coupon, term_down, term_recall, term_resell,
                           y_day, is_include):
        k_arr = np.zeros(shape=(arr_mc.shape[0], 2))
        arr_mc_k = np.full_like(arr_mc, k)
        end_time = (m + 1) / y_day  # 剩余m天数不包括今天,so，m+1才是折现到今天s0处
        for key, row in enumerate(arr_mc):  # arr_mc=i行m列
            row_k = k
            flag = ''
            for k2, v2 in enumerate(row):  # 循环的是i，说明i发生不会同时满足赎回和回售
                """if转股价>净资产或面值，下修"""
                if (row_k > term_down[1]) and (row_k > term_down[2]):
                    # 是否下修,进入回售期？
                    is_down, down_value, mean_20 = self._process_down(term_down, row, k2, arr_time, y_day, row_k, m,
                                                                      term_resell)
                    if is_down:
                        # 'down': [5.5,1, 10, 20, 15, 30, 0.85 * row_k]  # 半年内不下调. 面值，净资产，20交易日，15交易日，30交易日
                        """下修后的转股价格一般要求不低于股东大会前20个交易日股票交易均价和前1交易日的交易均价，并低于之前的转股价格"""
                        row_k = min([max([term_down[1], term_down[2], mean_20, down_value]), row_k])
                        arr_mc_k[key, k2:] = row_k  # 触发当天不下调?
                # 是否赎回
                is_recall, call_end_time, call_value = self._process_recall(term_recall, row, k2, arr_time, y_day, row_k)
                if is_recall:
                    """触发赎回则计算转股价值并折现为现值.第一个触发赎回就break，不再运行后面"""
                    k_arr[key, 0] = k2
                    k_arr[key, 1] = 100 * (call_value / row_k)  # 转股价值
                    flag += 'break'
                    break
                # 是否触及回售
                is_resell, sell_value = self._process_resell(term_resell, row, k2, arr_time, y_day, row_k, r,
                                                             end_time, term_coupon)
                if is_resell:   # 回售考虑最优停时？
                    print("回售", key, k2, sell_value)
                    vv_ = self.is_include_coupon(is_include, k2, m, term_coupon, y_day)  # 回售是否包含利息
                    k_arr[key, 0] = k2
                    k_arr[key, 1] = sell_value + vv_  # 不折现
                    flag += 'break'
                    break
            if flag == '':  # 如果为空说明没有发生赎回和回售
                # 如果整个生命周期都没有触发附加条款，则计算到期价值(或转股或还债)
                kzz_v = 100 * (row[-1] / row_k)  # 转股价值
                k_arr[key, 0] = k2
                k_arr[key, 1] = np.max([kzz_v, term_coupon[-1]])  # 包含最后一期利息？

        return k_arr, arr_mc_k

    @staticmethod  # 最优停时蒙特卡洛son,倒推计算最优价值
    def optimal_mc_son(i_k, arr_mc_k, arr_mc, r, y_day):
        discount = 1/((1+r)**(1/y_day))
        row_value = []
        for key, row in enumerate(arr_mc):  # arr_mc=i行m列:
            i_k_row = i_k[key]
            i = int(i_k_row[0])  # 该行停止时位置
            row = row[:i+1]  # 该行停止时长度集合
            row_v = np.zeros_like(row)
            row_v[-1] = i_k_row[1]  # 该行停止时价值
            mc_k_row = arr_mc_k[key, :i+1]  # 该行行权价长度=row
            for t in range(row.shape[0]-2, -1, -1):
                convert = (row[t]/mc_k_row[t])*100
                row_v_dis = row_v[t + 1] * discount
                row_v[t] = np.where(convert > row_v_dis, convert, row_v_dis)
            row_value.append(row_v[0])
        return (np.mean(row_value)*discount).round(3)

    # 省时间最优停时蒙特卡洛
    def less_analyse_optimal_mc(self, arr_mc, arr_time, k, m, r, term_coupon, term_down, term_recall,
                                term_resell, y_day):
        column_len = arr_mc.shape[1]
        end_time = (column_len + 1) / y_day  # 剩余m天数不包括今天,so，m+1才是折现到今天s0处
        mean_v = []
        for key, row in enumerate(arr_mc):  # arr_mc=i行m列
            row_mc_k = np.full_like(row, k)
            row_k = k
            flag = ''
            for k2, v2 in enumerate(row):  # 循环的是i，说明i发生不会同时满足赎回和回售
                """if转股价>净资产或面值，下修"""
                if (row_k > term_down[1]) and (row_k > term_down[2]):
                    # 是否下修,进入回售期？
                    is_down, down_value, mean_20 = self._process_down(term_down, row, k2, arr_time, y_day, row_k, m,
                                                                      term_resell)
                    if is_down:
                        # 'down': [5.5,1, 10, 20, 15, 30, 0.85 * row_k]  # 半年内不下调. 面值，净资产，20交易日，15交易日，30交易日
                        """下修后的转股价格一般要求不低于股东大会前20个交易日股票交易均价和前1交易日的交易均价，并低于之前的转股价格"""
                        row_k = min([max([term_down[1], term_down[2], mean_20, down_value]), row_k])
                        if k2 < (column_len-1):
                            row_mc_k[k2+1:] = row_k  # 触发当天不下调, +1下调
                        # print()
                # 是否赎回
                is_recall, call_end_time, call_value = self._process_recall(term_recall, row, k2, arr_time, y_day,
                                                                            row_k)
                if is_recall:
                    """触发赎回则计算转股价值并折现为现值.第一个触发赎回就break，不再运行后面"""
                    end_v = 100 * (call_value / row_k)  # 转股价值
                    mean_v.append(self.less_optimal_mc_son(k2, end_v, row_mc_k, row, r, y_day))
                    flag += 'break'
                    break
                # 是否触及回售
                is_resell, sell_value = self._process_resell(term_resell, row, k2, arr_time, y_day, row_k, r,
                                                             end_time, term_coupon)
                if is_resell:  # 回售考虑最优停时？
                    print("回售", key, k2, sell_value)
                    mean_v.append(self.less_optimal_mc_son(k2, sell_value, row_mc_k, row, r, y_day))
                    flag += 'break'
                    break
            if flag == '':  # 如果为空说明没有发生赎回和回售
                # 如果整个生命周期都没有触发附加条款，则计算到期价值(或转股或还债)
                kzz_v = 100 * (row[-1] / row_k)  # 转股价值
                end_value = np.max([kzz_v, term_coupon[-1]])  # 包含最后一期利息？
                mean_v.append(self.less_optimal_mc_son(k2, end_value, row_mc_k, row, r, y_day))

        return np.mean(mean_v).round(3)  # 折现到s0

    @staticmethod  # 省时间最优停时蒙特卡洛son,倒推计算最优价值
    def less_optimal_mc_son(k2, end_v, row_mc_k, row, r, y_day):
        discount = 1 / ((1 + r) ** (1 / y_day))
        row = row[:k2 + 1]  # 该行停止时长度集合
        row_v = np.zeros_like(row)
        row_v[-1] = end_v  # 该行停止时价值
        row_mc_k = row_mc_k[:k2 + 1]  # 该行行权价长度=row
        for t in range(row.shape[0] - 2, -1, -1):
            convert = (row[t] / row_mc_k[t]) * 100
            row_v_dis = row_v[t + 1] * discount
            row_v[t] = np.where(convert > row_v_dis, convert, row_v_dis)
        return row_v[0]*discount  # 折现到s0

    # 分析各个路径股价.普通,最优停时蒙特卡洛
    def analyse_mc(self, k, r, m, y_day, term, arr_mc, f, is_include):  # arr_mc=i行m+1列
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
        # 每一天距离到期日的时间(以年为单位)??? # m = inverter*y_day,用交易日244对365进行近似365/244=1.5天
        """这里m不包含当天但包含最后一天，如果包含当天而不包含最后一天则是另外一种算法。不能包含当天同时又包含最后一天。
        第3天=(3-1-2)/244=0/244,还有0天到期。第2天=(3-1-1)/244=1/244,还有1天到期。第3天=(3-1-0)/244=2/244,还有2天到期"""
        arr_time = [(m-1-i) / y_day for i in range(m)]  # m已经包含到期的最后一天，不需要m+1,前面需要m-1。
        if f == 'analyse_general_mc':
            # 普通蒙特卡洛
            general_mc = self.analyse_general_mc(arr_mc, arr_time, k, m, r, term_coupon, term_down, term_recall,
                                                 term_resell, y_day, is_include)
            return general_mc
        if f == 'analyse_optimal_mc':
            # 最优停时蒙特卡洛
            i_k, arr_mc_k = self.analyse_optimal_mc(arr_mc, arr_time, k, m, r, term_coupon, term_down, term_recall,
                                                    term_resell, y_day, is_include)
            son_v = self.optimal_mc_son(i_k, arr_mc_k, arr_mc, r, y_day)
            return son_v
        if f == 'less_analyse_optimal_mc':
            # 省时间最优停时蒙特卡洛
            _v = self.less_analyse_optimal_mc(arr_mc, arr_time, k, m, r, term_coupon, term_down, term_recall,
                                              term_resell, y_day, is_include)
            return _v

    def kzz_mc(self, s0, k, v, r, m, i, y_day, term, f, is_include):
        r = np.log(1 + r)  # 0.029,计算年华利率
        # sn = self.gen_sn(m, i, anti_paths=False)  # return: i行m列的二维数组
        sn = self.gen_sn(m, i, anti_paths=True)
        arr_mc = self._monte_carlo(s0, v, r, m, sn)  # 模拟出各个路径股价=i行m+1列
        # self.get_plt(arr_mc)  # 前1000条模拟路径
        # kzz_v = self.analyse_mc(k, r, m, y_day, term, arr_mc)  # 分析各个路径股价
        # return kzz_v
        return self.analyse_mc(k, r, m, y_day, term, arr_mc, f, is_include)  # 分析各个路径股价

    def monte_carlo_call_option_value(self, s_code, b_code, f):
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            """加列"""
            # sql2 = r"""alter table '{}' add bs_option number(6)""".format(b_code)
            # sql2 = r"""alter table '{}' add general_mc number(6)""".format(b_code)
            # cur.execute(sql2)
            # sql_s_b = "select date,interval from '{}' where date>=?".format(b_code)
            # cur_b = cur.execute(sql_s_b, ('2022-06-13', ))
            sql_s_b = "select date,interval from '{}'".format(b_code)
            cur_b = cur.execute(sql_s_b)
            # row_b = cur_b.fetchmany(200)
            row_b = cur_b.fetchall()
            sql_s_s = "select close,y_v_log from '{}' where date>=?".format(s_code)
            cur_s = cur.execute(sql_s_s, (row_b[0][0][:10], ))
            # row_s = cur_s.fetchmany(400)
            row_s = cur_s.fetchall()
            len_b = len(row_b)
            if len_b != len(row_s):
                print('股票和可转债数据数量不等')

            # s0 = 10
            # k = 83.85  # 大参
            k = 34.94  # 柳药
            # k = 15
            r = 0.03
            # v = 0.4
            # 这里m不包含当天但包含最后一天，如果包含当天而不包含最后一天则是另外一种算法。不能包含当天同时又包含最后一天。
            # m = int(244*5.5)
            # m = 3  # m = inverter*y_day,剩余m天数不包括今天
            i = 3000
            y_day = 244
            # term = {  # 大参
            #     'coupon': [0.3, 0.5, 0.8, 1.3, 1.8, 110],  # 票息
            #     'recall': [5.5, 15, 30, 1.3],  # 赎回条款，一般半年后可以赎回=5.5年，15=30天内有15天大于转股价*1.3
            #     'resell': [2, 30, 30, 0.7, 100],  # 回售条款,最后2年,是否包含最后一期利息？
            #     'down': [5.5, 1, 7.7, 20, 15, 30, 0.85]  # 半年内不下调. 面值，净资产，20交易日，15交易日，30交易日
            # }
            term = {  # 柳药
                'coupon': [0.3, 0.5, 0.8, 1.3, 1.8, 108],  # 票息
                'recall': [5.5, 15, 30, 1.3],  # 赎回条款，一般半年后可以赎回=5.5年，15=30天内有15天大于转股价*1.3
                'resell': [2, 30, 30, 0.7, 100],  # 回售条款,最后2年,是否包含最后一期利息？
                'down': [5.5, 1, 15.256, 20, 15, 30, 0.85]  # 半年内不下调. 面值，净资产，20交易日，15交易日，30交易日
            }

            # kzz_value = self.kzz_mc(s0, k, v, r, m, i, y_day, term, f='analyse_general_mc', is_include=0)
            # print(kzz_value)
            if len_b == len(row_s):
                # val_list = []
                import time
                t_total = 0
                for ii in range(len_b):
                    time_start = time.clock()  # 记录开始时间
                    r_s = row_s[ii]
                    r_b = row_b[ii]
                    r_day = r_b[0][:10]
                    k = self.day_k(k, r_day, b_code)  # 判断行权价
                    self.day_net(r_day, term, b_code)  # 判断净资产
                    """
                    r_b[1] 债券剩余时间,r_s[0]  股票收盘价, r_s[1]股票log波动率
                    """
                    kzz_value = self.kzz_mc(r_s[0], k, r_s[1], r, int(244*float(r_b[1])), i, y_day, term,
                                            f, is_include=0)
                    sql_u = "UPDATE '{}' SET general_mc=(?) WHERE date=(?)".format(b_code)
                    # sql_u = "UPDATE '{}' SET optimal_mc=(?) WHERE date=(?)".format(b_code)

                    """批量更新"""
                    # val_list = [(-5.1, '2020-11-13 00:00:00'), (1.6, '2020-11-16 00:00:00')]
                    cur.executemany(sql_u, [(kzz_value, r_b[0])])
                    conn.commit()
                    time_end = time.clock()  # 记录结束时间
                    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
                    t_total += time_sum
                    print(r_day, ii, (t_total/60), time_sum, kzz_value, k, term['down'][2])
            cur.close()


kzz_o = KzzOptionsMC()
"""analyse_general_mc,analyse_optimal_mc, less_analyse_optimal_mc"""
kzz_o.monte_carlo_call_option_value(s_code='603233', b_code='sh113605',  f='analyse_general_mc')  # 大参
# kzz_o.monte_carlo_call_option_value(s_code='603368', b_code='sh113563', f='analyse_general_mc')  # 柳药
