import sqlite3
from ...tool import mysetting, tools
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import requests
import demjson


# get可转债85-120 into table kzz80-120
def bond_price():
    import re
    with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("DELETE FROM kzz_80_120")
        col = """
        security_code,
        secucode,
        security_name_abbr,
        delist_date,
        listing_date,
        convert_stock_code,
        bond_expire,
        rating,
        value_date,
        issue_year,
        cease_date,
        expire_date,
        pay_interest_day,
        interest_rate_explain,
        bond_combine_code,
        actual_issue_scale,
        issue_price,
        remark,
        par_value,
        issue_object,
        redeem_type,
        execute_reason_hs,
        notice_date_hs,
        notice_date_sh,
        execute_price_hs,
        execute_price_sh,
        record_date_sh,
        execute_start_datesh,
        execute_start_datehs,
        execute_end_date,
        public_start_date,
        bond_start_date,
        security_start_date,
        security_short_name,
        first_per_preplacing,
        online_general_aau,
        online_general_lwr,
        initial_transfer_price,
        transfer_end_date,
        transfer_start_date,
        resale_clause,
        redeem_clause,
        convert_stock_price,
        transfer_price,
        transfer_value,
        current_bond_price,
        transfer_premium_ratio,
        convert_stock_pricehq,
        market,
        resale_trig_price,
        redeem_trig_price,
        pbv_ratio,
        ib_start_date,
        ib_end_date,
        cashflow_date,
        coupon_ir,
        execute_reason_sh,
        paydaynew,
        current_bond_pricenew """
        sql = "select {} from kzz where current_bond_price>? and current_bond_price<?".format(col)
        # sql = "select * from kzz where current_bond_price>? and current_bond_price<?"
        cursor = conn.execute(sql, (85, 120))
        # instead of cursor.description:
        rows = cursor.fetchall()
        # rows = cursor.fetchmany(5)
        # total 66 colum
        col_insert = """
        security_code,
        secucode,
        security_name_abbr,
        delist_date,
        listing_date,
        convert_stock_code,
        bond_expire,
        rating,
        value_date,
        issue_year,

        cease_date,
        expire_date,
        pay_interest_day,
        interest_rate_explain,
        bond_combine_code,
        actual_issue_scale,
        issue_price,
        remark,
        par_value,
        issue_object,

        redeem_type,
        execute_reason_hs,
        notice_date_hs,
        notice_date_sh,
        execute_price_hs,
        execute_price_sh,
        record_date_sh,
        execute_start_datesh,
        execute_start_datehs,
        execute_end_date,

        public_start_date,
        bond_start_date,
        security_start_date,
        security_short_name,
        first_per_preplacing,
        online_general_aau,
        online_general_lwr,
        initial_transfer_price,
        transfer_end_date,
        transfer_start_date,

        resale_clause,
        redeem_clause,
        convert_stock_price,
        transfer_price,
        transfer_value,
        current_bond_price,
        transfer_premium_ratio,
        convert_stock_pricehq,
        market,
        resale_trig_price,

        redeem_trig_price,
        pbv_ratio,
        ib_start_date,
        ib_end_date,
        cashflow_date,
        coupon_ir,
        execute_reason_sh,
        paydaynew,
        current_bond_pricenew,
        one,

        two,three,four,five,six,redeem_price
        """
        sql_insert = "INSERT INTO kzz_80_120 ({}) VALUES(?,?,?,?,?,?,?,?,?,?, ?,?,?,?,?,?,?,?,?,?, ?,?,?,?,?,?,?,?,?,?, ?,?,?,?,?,?,?,?,?,?, ?,?,?,?,?,?,?,?,?,?, ?,?,?,?,?,?,?,?,?,?, ?,?,?,?,?,?)".format(col_insert)
        for gg in rows:
            # print(list(gg))
            interest_rate_explain = gg["interest_rate_explain"]
            redeem_clause = gg["redeem_clause"]  # 赎回条款
            interest = []
            for num_year in ["第一年", "第二年", "第三年", "第四年", "第五年", "第六年"]:
                interest += get_interest(interest_rate_explain, num_year)
            # print(interest)
            try:
                search_span = re.search(r'(含最.*利息)', redeem_clause).span()  # （.*表示多个任意字符）（.表示任意字符）
                # print("dr2324", search_span[0])
                # print("www", redeem_clause[search_span[0]-7: search_span[0]])
                yy = [float(s) for s in re.findall(r'-?\d+\.?\d*', redeem_clause[search_span[0]-7: search_span[0]])]  # 字符串中取浮点数，int和负数
                y = yy[0]
                if y < 100:
                    yy[0] = y + 100
                # print(yy)
                interest += yy
            except:
                interest += [east_kzz_redeem(code=gg["security_code"])]
                print(gg["security_code"] + "到期赎回值not", redeem_clause)
            gg = list(gg)
            gg += interest
            # print(gg)
            cur.execute(sql_insert, gg)
        cursor.close()
        cur.close()


# 可转债年利率
def get_interest(g, ii):
    if ii in g:
        # print(g)
        ggg = g.find(ii)
        yy = [float(s) for s in re.findall(r'-?\d+\.?\d*', g[(ggg+3): (ggg+7)])]  # 字符串中取浮点数，int和负数
        # print(yy)
        return yy
    else:
        print("not", ii)
        return [""]


# 读东财可转债的到期赎回价
def east_kzz_redeem(code="123140"):
    # print("11h", code.startswith("11"))
    if code.startswith("11"):
        code = "1."+code
    elif code.startswith("12"):
        code = "0."+code
    else:
        print("读东财可转债的到期赎回价,代码有误", code)
    net = """https://push2.eastmoney.com/api/qt/stock/get?invt=2&fltt=1&cb=&fields=f264%2Cf263%2Cf262%2Cf267%2Cf265%2Cf268%2Cf433%2Cf426%2Cf427%2Cf154%2Cf428%2Cf152%2Cf430%2Cf431%2Cf432%2Cf424&secid={}&ut=fa5fd1943c7b386f172d6893dbfba10b&wbp2u=%7C0%7C0%7C0%7Cweb&_=1649465088820"""
    # dragon_t = requests.get(net)
    dragon_t = requests.get(net.format(code))
    dragon_tiger = dragon_t.text
    print(dragon_tiger)
    if dragon_t.status_code == 200 and dragon_tiger:
        dragon_tiger = demjson.decode(dragon_tiger)
        # d = dragon_tiger.get('result', '')
        dr = dragon_tiger.get('data', '')
        if dr:
            redeem = dr.get("f432", "")/100
            # print("redeem", redeem)
            return redeem
        else:
            print("债券数据not", code)
            return ""
    time.sleep(1.5)


# my可转债纯债值   默认只更新my_bond_value
def my_interest_value(f="my"):
    with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
        cur = conn.cursor()
        if f == "my":  # 默认只更新my_bond_value
            cur.execute("update kzz_80_120 set my_bond_value=''")
            sql_update = "update kzz_80_120 set my_bond_value=? where security_code=?"
        else:
            cur.execute("update kzz_80_120 set east_pure_bond_value='',my_bond_value=''")
            sql_update = "update kzz_80_120 set east_pure_bond_value=?,my_bond_value=? where security_code=?"
        sql = "select security_code,one,two,three,four,five,six,redeem_price from kzz_80_120"
        cur.execute(sql)
        dat = cur.fetchall()
        # dat = cur.fetchmany(20)
        print("len(dat)", len(dat))
        i = 0
        for da in dat:
            if f != "my":  # 默认只更新my_bond_value
                east_pure_bond_value = east_kzz_only_bond_value(code=da[0])
            # print("east_pure_bond_value", east_pure_bond_value)
            new_list = list(filter(None, da[1:6]))
            i += 1
            if i % 15 == 0:
                print(i, da[0])
            if len(new_list) == 5:
                if da[7]:
                    if da[6]:
                        pp = da[7]/pow(1.03, 6)
                        if f == "my":  # 默认只更新my_bond_value
                            cur.execute(sql_update, (round(pp, 3), da[0]))
                        else:
                            cur.execute(sql_update, (east_pure_bond_value, round(pp, 3), da[0]))
                    else:
                        pp = da[7] / pow(1.03, 5)
                        if f == "my":  # 默认只更新my_bond_value
                            cur.execute(sql_update, (round(pp, 3), da[0]))
                        else:
                            cur.execute(sql_update, (east_pure_bond_value, round(pp, 3), da[0]))
                if da[6] and not da[7]:
                    p = da[1] / pow(1.03, 1) + da[2] / pow(1.03, 2) + da[3] / pow(1.03, 3) + da[4] / pow(1.03, 4) + da[5] / pow(1.03, 5) + da[6] / pow(1.03, 6) + 100 / pow(1.03, 6)
                    # print(p)
                    if f == "my":  # 默认只更新my_bond_value
                        cur.execute(sql_update, (round(p, 3), da[0]))
                    else:
                        cur.execute(sql_update, (east_pure_bond_value, round(p, 3), da[0]))
                if da[5] and not da[6] and not da[7]:
                    p = da[1] / pow(1.03, 1) + da[2] / pow(1.03, 2) + da[3] / pow(1.03, 3) + da[4] / pow(1.03, 4) + da[5] / pow(1.03, 5) + 100 / pow(1.03, 5)
                    print("five year", p)
                    if f == "my":  # 默认只更新my_bond_value
                        cur.execute(sql_update, (round(p, 3), da[0]))
                    else:
                        cur.execute(sql_update, (east_pure_bond_value, round(p, 3), da[0]))
            else:
                print("have problem", da)
        print("bond number", i)
        cur.close()


# 读东财可转债的纯债价值
def east_kzz_only_bond_value(code="113640"):
    net = """https://datacenter-web.eastmoney.com/api/data/get?callback=&sty=ALL&token
    =894050c76af8597a853f5b408b759f5d&st=date&sr=1&source=WEB&type=RPTA_WEB_KZZ_LS&filter=
    (zcode%3D%22{}%22)&p=1&ps=8000&_=1649335592927"""
    # dragon_t = requests.get(net)
    dragon_t = requests.get(net.format(code))
    dragon_tiger = dragon_t.text
    # print(dragon_tiger)
    if dragon_t.status_code == 200 and dragon_tiger:
        dragon_tiger = demjson.decode(dragon_tiger)
        d = dragon_tiger.get('result', '')
        dr = d.get('data', '')
        if dr:
            pure_bond_value = dr[-1].get("PUREBONDVALUE", "")
            # print("purebondvalue", pure_bond_value)
            # print("dr", dr[-1])
            if not pure_bond_value:
                try:
                    pure_bond_value = dr[-2].get("PUREBONDVALUE", "")
                    # print("purebondvalue22", pure_bond_value)
                    # print("dr", dr[-2])
                    # print("纯债值为空或时间过12点", dr[-1])
                except:
                    print("刚上市债券？", dr[-1])
            return round(pure_bond_value, 3)
        else:
            print("债券数据err", code)
    time.sleep(1.5)


# 更新债券剩余天数和年数
def update_bond_day_year():
    with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
        cur = conn.cursor()
        sql = """select security_code, expire_date, redeem_price, current_bond_price from kzz_80_120"""
        cur.execute(sql)
        # dat = cur.fetchall()
        dat = cur.fetchmany(3)
        # 格式化成2016-03-20形式
        today_day = time.strftime("%Y-%m-%d", time.localtime())
        for dd in dat:
            print(dd)
            expire_date = dd[1]
            if expire_date:
                if len(expire_date) > 10:
                    expire_date = expire_date[:10]
                day_diff = tools.interval_days(today_day, expire_date)
                print("两个日期的间隔天数：{} ".format(day_diff))
                year_diff = day_diff / 365
                print("两个日期的间隔天数/365：{} ".format(year_diff))
            else:
                day_diff = ""
                year_diff = ""
                print("not day", dd)
            redeem_price = dd[2]
            current_bond_price = dd[3]
            if redeem_price and current_bond_price and year_diff:
                # 债券年收益率
                year_bond_yield = (redeem_price/current_bond_price)**(1/year_diff)
                print("债券年收益率：{} ".format(year_bond_yield - 1))
            else:
                year_bond_yield = ""
                print("not redeem_price, year_diff", dd)
        cur.close()


# 获取可转债对应股票涨幅
def get_bond_stock_up():
    import baostock as bs
    lg = bs.login()
    with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM kzz_date_frequent")
        sql = "select DISTINCT convert_stock_code from kzz_80_120"
        cur.execute(sql)
        dat = cur.fetchall()
        # dat = cur.fetchmany(2)
        # 格式化成2016-03-20形式
        end_time = time.strftime("%Y-%m-%d", time.localtime())
        # print(end_time)
        start_year = int(time.strftime('%Y', time.localtime(time.time()))) - 2
        month_day = time.strftime('%m-%d', time.localtime(time.time()))
        start_time = '{}-{}'.format(start_year, month_day)
        # print(start_time)
        sql_insert = "INSERT INTO kzz_date_frequent VALUES(?,?,?)"
        iii = 0
        for dd in dat:
            iii += 1
            k_data = tools.for_history_k_data(code=dd[0], start_date=start_time, end_date=end_time, col="only_up_rate", bs=bs)
            # print(k_data[:2])
            # print(dd[0])
            # print(iii, len(k_data))
            for gg in k_data:
                gg.append(dd[0])
                # print(gg)
                cur.execute(sql_insert, gg)
        cur.close()
    bs.logout()


# 计算可转债对应stock of 标准差
def kzz_stock_standard_deviation():
    with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
        cur = conn.cursor()
        sql_updat = """update kzz_80_120 set mean_up = '',mean_up_year = '',standard_deviation = '',
        standard_deviation_total = '',standard_deviation_year = '',standard_deviation_year_total = ''"""
        cur.execute(sql_updat)
        sql = "select convert_stock_code from kzz_80_120"
        cur.execute(sql)  # 取股票代码
        dat = cur.fetchall()
        # dat = cur.fetchmany(2)
        # print(dat)
        sql_select = "select up_rate from kzz_date_frequent where code=?"
        sql_update = """update kzz_80_120 set mean_up=?, mean_up_year=?,
        standard_deviation=?, standard_deviation_total=?,
        standard_deviation_year=?, standard_deviation_year_total=? where convert_stock_code=?"""
        for code_tu in dat:
            cur.execute(sql_select, (code_tu[0],))
            dat2 = cur.fetchall()
            # dat2 = cur.fetchmany(5)
            # print(dat2)
            up_rate_list = []  # 股票涨幅list
            for up_rate_tu in dat2:
                try:
                    up_rate_tu1 = float(up_rate_tu[0])
                except:
                    up_rate_tu1 = 0
                    print(code_tu[0], up_rate_tu)
                    print(dat2)
                up_rate_list.append(up_rate_tu1)
            len_up_rate_list = len(up_rate_list)  # 股票涨幅list的个数
            # print(up_rate_list[(len_up_rate_list-245):])
            # 画数据的概率分布
            p = ""
            if p:
                plot_probability_distribution(data=up_rate_list[(len_up_rate_list-245):])
            if (len_up_rate_list >= 445) and (len_up_rate_list < 495):  # 涨幅数据为接近2年
                up_rate_list_year = up_rate_list[(len_up_rate_list - 245):]  # 年(243)股票涨幅list
                # 求均值
                arr_mean = np.mean(up_rate_list)
                arr_mean_year = np.mean(up_rate_list_year)
                # 求方差
                # arr_var = np.var(up_rate_list)
                # arr_var_year = np.var(up_rate_list_year)
                # 求标准差
                """1、在统计学中，标准差分为两种：
                （1）总体标准差：标准差公式根号内除以n，是有偏的。
                （2）样本标准差：标准差公式根号内除以n-1，是无偏的。
                2、pandas与numpy在计算标准差时的区别
                （1）numpy在numpy中计算标准差时，括号内要指定ddof的值，ddof表示自由度，当ddof=0时计算的是总体标准差；
                当ddof=1时计算的是样本标准差，当不为ddof设置值时，其默认为总体标准差。
                （2）pandas
             在使用pandas计算标准差时，其与numpy的默认情况是相反的，在默认情况下，pandas计算的标准差为样本标准差。"""
                arr_std = np.std(up_rate_list, ddof=0)
                arr_std_total = arr_std * (len_up_rate_list ** 0.5)  # 总波动
                arr_std_year = np.std(up_rate_list_year, ddof=0)
                arr_std_year_total = arr_std_year*(len(up_rate_list_year)**0.5)  # year总波动
                # print("平均值为：%f" % arr_mean)
                # print("平均值2为：%f" % arr_mean_year)
                # print("方差为：%f" % arr_var)
                # print("方差2为：%f" % arr_var_year)
                # print("标准差为:%f" % arr_std)
                # print("标准差2为:%f" % arr_std_year)
                # print("总波动为:%f" % (arr_std*(len_up_rate_list**0.5)))
                # print("总波动2为:%f" % (arr_std_year*(len(up_rate_list_year)**0.5)))
                # print("标准差为:%f" % 6**0.5)
                f = [round(arr_mean, 2), round(arr_mean_year, 2), round(arr_std, 2), round(arr_std_total, 2), round(arr_std_year, 2), round(arr_std_year_total, 2), code_tu[0]]
                cur.execute(sql_update, f)
            elif (len_up_rate_list >= 245) and (len_up_rate_list < 445):
                up_rate_list_year = up_rate_list[(len_up_rate_list - 245):]
                print(code_tu[0], len_up_rate_list)
                # 求均值
                arr_mean_year = np.mean(up_rate_list_year)
                # 求方差
                arr_std_year = np.std(up_rate_list_year, ddof=0)
                arr_std_year_total = arr_std_year * (len(up_rate_list_year) ** 0.5)  # year总波动
                # print("平均值2为：%f" % arr_mean_year)
                # print("方差2为：%f" % arr_var_year)
                # print("标准差2为:%f" % arr_std_year)
                # print("总波动2为:%f" % (arr_std_year * (len(up_rate_list_year) ** 0.5)))
                cur.execute(sql_update, ("", round(arr_mean_year, 2), "", "", round(arr_std_year, 2), round(arr_std_year_total, 2), code_tu[0]))
            else:
                print(code_tu[0], len_up_rate_list)
        cur.close()


# 计算可转债对应stock期权价值
def kzz_stock_option_value():
    from scipy import stats
    # 格式化成2016-03-20形式
    today_day = time.strftime("%Y-%m-%d", time.localtime())
    with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
        cur = conn.cursor()
        cur.execute("update kzz_80_120 set option_value = '',bond_total_value = ''")
        """（1）C：当前看涨期权的价值，这个是正股的期权价值（最后要转成可转债的）。
        （2）S：正股当前的价格，convert_stock_price
        （3）X：期权的行权价，就是可转债的转股价格，transfer_price
        （4）T：当前到转债到期日的时间，换算成以年为单位。expire_date
        当前可转债到期日还有452天，那么T=452/360=1.26。分母中你也可以取365，习惯上是360。
        （5）r：无风险利率，且为连续复利下的年化收益率。
        （6）σ：正股的年化收益率标准差
        （7）N(d)：标准正态分布<=d的累积概率。比如标准正态分布的均值为0，那么N(0)=0.5
        """
        sql = """select security_code, convert_stock_code, convert_stock_price,transfer_price, expire_date,
        standard_deviation_year_total,my_bond_value from kzz_80_120 where standard_deviation_year_total !=''"""
        cur.execute(sql)
        dat = cur.fetchall()
        # dat = cur.fetchmany(3)
        for dd in dat:
            # print(dd)
            if dd[4]:
                if len(dd[4]) > 10:
                    end_day = dd[4][:10]
            else:
                print("not day", dd)
            day_diff = tools.interval_days(today_day, end_day)
            # print("两个日期的间隔天数：{} ".format(day_diff))
            # 年波动dd[5]*根号t
            year_up = dd[5]/100
            year_num = day_diff / 365
            # print("两个日期的间隔天数/365：{} ".format(year_num))
            year_variance_t = year_up*(year_num**0.5)
            # print("两个日期的间隔年数的根号：{} ".format(year_num**0.5))
            # print("year_up*(year_num**0.5)：{} ".format(year_variance_t))
            convert_stock_price = dd[2]  # S：正股当前的价格
            transfer_price = dd[3]  # 期权的行权价，就是可转债的转股价格
            # print(convert_stock_price, transfer_price)
            try:
                #  S：正股当前的价格/X：期权的行权价，就是可转债的转股价格, 取e为底对数
                log2 = math.log((convert_stock_price/transfer_price), math.e)
                # print("dd[2]", dd[2])
                # print("dd[3]", dd[3])
                # print("dd[2]/dd[3]", dd[2]/dd[3])
                # print("log", log2)
                d1 = (log2 + (0.03 + 0.5*year_up*year_up)*year_num)/year_variance_t
                # print("d1", d1)
                d2 = d1 - year_variance_t
                # print("d2", d2)
                # norm pdf用来获得正太分布的概率密度函数probility density function。
                # 其实 概率密度函数值 即为 概率在该点的变化率.千万不要误认为：概率密度函数值是 该点的概率.
                # prob = stats.norm.pdf(-0.3, 0, 1)
                # print("prob", prob)
                # norm.cdf (Cumulative Distribution Function) 计算累积标准正态分布函数
                # 概率分布函数" 的定义，就会发现它就是概率函数中的概率累加
                cdf1 = stats.norm.cdf(d1, 0, 1)
                cdf2 = stats.norm.cdf(d2, 0, 1)
                # print("cdf1", cdf1)
                # print("cdf2", cdf2)
                e_power = 2.718**(-0.03*year_num)
                # print(e_power)
                # C：当前看涨期权的价值，这个是正股的期权价值（最后要转成可转债的）
                c = convert_stock_price*cdf1 - transfer_price*e_power*cdf2
                # print("c", c)
                option_value = (100/transfer_price)*c
                # print("option_value", option_value)
                # print("dd[6]", dd[6])
                bond_total_value = option_value + dd[6]
                # print("bond_total_value", bond_total_value)
                lis = [round(option_value, 3), round(bond_total_value, 3), dd[0]]
                cur.execute("update kzz_80_120 set option_value=?,bond_total_value=? where security_code=?", lis)
            except:
                print("当前股价或转股价异常", dd)
        cur.close()


# 画数据的概率分布,概率密度图
def plot_probability_distribution(data=""):
    if not data:
        data = np.random.normal(size=200)
    # plot histogram
    plt.subplot(221)  # plt.subplot(nrows, ncols, index, **kwargs)
    plt.hist(data)  # plt.hist（）函数绘制直方图
    # obtain histogram data
    plt.subplot(222)
    hist, bin_edges = np.histogram(data)  # np.histogram()直方图分布
    plt.plot(hist)
    plt.subplot(223)
    hist, bin_edges = np.histogram(data)  # 概率密度图
    cdf = np.cumsum(hist)
    plt.plot(cdf)
    plt.show()
