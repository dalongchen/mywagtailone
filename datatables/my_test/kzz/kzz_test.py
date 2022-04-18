import sqlite3
from ...tool import mysetting, tools
import re
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import requests
import demjson
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s: %(message)s')


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
            except:  # 到期赎回值为空则调东财的 east_kzz_redeem
                interest += [east_kzz_redeem(code=gg["security_code"])]
                # print(gg["security_code"] + "到期赎回值not", redeem_clause)
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
    else:  # 第6年没有返回空list
        # print("not", ii)
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
    # print(dragon_tiger)
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


# 更新债券剩余天数,年数,my债券年收益率，是否包含最后一期利息 # include="1"为不包含最后一期利息
def update_bond_day_year_yield():
    with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
        cur = conn.cursor()
        cur.execute("update kzz_80_120 set day_diff='',year_diff='',year_bond_yield='',include=''")
        sql = """select security_code,expire_date,redeem_price,current_bond_price,execute_price_hs,
        execute_price_sh from kzz_80_120"""
        cur.execute(sql)
        dat = cur.fetchall()
        # dat = cur.fetchmany(10)
        # 格式化成2016-03-20形式
        today_day = time.strftime("%Y-%m-%d", time.localtime())
        sql_update = """update kzz_80_120 set day_diff=?,year_diff=?,year_bond_yield=?,include=?
        where security_code=?"""
        # 赎回价是否包含最后一期利息. # include="1"为不包含最后一期利息
        sql_select = """select security_code FROM kzz_80_120 WHERE INSTR(redeem_clause, '(不含最') >0"""
        row2 = cur.execute(sql_select).fetchall()
        # print("row2", row2)
        list2 = [rr[0] for rr in row2]
        # print("list2", list2)
        for dd in dat:
            # print("dd", dd)
            security_code = dd[0]
            expire_date = dd[1]
            if expire_date:
                if len(expire_date) > 10:
                    expire_date = expire_date[:10]
                day_diff = tools.interval_days(today_day, expire_date)
                # print("两个日期的间隔天数：{} ".format(day_diff))
                year_diff = day_diff / 365
                # print("两个日期的间隔天数/365：{} ".format(year_diff))
            else:
                day_diff = ""
                year_diff = ""
                print("not day", dd)
            redeem_price = dd[2]
            current_bond_price = dd[3]
            execute_price_hs = dd[4]
            execute_price_sh = dd[5]
            if current_bond_price and year_diff:
                if execute_price_hs:
                    year_bond_yield = (execute_price_hs/current_bond_price)**(1/year_diff)
                elif execute_price_sh:
                    year_bond_yield = (execute_price_sh/current_bond_price)**(1/year_diff)
                elif redeem_price:
                    # 债券年收益率
                    year_bond_yield = (redeem_price/current_bond_price)**(1/year_diff)
                    # print("债券年收益率：{} ".format(year_bond_yield - 1))
                else:
                    year_bond_yield = ""
                    print("not redeem_price, year_diff", dd)
            else:
                year_bond_yield = ""
                print("2 not redeem_price, year_diff", dd)
            add_data = [round(day_diff, 3), round(year_diff, 3), round((year_bond_yield - 1), 3), "", security_code]
            if security_code in list2:   # include="1"为不包含最后一期利息
                add_data = [round(day_diff, 3), round(year_diff, 6), round((year_bond_yield - 1)*100, 2), "1", security_code]
            cur.execute(sql_update, add_data)
        cur.close()


# my可转债纯债值   默认只更新my_bond_value
def my_interest_value(f="my"):
    treasury_interest = 1.025  # 国债利率
    with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
        cur = conn.cursor()
        if f == "my":  # 默认只更新my_bond_value
            cur.execute("update kzz_80_120 set my_bond_value=''")
            sql_update = "update kzz_80_120 set my_bond_value=? where security_code=?"
        else:
            cur.execute("update kzz_80_120 set east_pure_bond_value='',my_bond_value=''")
            sql_update = "update kzz_80_120 set east_pure_bond_value=?,my_bond_value=? where security_code=?"
        sql = """select security_code,redeem_price,execute_price_hs,execute_price_sh,year_diff from kzz_80_120"""
        cur.execute(sql)
        dat = cur.fetchall()
        # dat = cur.fetchmany(2)
        # print("len(dat)", len(dat))
        i = 0
        for da in dat:
            if f != "my":  # 默认只更新my_bond_value
                east_pure_bond_value = east_kzz_only_bond_value(code=da[0])
            i += 1
            security_code = da[0]
            if i % 15 == 0:
                print(i, security_code)
            redeem_price = da[1]
            # execute_price_hs = da[2]
            execute_price_sh = da[3]
            year_diff = da[4]
            # print(year_diff, redeem_price)
            # if execute_price_hs:
            #     my_bond_value = execute_price_hs/pow(1.03, year_diff)
            if execute_price_sh:
                my_bond_value = execute_price_sh / pow(treasury_interest, year_diff)
            elif redeem_price:
                # print(pow(1.03, year_diff), pow(1.03, 2.5))
                my_bond_value = redeem_price / pow(treasury_interest, year_diff)
            else:
                print("数据有eer", security_code)
            # print(my_bond_value)
            if f == "my":  # 默认只更新my_bond_value
                cur.execute(sql_update, (round(my_bond_value, 3), security_code))
            else:
                cur.execute(sql_update, (east_pure_bond_value, round(my_bond_value, 3), security_code))
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


# 添加未付利息贴现，修正my可转债纯债值   更新 my_bond_revise,到期年化收益 my_bond_revise_rate，到期估值 my_bond_revise_expire
def my_bond_value_revise():
    with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
        cur = conn.cursor()
        cur.execute("update kzz_80_120 set my_bond_revise=''")
        sql_update = """update kzz_80_120 set my_bond_revise=?,my_bond_revise_expire=?,my_bond_revise_rate=?
        where security_code=?"""
        # bond_expire 债券期限  execute_price_sh 赎回价  coupon_ir 当期利息
        sql = """select security_code,one,two,three,four,five,six,current_bond_price,pay_interest_day,coupon_ir,
        execute_price_sh,year_diff,include,redeem_price from kzz_80_120"""
        cur.execute(sql)
        dat = cur.fetchall()
        # dat = cur.fetchmany(20)
        # print("len(dat)", len(dat))
        # 格式化成2016-03-20形式
        # today_day = time.strftime("%Y-%m-%d", time.localtime())
        for da in dat:
            security_code = da[0]
            interest_period = da[1:7]
            current_bond_price = da[7]  # 债券当前价格
            # pay_interest_day = da[8]
            coupon_ir = da[9]
            execute_price_sh = da[10]
            year_diff = da[11]
            include = da[12]  # 期满赎回价是否包含最后一期利息
            redeem = da[13]  # 期满赎回价
            one = interest_period[0]*0.8
            two = interest_period[1]*0.8
            three = interest_period[2]*0.8
            four = interest_period[3]*0.8
            five = interest_period[4]*0.8
            six = interest_period[5]*0.8
            treasury_interest = 1.025  # 国债利率
            try:
                redeem_discount = redeem/pow(treasury_interest, year_diff)  # 到期赎回价贴现
                if six:
                    six_discount = six/pow(treasury_interest, year_diff)  # 第6年附息按剩余天（年）数贴现

                five_discount = five/pow(treasury_interest, year_diff)  # 第5年附息按剩余天（年）数贴现,this only 5年期债券
                five_discount_1 = five/pow(treasury_interest, (year_diff - 1))  # 第5年附息按剩余天（年）数贴现,this only 6年期债券

                four_discount_1 = four/pow(treasury_interest, (year_diff - 1))  # 小数后缀为5年期
                four_discount_2 = four/pow(treasury_interest, (year_diff - 2))  # 大数后缀为6年期

                three_discount_2 = three/pow(treasury_interest, (year_diff - 2))
                three_discount_3 = three/pow(treasury_interest, (year_diff - 3))

                two_discount_3 = two/pow(treasury_interest, (year_diff - 3))
                two_discount_4 = two/pow(treasury_interest, (year_diff - 4))

                one_discount_4 = one/pow(treasury_interest, (year_diff-4))
                one_discount_5 = one/pow(treasury_interest, (year_diff-5))

                one_rate_5 = one*pow(treasury_interest, 5)  # 大数后缀为6年期
                one_rate_4 = one*pow(treasury_interest, 4)  # 小数后缀为5年期

                two_rate_4 = two*pow(treasury_interest, 4)
                two_rate_3 = two*pow(treasury_interest, 3)

                three_rate_3 = three*pow(treasury_interest, 3)
                three_rate_2 = three*pow(treasury_interest, 2)

                four_rate_2 = four*pow(treasury_interest, 2)
                four_rate_1 = four*pow(treasury_interest, 1)

                five_rate_1 = five*pow(treasury_interest, 1)
            except:
                logging.info("空值错误：%s" % str(da))

            if execute_price_sh:  # 赎回价
                my_bond_revise = execute_price_sh / pow(treasury_interest, year_diff)
            else:
                try:  # 如果当前利息率coupon_ir没有在利息表里，抛异常
                    interest_period = list(interest_period)
                    # print("interest_period", interest_period)
                    interest_index = interest_period.index(coupon_ir)  # 从列表中找出某个值第一个匹配项的索引位置
                    if interest_index == 0:  # 第一年利息未付
                        if include == "":  # 表示到期赎回价格包括最后一期利息
                            if six:  # if有6年期
                                my_bond_revise = redeem_discount + five_discount_1 + four_discount_2 \
                                                 + three_discount_3+two_discount_4+one_discount_5
                                redeem_total = redeem+five_rate_1+four_rate_2+three_rate_3+two_rate_4+one_rate_5
                            else:  # if没有6年期，只有5年期
                                my_bond_revise = redeem_discount + four_discount_1 + three_discount_2 \
                                                 + two_discount_3 + one_discount_4
                                redeem_total = redeem + four_rate_1 + three_rate_2 + two_rate_3 + one_rate_4
                        elif include == "1":  # 表示到期赎回价格不包括最后一期利息
                            if six:  # if有6年期
                                my_bond_revise = redeem_discount + six_discount + five_discount_1 + four_discount_2 \
                                                 + three_discount_3 + two_discount_4 + one_discount_5
                                redeem_total = redeem+six+five_rate_1+four_rate_2+three_rate_3+two_rate_4+one_rate_5
                            else:  # if没有6年期，只有5年期
                                my_bond_revise = redeem_discount + five_discount + four_discount_1 \
                                                 + three_discount_2 + two_discount_3 + one_discount_4
                                redeem_total = redeem + five + four_rate_1 + three_rate_2 + two_rate_3 + one_rate_4
                        else:
                            logging.info("是否包含最后一期的值有误: %s" % interest_index+":"+str(da))
                    elif interest_index == 1:  # 第2年利息未付
                        if include == "":  # 表示到期赎回价格包括最后一期利息
                            if six:  # if有6年期
                                my_bond_revise = redeem_discount + five_discount_1 + four_discount_2 \
                                                 + three_discount_3 + two_discount_4
                                redeem_total = redeem + five_rate_1 + four_rate_2 + three_rate_3 + two_rate_4
                            else:  # if没有6年期，只有5年期
                                my_bond_revise = redeem_discount + four_discount_1 + three_discount_2 \
                                                 + two_discount_3
                                redeem_total = redeem + four_rate_1 + three_rate_2 + two_rate_3
                        elif include == "1":  # 表示到期赎回价格不包括最后一期利息
                            if six:  # if有6年期
                                my_bond_revise = redeem_discount + six_discount + five_discount_1 + four_discount_2 \
                                                 + three_discount_3 + two_discount_4
                                redeem_total = redeem + six + five_rate_1 + four_rate_2 + three_rate_3 + two_rate_4
                            else:  # if没有6年期，只有5年期
                                my_bond_revise = redeem_discount + five_discount + four_discount_1 \
                                                 + three_discount_2 + two_discount_3
                                redeem_total = redeem + five + four_rate_1 + three_rate_2 + two_rate_3
                        else:
                            logging.info("是否包含最后一期的值有误: %s" % interest_index+":"+str(da))
                    elif interest_index == 2:  # 第3年利息未付
                        if include == "":  # 表示到期赎回价格包括最后一期利息
                            if six:  # if有6年期
                                my_bond_revise = redeem_discount + five_discount_1 + four_discount_2 \
                                                 + three_discount_3
                                redeem_total = redeem + five_rate_1 + four_rate_2 + three_rate_3
                            else:  # if没有6年期，只有5年期
                                my_bond_revise = redeem_discount + four_discount_1 + three_discount_2
                                redeem_total = redeem + four_rate_1 + three_rate_2
                        elif include == "1":  # 表示到期赎回价格不包括最后一期利息
                            if six:  # if有6年期
                                my_bond_revise = redeem_discount + six_discount + five_discount_1 + four_discount_2 \
                                                 + three_discount_3
                                redeem_total = redeem + six + five_rate_1 + four_rate_2 + three_rate_3
                            else:  # if没有6年期，只有5年期
                                my_bond_revise = redeem_discount + five_discount + four_discount_1 \
                                                 + three_discount_2
                                redeem_total = redeem + five + four_rate_1 + three_rate_2
                        else:
                            logging.info("是否包含最后一期的值有误: %s" % interest_index+":"+str(da))
                    elif interest_index == 3:  # 第4年利息未付
                        if include == "":  # 表示到期赎回价格包括最后一期利息
                            if six:  # if有6年期
                                my_bond_revise = redeem_discount + five_discount_1 + four_discount_2
                                redeem_total = redeem + five_rate_1 + four_rate_2
                            else:  # if没有6年期，只有5年期
                                my_bond_revise = redeem_discount + four_discount_1
                                redeem_total = redeem + four_rate_1
                        elif include == "1":  # 表示到期赎回价格不包括最后一期利息
                            if six:  # if有6年期
                                my_bond_revise = redeem_discount + six_discount + five_discount_1 + four_discount_2
                                redeem_total = redeem + six + five_rate_1 + four_rate_2
                            else:  # if没有6年期，只有5年期
                                my_bond_revise = redeem_discount + five_discount + four_discount_1
                                redeem_total = redeem + five + four_rate_1
                        else:
                            logging.info("是否包含最后一期的值有误: %s" % interest_index+":"+str(da))
                    elif interest_index == 4:  # 第5年利息未付
                        if include == "":  # 表示到期赎回价格包括最后一期利息
                            if six:  # if有6年期
                                my_bond_revise = redeem_discount + five_discount_1
                                redeem_total = redeem + five_rate_1
                            else:  # if没有6年期，只有5年期
                                my_bond_revise = redeem_discount
                                redeem_total = redeem
                        # six_discount = six / pow(treasury_interest, year_diff)  # 第6年附息按剩余天（年）数贴现
                        # five_discount = five / pow(treasury_interest, year_diff)  # 第5年附息按剩余天（年）数贴现,this only 5年期债券
                        # five_discount_1 = five / pow(treasury_interest, (year_diff - 1))
                        elif include == "1":  # 表示到期赎回价格不包括最后一期利息
                            if six:  # if有6年期
                                my_bond_revise = redeem_discount + six_discount + five_discount_1
                                redeem_total = redeem + six + five_rate_1
                            else:  # if没有6年期，只有5年期
                                my_bond_revise = redeem_discount + five_discount
                                redeem_total = redeem + five
                        else:
                            logging.info("是否包含最后一期的值有误: %s" % interest_index+":"+str(da))
                    elif interest_index == 5:  # 第6年利息未付
                        if include == "":  # 表示到期赎回价格包括最后一期利息
                            # 这里无需判断six是否存在，因为5年期已经退市，没有交易价格
                            # 且如果第6期为空值，则interest_index ！= "5"
                            my_bond_revise = redeem_discount
                            redeem_total = redeem
                        elif include == "1":  # 表示到期赎回价格不包括最后一期利息
                            my_bond_revise = redeem_discount + six_discount
                            redeem_total = redeem + six
                        else:
                            logging.info("是否包含最后一期的值有误: %s" % interest_index+":"+str(da))
                    else:
                        logging.info("interest_index利息索引有误: %s" % interest_index+":"+str(da))
                    # 计算加各年利息复利后的年平均收益率
                    # print(redeem_total / current_bond_price)
                    my_bond_revise_rate = pow(redeem_total / current_bond_price, 1 / year_diff) - 1
                except:
                    logging.info("2,利息索引有误: %s" % str(da))
            # print("my_bond_revise", my_bond_revise)
            # print("my_bond_revise_rate", my_bond_revise_rate)
            cur.execute(sql_update, (round(my_bond_revise, 3), round(redeem_total, 3), round(my_bond_revise_rate*100, 2), security_code))
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
            if iii % 15 == 0:
                print(iii, dd[0])
            k_data = tools.for_history_k_data(code=dd[0], start_date=start_time, end_date=end_time, col="only_up_rate", bs=bs)
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
            i = 0
            for up_rate_tu in dat2:
                try:
                    up_rate_tu1 = float(up_rate_tu[0])
                except:
                    i += 1
                    up_rate_tu1 = 0
                up_rate_list.append(up_rate_tu1)
            if i > 1:
                print("有空值" + str(i) + "个", code_tu[0] + str(dat2))
            len_up_rate_list = len(up_rate_list)  # 股票涨幅list的个数
            # print(up_rate_list[(len_up_rate_list-245):])

            p = ""   # 画数据的概率分布
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
                # print(code_tu[0], len_up_rate_list)
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
                print("数据少于245" + code_tu[0], len_up_rate_list)
        cur.close()


# 计算可转债对应stock期权价值
def kzz_stock_option_value():
    from scipy import stats
    with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
        cur = conn.cursor()
        cur.execute("update kzz_80_120 set option_value = '',bond_total_value = '',bond_total_premium=''")
        """（1）C：当前看涨期权的价值，这个是正股的期权价值（最后要转成可转债的）。
        （2）S：正股当前的价格，convert_stock_price
        （3）X：期权的行权价，就是可转债的转股价格，transfer_price
        （4）T：当前到转债到期日的时间，换算成以年为单位。year_diff
        当前可转债到期日还有452天，那么T=452/360=1.26。分母中你也可以取365，习惯上是360。
        （5）r：无风险利率，且为连续复利下的年化收益率。
        （6）σ：正股的年化收益率标准差
        （7）N(d)：标准正态分布<=d的累积概率。比如标准正态分布的均值为0，那么N(0)=0.5
        N(d2)：期权被执行的概率
        s*N(d1)：是期权与当前股票price之间的变化关系
        standard_deviation_year_total年华波动率
        convert_stock_price股票当前价
        transfer_price换股价
        """
        sql = """select security_code, convert_stock_code, convert_stock_price,transfer_price, year_diff,
        standard_deviation_year_total,my_bond_revise,current_bond_price from kzz_80_120 where standard_deviation_year_total !=''"""
        cur.execute(sql)
        dat = cur.fetchall()
        # dat = cur.fetchmany(3)
        treasury_interest = 0.025  # 国债利率
        for dd in dat:
            # print(dd)
            year_up = dd[5]/100  # 年波动dd[5]
            year_diff = dd[4]  # 剩余年数
            # print("两个日期的间隔天数/365：{} ".format(year_num))
            year_variance_t = year_up*(year_diff**0.5)  # 年波动dd[5]*根号t
            # print("两个日期的间隔年数的根号：{} ".format(year_num**0.5))
            # print("year_up*(year_num**0.5)：{} ".format(year_variance_t))
            convert_stock_price = dd[2]  # S：正股当前的价格
            transfer_price = dd[3]  # 期权的行权价，就是可转债的转股价格
            my_bond_revise = dd[6]  # 调整后可转债现值
            current_bond_price = dd[7]  # 可转债当前价格
            # print(convert_stock_price, transfer_price)
            try:
                #  S：正股当前的价格/X：期权的行权价，就是可转债的转股价格, 取e为底对数
                ln2 = math.log((convert_stock_price/transfer_price), math.e)
                # print("dd[2]", dd[2])
                # print("dd[3]", dd[3])
                # print("dd[2]/dd[3]", dd[2]/dd[3])
                # print("log", log2)
                d1 = (ln2 + (treasury_interest + 0.5*year_up*year_up)*year_diff)/year_variance_t
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
                e_power = 2.718**(-treasury_interest*year_diff)
                # print(e_power)
                # C：当前看涨期权的价值，这个是正股的期权价值（最后要转成可转债的）
                c = convert_stock_price*cdf1 - transfer_price*e_power*cdf2
                # print("c", c)
                option_value = (100/transfer_price)*c
                # print("option_value", option_value)
                # year_option_value = option_value/  #  计算期权的年化收益率
                # print("dd[6]", dd[6])
                bond_total_value = option_value + my_bond_revise
                # print("bond_total_value", bond_total_value)
                bond_total_premium = round((bond_total_value-current_bond_price)/current_bond_price, 4)  # 总溢价率
                lis = [round(option_value, 3), round(bond_total_value, 3), bond_total_premium, dd[0]]
                sql_up2 = """update kzz_80_120 set option_value=?,bond_total_value=?,bond_total_premium=?
                where security_code=?"""
                cur.execute(sql_up2, lis)
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
