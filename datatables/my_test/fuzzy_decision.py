import requests
import json
import os
from ..tool import mysetting
import sqlite3


def fuzzy_decision():
    # 模糊综合评价，计算模糊矩阵和指标权重
    import pandas as pd
    pp = r'C:\Users\Administrator\Desktop\test1.xlsx'
    df = pd.read_excel(pp, engine='openpyxl')

    # 综合评价矩阵
    def comprehensive_evaluation_matrix_(dff):
        sha = dff.iloc[:, 2:7].shape
        # print(sha[0])
        comprehensive_evaluation = []
        for ii in range(0, sha[0]):
            # print(ii)
            mi = []
            ma = []
            uu = dff.loc[ii, "a1"]
            # print("uu", uu)
            for u in dff.iloc[ii, 2:7]:
                # print("u", u)
                if uu >= u:
                    mi.append(u)
                if uu <= u:
                    ma.append(u)
            length = len(mi)
            if length == 0:
                comprehensive_evaluation.append([0, 0, 0, 0, 0])
            elif length == 1:
                small = max(mi)
                big = min(ma)
                if small == big:
                    comprehensive_evaluation.append([1, 0, 0, 0, 0])
                else:
                    small_degree = (big - uu)/(big - small)
                    comprehensive_evaluation.append([small_degree, 1 - small_degree, 0, 0, 0])
            elif length == 2:
                small = max(mi)
                big = min(ma)
                if small == big:
                    comprehensive_evaluation.append([0, 1, 0, 0, 0])
                else:
                    small_degree = (big - uu)/(big - small)
                    comprehensive_evaluation.append([0, small_degree, 1 - small_degree, 0, 0])
            elif length == 3:
                small = max(mi)
                big = min(ma)
                if small == big:
                    comprehensive_evaluation.append([0, 0, 1, 0, 0])
                else:
                    small_degree = (big - uu)/(big - small)
                    comprehensive_evaluation.append([0, 0, small_degree, 1 - small_degree, 0])
            elif length == 4:
                small = max(mi)
                big = min(ma)
                if small == big:
                    comprehensive_evaluation.append([0, 0, 0, 1, 0])
                else:
                    small_degree = (big - uu)/(big - small)
                    comprehensive_evaluation.append([0, 0, 0, small_degree, 1 - small_degree])
            elif length == 5:
                comprehensive_evaluation.append([0, 0, 0, 0, 1])
        return comprehensive_evaluation
    co = comprehensive_evaluation_matrix_(df)
    print(pd.DataFrame(co))
    weight = df["权重"]
    print(weight)
    ddd = "ha"
    if ddd == "ha":
        print("aa,矩阵乘法")
        print(weight.dot(co))

    # 模糊矩阵合成，先取小，再取大,M（Λ，V）
    def fuzzy_matrix_synthesis(matrix_a, matrix_b):
        ma = list(map(list, zip(*matrix_b)))
        # print(ma)
        synthesis_big = []
        for i, tt in enumerate(ma):
            synthesis = []
            if matrix_a[0] >= tt[0]:
                synthesis.append(tt[0])
            else:
                synthesis.append(matrix_a[0])
            if matrix_a[1] >= tt[1]:
                synthesis.append(tt[1])
            else:
                synthesis.append(matrix_a[1])
            if matrix_a[2] >= tt[2]:
                synthesis.append(tt[2])
            else:
                synthesis.append(matrix_a[2])
            synthesis_big.append(max(synthesis))
        return synthesis_big

    ddd = "ha"
    if ddd == "ha":
        aa = fuzzy_matrix_synthesis(weight, co)
        print("aa1,先取小，再取大")
        print(aa)

    # 模糊矩阵合成，先乘，再取大,M（*，V）
    def fuzzy_matrix_synthesis2(matrix_a, matrix_b):
        ma = list(map(list, zip(*matrix_b)))
        # print(ma)
        synthesis_big = []
        for tt in ma:
            synthesis = [
                matrix_a[0] * tt[0], matrix_a[1] * tt[1], matrix_a[2] * tt[2]
            ]
            synthesis_big.append(max(synthesis))
        return synthesis_big
    ddd = "2"
    if ddd == "2":
        aa = fuzzy_matrix_synthesis2(weight, co)
        print("aa2,先乘，再取大")
        print(aa)

    # 模糊矩阵合成，先取小，再求和,M（Λ，+）
    def fuzzy_matrix_synthesis3(matrix_a, matrix_b):
        ma = list(map(list, zip(*matrix_b)))
        # print(ma)
        synthesis_big = []
        for tt in ma:
            if matrix_a[0] >= tt[0]:
                synthesis = tt[0]
            else:
                synthesis = matrix_a[0]
            if matrix_a[1] >= tt[1]:
                synthesis += tt[1]
            else:
                synthesis += matrix_a[1]
            if matrix_a[2] >= tt[2]:
                synthesis += tt[2]
            else:
                synthesis += matrix_a[2]
            synthesis_big.append(synthesis)
        return synthesis_big
    ddd = "3"
    if ddd == "3":
        aa = fuzzy_matrix_synthesis3(weight, co)
        print("aa3,先取小，再求和")
        print(aa)


# 多目标模糊综合评价
def multiple_fuzzy_decision():
    # 计算模糊矩阵和指标权重
    import pandas as pd
    import numpy as np
    # from sklearn import preprocessing
    pp = r'C:\Users\Administrator\Desktop\test1.xlsx'

    # lgt,organization buy 综合评价矩阵
    def lgt_comprehensive_evaluation_matrix(f):
        dff = pd.read_excel(pp, sheet_name=f, engine='openpyxl')
        ind = dff.columns.get_loc("权重")  # 获取列所在的索引
        # 把'指标'这列，转为行索引
        dff.set_index('指标', inplace=True)
        net_purchase = dff.loc['净买入', ][ind:]
        buy = dff.loc['buy', ][ind:]
        sell = dff.loc['sell', ][ind:]
        ne = net_purchase/7
        n2 = np.where(ne > 1, 1, ne)  # numpy.where (condition[, x, y])满足条件(condition)，输出x，不满足输出y。
        l_se_bu = 1 - (sell / buy)
        n = n2*l_se_bu  # 净买入*买卖比
        for i, net_pur in enumerate(net_purchase):
            if net_pur < 1:  # 小于1k是处理前净买入*买卖比
                l_se_bu[i] = net_pur*l_se_bu[i]
        lis = [n.to_list(), l_se_bu]
        print(pd.DataFrame(lis))
        weigh = dff["权重"]
        weigh = weigh[weigh.notnull()]
        print(weigh)
        print("aa,矩阵乘法")
        lg = weigh.dot(lis)
        return lg
    ddd = "lgt"
    if ddd == "lgt":
        lgt = lgt_comprehensive_evaluation_matrix("lgt")
        print(lgt)
    # elif ddd == "organization":
        organization = lgt_comprehensive_evaluation_matrix("organization")
        print(organization)

    # total综合评价矩阵
    def total_comprehensive_evaluation_matrix(f):
        dff = pd.read_excel(pp, sheet_name=f, engine='openpyxl')
        ind = dff.columns.get_loc("权重")  # 获取列所在的索引
        # 把'指标'这列，转为行索引
        dff.set_index('指标', inplace=True)
        net_purchase = dff.loc['净买入', ][ind:]
        buy = dff.loc['buy', ][ind:]
        sell = dff.loc['sell', ][ind:]
        # net_purchase = np.where(net_purchase <= 0, 0, net_purchase)  # numpy.where (condition[, x, y])满足条件(condition)，输出x，不满足输出y。
        # for i, net_pur in enumerate(net_purchase):
        #     if net_pur > 0:
        #         net_purchase[i] = 1-np.power(2.718, -0.8*net_purchase[i])
        l_se_bu = buy
        for i, buy0 in enumerate(buy):
            if buy0 < 1:  #
                net_purchase[i] = 0
                l_se_bu[i] = 0
            else:
                if net_purchase[i] < 0:
                    net_purchase[i] = 0
                    l_se_bu[i] = 0
                else:
                    net_purchase[i] = 1-np.power(2.718, -0.3*(net_purchase[i]+0))
                    l_se_bu[i] = np.power(2.718, -1.5*(sell[i]/buy0))
        lis = [net_purchase, l_se_bu]
        print(pd.DataFrame(lis))
        weigh = dff["权重"]
        weigh = weigh[weigh.notnull()]
        print(weigh)
        print("aa,矩阵乘法")
        lg = weigh.dot(lis)
        return lg
    ddd = "total"
    if ddd == "total":
        total = total_comprehensive_evaluation_matrix("total")
        print(total)

    # buy organization number综合评价矩阵
    def organization_num_comprehensive_evaluation_matrix(f):
        dff = pd.read_excel(pp, sheet_name=f, engine='openpyxl')
        ind = dff.columns.get_loc("权重")  # 获取列所在的索引
        # 把'指标'这列，转为行索引
        dff.set_index('指标', inplace=True)
        net_purchase = dff.loc['净买入', ][ind:]
        buy = dff.loc['buy', ][ind:]
        sell = dff.loc['sell', ][ind:]
        # n = np.where(net_purchase == 1, 1 - np.power(2.718, -0.8 * 1), net_purchase)
        # n = np.where(n == 2, 1 - np.power(2.718, -0.8 * 2), n)
        # # print(n)
        # n = np.where(n >= 3, 1, n)  # numpy.where (condition[, x, y])满足条件(condition)，输出x，不满足输出y。
        # n = np.where(n <= 0, 0, n)
        dd9 = "matrix"
        if dd9 == "matrix":
            se_bu = sell / buy
            for i, bu in enumerate(buy):
                if bu == 1:  #
                    if sell[i] == 0:
                        se_bu[i] = 0
                        net_purchase[i] = 1
                    else:
                        se_bu[i] = np.power(2.718, -3*se_bu[i])
                        net_purchase[i] = 0
                elif bu == 2:
                    if sell[i] == 0:
                        se_bu[i] = 0.75
                        net_purchase[i] = 0.75
                    else:
                        se_bu[i] = np.power(2.718, -1.4*se_bu[i])
                        net_purchase[i] = 1 - np.power(2.718, -0.15 * (net_purchase[i]+3))
                elif bu == 3:
                    if sell[i] == 0:
                        se_bu[i] = 1
                        net_purchase[i] = 1
                    else:
                        se_bu[i] = np.power(2.718, -0.35 * se_bu[i])  # k小越大 k大越小
                        net_purchase[i] = 1 - np.power(2.718, -0.75 * (net_purchase[i] + 3))  # k大越大
                elif bu > 3:
                    se_bu[i] = 1
                    net_purchase[i] = 1
                else:
                    print("miss")
            lis = [net_purchase, se_bu]
            print(pd.DataFrame(lis))
            weigh = dff["权重"]
            weigh = weigh[weigh.notnull()]
            print(weigh)
            print("organization_num矩阵乘法")
            lg = weigh.dot(lis)
            return lg
    ddd = "organization_num"
    if ddd == "organization_num":
        organization_num = organization_num_comprehensive_evaluation_matrix("organization_num")
        print(organization_num)

    # other综合评价矩阵
    def other_comprehensive_evaluation_matrix(f):
        dff = pd.read_excel(pp, sheet_name=f, engine='openpyxl')
        ind = dff.columns.get_loc("权重")  # 获取列所在的索引
        # 把'指标'这列，转为行索引
        dff.set_index('指标', inplace=True)
        market_value = dff.loc['市值', ][ind:]
        lg_tong = dff.loc['陆股通', ][ind:]
        add = dff.loc['增长', ][ind:]
        organizate = dff.loc['机构股东', ][ind:]
        lose = dff.loc['亏', ][ind:]
        # net_purchase = np.where(net_purchase <= 0, 0, net_purchase)  # numpy.where (condition[, x, y])满足条件(condition)，输出x，不满足输出y。
        for i, market_value0 in enumerate(market_value):
            market_value[i] = 1 - np.power(2.718, -1.7 * (market_value0/100 + 0))
        for i, lgt0 in enumerate(lg_tong):
            lg_tong[i] = 1-np.power(2.718, -60*lgt0)
        lis = [market_value, lg_tong, add, organizate, lose]
        print(pd.DataFrame(lis))
        weigh = dff["权重"]
        weigh = weigh[weigh.notnull()]
        print(weigh)
        print("aa,矩阵乘法")
        lg = weigh.dot(lis)
        return lg
    ddd = "other"
    if ddd == "other":
        other = other_comprehensive_evaluation_matrix("other")
        print(other)
    lis_total = pd.DataFrame([lgt, organization, total, organization_num, other])
    print(lis_total)
    print("lis_total,矩阵乘法")
    data = pd.DataFrame([0.2, 0.2, 0.2, 0.2, 0.2])
    data = pd.DataFrame(data.values.T, index=data.columns, columns=data.index)
    print(data.shape)
    total_comp = data.dot(lis_total)
    print(total_comp)

    # 模糊矩阵合成，先取小，再取大,M（Λ，V）
    def fuzzy_matrix_synthesis(matrix_a, matrix_b):
        ma = list(map(list, zip(*matrix_b)))
        # print(ma)
        synthesis_big = []
        for i, tt in enumerate(ma):
            synthesis = []
            if matrix_a[0] >= tt[0]:
                synthesis.append(tt[0])
            else:
                synthesis.append(matrix_a[0])
            if matrix_a[1] >= tt[1]:
                synthesis.append(tt[1])
            else:
                synthesis.append(matrix_a[1])
            if matrix_a[2] >= tt[2]:
                synthesis.append(tt[2])
            else:
                synthesis.append(matrix_a[2])
            synthesis_big.append(max(synthesis))
        return synthesis_big
    ddd = ""
    if ddd == "ha":
        aa = fuzzy_matrix_synthesis(weight, co)
        print("aa1,先取小，再取大")
        print(aa)

    # 模糊矩阵合成，先乘，再取大,M（*，V）
    def fuzzy_matrix_synthesis2(matrix_a, matrix_b):
        ma = list(map(list, zip(*matrix_b)))
        # print(ma)
        synthesis_big = []
        for tt in ma:
            synthesis = [
                matrix_a[0] * tt[0], matrix_a[1] * tt[1], matrix_a[2] * tt[2]
            ]
            synthesis_big.append(max(synthesis))
        return synthesis_big
    ddd = ""
    if ddd == "2":
        aa = fuzzy_matrix_synthesis2(weight, co)
        print("aa2,先乘，再取大")
        print(aa)

    # 模糊矩阵合成，先取小，再求和,M（Λ，+）
    def fuzzy_matrix_synthesis3(matrix_a, matrix_b):
        ma = list(map(list, zip(*matrix_b)))
        # print(ma)
        synthesis_big = []
        for tt in ma:
            if matrix_a[0] >= tt[0]:
                synthesis = tt[0]
            else:
                synthesis = matrix_a[0]
            if matrix_a[1] >= tt[1]:
                synthesis += tt[1]
            else:
                synthesis += matrix_a[1]
            if matrix_a[2] >= tt[2]:
                synthesis += tt[2]
            else:
                synthesis += matrix_a[2]
            synthesis_big.append(synthesis)
        return synthesis_big
    ddd = ""
    if ddd == "3":
        aa = fuzzy_matrix_synthesis3(weight, co)
        print("aa3,先取小，再求和")
        print(aa)


# 获取龙虎榜数据
def fuzzy_dragon_tiger():
    # 判断路径是否存在
    def is_not_path(file):
        for u in mysetting.ZQ_URL:  # 证券行情软件路径
            if os.access(u + file, os.F_OK):
                # print(u + file)
                return u + file

    # 读交易软件里龙虎榜股票代码
    def read_dragon_code(file):
        path = is_not_path(file)
        # print("read...." + file)
        with open(path) as f:  # 自动关闭
            # stock_list = f.readlines()
            # splitlines() 按照行('\r', '\r\n', \n')分隔，返回一个包含各行作为元素的列表，如果参数 keepends 为默认 False，不包含换行符，如果为 True，则保留换行符
            stock_list = f.read().splitlines()
            # print(stock_list)
            if stock_list:
                for i, value in enumerate(stock_list):
                    if value == "\n":
                        stock_list.remove("\n")
                    elif value == "":
                        stock_list.remove("")
            # print("stock_list", stock_list)
            return stock_list

    # nineteen 龙虎榜
    def per_dragon_tiger(code, header):
        date_li = per_dragon_tiger1(code, header)
        if len(date_li) > 2:
            date_li = date_li[0:1]
        # print(date_li)
        # return ""
        return per_dragon_tiger2(code, date_li, header)

    #  龙虎榜son1,获取个股龙虎榜日期
    def per_dragon_tiger1(code, header):
        url = 'http://datainterface3.eastmoney.com/EM_DataCenter_V3/api/LHBGGSBRQ/GetLHBGGSBRQ?tkn=eastmoney&scode={}&dayNum=100&startDateTime=&endDateTime=&sortField=1&sortDirec=1&pageNum=1&pageSize=1000&cfg=ggsbrq&js='
        vv = requests.get(url.format(code), headers=header)
        # print(vv.status_code)
        text = vv.text
        # print(text)
        lgt = []
        if vv.status_code == 200 and text:
            detail = json.loads(text).get("Data", "")
            # print(detail)
            # detail = ""
            if detail:
                li = detail[0].get("Data", "")
                if li:
                    for v in li:
                        # print(v)
                        # v = ""
                        if v:
                            notice_dat = v.split("|")
                            if notice_dat:
                                lgt.append(notice_dat[1])
                                # print(notice_dat[1])
        # print(lgt)
        return lgt

    #  龙虎榜son2
    def per_dragon_tiger2(code, date_li, header):
        url = 'http://datainterface3.eastmoney.com/EM_DataCenter_V3/api/LHBMMMX/GetLHBMXKZ?tkn=eastmoney&Code={}&dateTime={}&pageNum=1&pageSize=50&cfg=lhbmxkz&js='
        lgt = []
        if date_li and code:
            for d in date_li:
                vv = requests.get(url.format(code, d), headers=header)
                # print(vv.status_code)
                text = vv.text
                # print(text)
                if vv.status_code == 200 and text:
                    detail = json.loads(text).get("Data", "")
                    # print(detail)
                    # detail = ""
                    if detail:
                        li = detail[0].get("Data", "")
                        if li:
                            # 13日期,14类型,排名2， 19交易营业， 21上榜次数，22胜率,11买入，17卖入占总成交比,卖出9，卖出占总成交比16，净额20
                            lg = []
                            flag = ""
                            for v in li:
                                # print(v)
                                # v = ""
                                if v:
                                    n = v.split("|")
                                    if n:
                                        if n[12]:  # 13日期
                                            n[12] = n[12].split(" ")[0]
                                        # print(flag != n[13])
                                        if flag != n[13]:  # 类型
                                            flag = n[13]
                                            lg.append([code, n[13], n[12], "", "", "", "", "", ""])
                                            # print("55555", code)
                                            lg.append(["", "营业部", "上榜次数", "胜率%", "买入", "买入占成交%", "卖出", "卖出占成交%", "余额"])
                                        if n[10]:
                                            n[10] = '{:.2f}亿'.format(float(n[10]) / 100000000)
                                        if n[8]:
                                            n[8] = '{:.2f}亿'.format(float(n[8]) / 100000000)
                                        if n[19]:
                                            n[19] = '{:.2f}亿'.format(float(n[19]) / 100000000)
                                        lg.append([n[1], n[18], n[20], n[21], n[10], n[16], n[8], n[15], n[19]])
                            # print(lg)
                            lgt.append(lg)
        # print(lgt)
        return lgt

    # 整理dragon数据
    def arrangement_dragon():
        stock_list = read_dragon_code("DRAGON_TIGER.blk")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36',
        }
        list_two = []
        for vvv in stock_list[:-1]:
            aaa = per_dragon_tiger(vvv[1:], headers)
            # print(dra)
            # aaa = [[['code', '涨幅偏离值达7%的证券', '2022/2/11', '', '', '', '', '', ''], ['', '营业部', '上榜次数', '胜率%', '买入', '买入占成交%', '卖出', '卖出占成交%', '余额'], ['1', '深股通专用', '153', '39.87', '0.71亿', '4.03', '0.35亿', '1.97', '0.36亿'], ['2', '机构专用', '1057', '40.68', '0.52亿', '2.96', '0.00亿', '0.00', '0.52亿'], ['3', '机构专用', '1057', '40.68', '0.47亿', '2.66', '0.00亿', '0.00', '0.47亿'], ['4', '华泰证券股份有限公司天津华昌道证券营业部', '3', '66.67', '0.46亿', '2.63', '0.00亿', '0.01', '0.46亿'], ['5', '机构专用', '1057', '40.68', '0.36亿', '2.06', '0.00亿', '0.00', '0.36亿'], ['1', '国泰君安证券股份有限公司宜春高安瑞州路证券营业部', '0', '', '0.19亿', '1.06', '0.44亿', '2.48', '-0.25亿'], ['2', '招商证券股份有限公司深圳水贝证券营业部', '1', '0.00', '0.02亿', '0.13', '0.41亿', '2.31', '-0.38亿'], ['3', '深股通专用', '153', '39.87', '0.71亿', '4.03', '0.35亿', '1.97', '0.36亿'], ['4', '机构专用', '1057', '40.68', '0.00亿', '0.00', '0.27亿', '1.55', '-0.27亿'], ['5', '华融证券股份有限公司武汉解放大道证券营业部', '0', '', '0.00亿', '0.00', '0.25亿', '1.40', '-0.25亿'], ['11', '', '1057', '40.68', '2.75亿', '15.53', '1.72亿', '9.71', '1.03亿'], ['code', '连续三个交易日内，涨幅偏离值累计达20%的证券', '2022/2/11', '', '', '', '', '', ''], ['', '营业部', '上榜次数', '胜率%', '买入', '买入占成交%', '卖出', '卖出占成交%', '余额'], ['1', '深股通专用', '153', '39.87', '1.27亿', '4.35', '1.45亿', '4.97', '-0.18亿'], ['2', '机构专用', '1057', '40.68', '1.21亿', '4.14', '0.00亿', '0.00', '1.21亿'], ['3', '机构专用', '1057', '40.68', '0.52亿', '1.79', '0.00亿', '0.00', '0.52亿'], ['4', '机构专用', '1057', '40.68', '0.47亿', '1.61', '0.00亿', '0.00', '0.47亿'], ['5', '华泰证券股份有限公司天津华昌道证券营业部', '3', '66.67', '0.47亿', '1.59', '0.00亿', '0.01', '0.46亿'], ['1', '深股通专用', '153', '39.87', '1.27亿', '4.35', '1.45亿', '4.97', '-0.18亿'], ['2', '国泰君安证券股份有限公司宜春高安瑞州路证券营业部', '0', '', '0.19亿', '0.65', '0.45亿', '1.53', '-0.26亿'], ['3', '招商证券股份有限公司深圳水贝证券营业部', '1', '0.00', '0.03亿', '0.09', '0.41亿', '1.40', '-0.38亿'], ['4', '机构专用', '1057', '40.68', '0.44亿', '1.51', '0.27亿', '0.94', '0.17亿'], ['5', '国盛证券有限责任公司上海徐汇区虹桥路证券营业部', '0', '', '0.02亿', '0.08', '0.26亿', '0.90', '-0.24亿'], ['11', '', '1057', '40.68', '4.62亿', '15.81', '2.85亿', '9.74', '1.77亿']]]
            # aaa = [[['code', '连续三个交易日内，涨幅偏离值累计达20%的证券', '2022/2/11', '', '', '', '', '', ''], ['', '营业部', '上榜次数', '胜率%', '买入', '买入占成交%', '卖出', '卖出占成交%', '余额'], ['1', '深股通专用', '153', '39.87', '1.27亿', '4.35', '1.45亿', '4.97', '-0.18亿'], ['2', '机构专用', '1057', '40.68', '1.21亿', '4.14', '0.00亿', '0.00', '1.21亿'], ['3', '机构专用', '1057', '40.68', '0.52亿', '1.79', '0.00亿', '0.00', '0.52亿'], ['4', '机构专用', '1057', '40.68', '0.47亿', '1.61', '0.00亿', '0.00', '0.47亿'], ['5', '华泰证券股份有限公司天津华昌道证券营业部', '3', '66.67', '0.47亿', '1.59', '0.00亿', '0.01', '0.46亿'], ['1', '深股通专用', '153', '39.87', '1.27亿', '4.35', '1.45亿', '4.97', '-0.18亿'], ['2', '国泰君安证券股份有限公司宜春高安瑞州路证券营业部', '0', '', '0.19亿', '0.65', '0.45亿', '1.53', '-0.26亿'], ['3', '招商证券股份有限公司深圳水贝证券营业部', '1', '0.00', '0.03亿', '0.09', '0.41亿', '1.40', '-0.38亿'], ['4', '机构专用', '1057', '40.68', '0.44亿', '1.51', '0.27亿', '0.94', '0.17亿'], ['5', '国盛证券有限责任公司上海徐汇区虹桥路证券营业部', '0', '', '0.02亿', '0.08', '0.26亿', '0.90', '-0.24亿'], ['11', '', '1057', '40.68', '4.62亿', '15.81', '2.85亿', '9.74', '1.77亿'],
            #         ['code', '涨幅偏离值达7%的证券', '2022/2/11', '', '', '', '', '', ''], ['', '营业部', '上榜次数', '胜率%', '买入', '买入占成交%', '卖出', '卖出占成交%', '余额'], ['1', '深股通专用', '153', '39.87', '0.71亿', '4.03', '0.35亿', '1.97', '0.36亿'], ['2', '机构专用', '1057', '40.68', '0.52亿', '2.96', '0.00亿', '0.00', '0.52亿'], ['3', '机构专用', '1057', '40.68', '0.47亿', '2.66', '0.00亿', '0.00', '0.47亿'], ['4', '华泰证券股份有限公司天津华昌道证券营业部', '3', '66.67', '0.46亿', '2.63', '0.00亿', '0.01', '0.46亿'], ['5', '机构专用', '1057', '40.68', '0.36亿', '2.06', '0.00亿', '0.00', '0.36亿'], ['1', '国泰君安证券股份有限公司宜春高安瑞州路证券营业部', '0', '', '0.19亿', '1.06', '0.44亿', '2.48', '-0.25亿'], ['2', '招商证券股份有限公司深圳水贝证券营业部', '1', '0.00', '0.02亿', '0.13', '0.41亿', '2.31', '-0.38亿'], ['3', '深股通专用', '153', '39.87', '0.71亿', '4.03', '0.35亿', '1.97', '0.36亿'], ['4', '机构专用', '1057', '40.68', '0.00亿', '0.00', '0.27亿', '1.55', '-0.27亿'], ['5', '华融证券股份有限公司武汉解放大道证券营业部', '0', '', '0.00亿', '0.00', '0.25亿', '1.40', '-0.25亿'], ['11', '', '1057', '40.68', '2.75亿', '15.53', '1.72亿', '9.71', '1.03亿']]]
            aaa = aaa[0]
            length = len(aaa)
            lis = aaa[0][:3]
            ss = json.dumps(aaa[2:13], ensure_ascii=False)
            lis.append(ss)
            list_two.append(lis)
            # print(list_two)
            if length > 13:
                if "连续三" in aaa[0][1]:
                    lis2 = aaa[13][:3]
                    sss2 = json.dumps(aaa[15:26], ensure_ascii=False)
                    lis2.append(sss2)
                    list_two.append(lis2)
                    # print(len(list_two))
                    # print(list_two)
                else:
                    a13 = aaa[13:]
                    for i, v in enumerate(a13):
                        # print(v)
                        if "连续三" in v[1]:
                            lis3 = v[:3]
                            # print(lis3)
                            sss3 = json.dumps(a13[i+2:i+13], ensure_ascii=False)
                            lis3.append(sss3)
                            list_two.append(lis3)
                            break
            # print(list_two)
        if os.path.isfile(mysetting.DATA_TABLE_DB):
            with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
                cu = conn.cursor()
                for t in list_two:
                    cu.execute("INSERT INTO fuzzy_dragon VALUES(?,?,?,?)", t)
                cu.close()

    arrangement_dragon()

