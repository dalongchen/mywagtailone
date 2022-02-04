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
