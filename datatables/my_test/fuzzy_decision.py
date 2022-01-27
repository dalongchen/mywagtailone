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

    # lgt综合评价矩阵
    def lgt_comprehensive_evaluation_matrix():
        dff = pd.read_excel(pp, sheet_name="lgt", engine='openpyxl')
        ind = dff.columns.get_loc("权重")  # 获取列所在的索引
        # 把'指标'这列，转为行索引
        dff.set_index('指标', inplace=True)
        net_purchase = dff.loc['净买入', ][ind:]
        buy = dff.loc['buy', ][ind:]
        sell = dff.loc['sell', ][ind:]
        ne = net_purchase/5
        sb = []
        n = np.where(ne > 1, 1, ne)  # numpy.where (condition[, x, y])满足条件(condition)，输出x，不满足输出y。
        for i, tt in enumerate(buy):
            if tt > 0.6:
                ttt0 = 1 - (sell[i] / tt)
                if ttt0 < 0.5:
                    sb.append(0)
                else:
                    sb.append(ttt0)
            else:
                ttt = 1 - (sell[i] / tt)
                if ttt == 1:
                    sb.append(ttt*tt)
                    # print(ttt*0.5)
                elif ttt < 0.5:
                    sb.append(0)
                else:
                    sb.append(ttt*tt)
                    # print(ttt*ttt)
        # bbb = buy[buy < 0.6].index
        # print(bbb.shape)
        # print(bbb.shape[0])
        # print(bbb == [])
        # sb = 1 - (sell / buy)
        # ff = sb[bbb]
        # sb[bbb] = ff*ff
        # print(sb)
        # b = np.where(sb < 0.5, 0, sb)
        lis = [n, sb]
        print(pd.DataFrame(lis))
        weigh = dff["权重"]
        weigh = weigh[weigh.notnull()]
        print(weigh)
        print("aa,矩阵乘法")
        lg = weigh.dot(lis)
        return lg
    ddd = "matrix"
    if ddd == "matrix":
        lgt = lgt_comprehensive_evaluation_matrix()
        print(lgt)

    # organization综合评价矩阵
    def organization_comprehensive_evaluation_matrix():
        dff = pd.read_excel(pp, sheet_name="lgt", engine='openpyxl')
        ind = dff.columns.get_loc("权重")  # 获取列所在的索引
        # print(ind)
        # dff2 = dff.iloc[:, ind+1:]
        # 把'指标'这列，转为航索引
        dff.set_index('指标', inplace=True)
        net_purchase = dff.loc['净买入', ][ind:]
        buy = dff.loc['buy', ][ind:]
        sell = dff.loc['sell', ][ind:]
        ne = net_purchase/5
        n = np.where(ne > 1, 1, ne)  # numpy.where (condition[, x, y])满足条件(condition)，输出x，不满足输出y。
        n = np.where(n < 0.6, 0, n)
        # print(n)
        sb = 1 - (sell / buy)
        b = np.where(sb < 0.5, 0, sb)
        # print(b)
        lis = [n, b]
        weight = dff["权重"]
        weight = weight[weight.notnull()]
        print(weight)
        print("aa,矩阵乘法")
        print(weight.dot(lis))
        return lis
    ddd = "matrix0"
    if ddd == "matrix":
        organization = organization_comprehensive_evaluation_matrix()
        print(pd.DataFrame(organization))

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
