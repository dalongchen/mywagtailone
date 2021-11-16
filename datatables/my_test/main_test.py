from pprint import pprint
import requests
from .. import views


def test_sq_lite(path):
    import sqlite3
    # from mywagtailone.datatables.tool import mysetting
    # from .tool import tools
    import os
    print(os.path.isfile(path))
    if os.path.isfile(path):
        with sqlite3.connect(path) as conn:
            # conn.text_factory = lambda x: str(x, 'gbk', 'ignore')
            cu = conn.cursor()
            # cu.execute("select * FROM ymd_1280194006")
            # cu.execute("select * FROM HDConfig")
            # cu.execute("select * FROM HDProp")
            # columns = [_[0].lower() for _ in cu.description]
            # results = [dict(zip(columns, _)) for _ in cu]
            # for ii in results:
            #     print(ii)
            # sq = "delete FROM ymd_1280194006 WHERE dat!='{}'".format(str(trade_date))
            # print(sq)
            # cu.execute("delete FROM ymd_1280194006 WHERE dat!='{}'".format(str(trade_date)))

            # cu.execute("select code FROM dragon_tiger where date=?", (ii,))
            cuu = [
                [
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 5, 4, 3, "tt", 16, 17, 18],
                    ["rr", 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 5, 4, 3, "tt", 16, 17, 18],
                ],
                [
                    [11, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 5, 4, 3, "tt", 16, 17, 18],
                    ["rrr", 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 5, 4, 3, "tt", 16, 17, 18],
                ],
            ]
            for _ in cuu:
                # code = views.add_sh(_[0], big="baostock")
                print(_)
                # data = test_get_k_code(code, ii, d_add.strftime('%Y-%m-%d'))
                # print(data)
                for d in _:
                    cu.execute("INSERT INTO dragon_tiger_k VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                        d[0],
                        d[1],
                        d[2],
                        d[3],
                        d[4],
                        d[5],
                        d[6],
                        d[7],
                        d[8],
                        d[9],
                        d[10],
                        d[11],
                        d[12],
                        d[13],
                        d[14],
                        d[15],
                        d[16],
                        d[17]))

            cu.close()


def test_my():
    a = [7, 1, 2, 5, 3]
    b = [2, 6, 3, 4]
    ret = [i for i in a if i not in b]
    # ret = list(set(a) ^ set(b))
    # ret = list(set(a).difference(set(b)))
    print(ret)
    import numpy as np
    a = np.arange(24).reshape(2, 3, 4)
    print("sa", a[:, -1, 1])
    print("a", a)


# 压缩文件
def zip_ya(start_dir, tagger, f="local"):
    import zipfile
    import os
    if f == "local":
        file_news = start_dir +".zip"  # 压缩后文件夹的名字
    else:
        file_news = tagger +".zip"  # 压缩后文件夹的名字
    print(file_news)
    z = zipfile.ZipFile(file_news, 'w', zipfile.ZIP_DEFLATED)  # 参数一：文件夹名
    for dirpath, dirnames, filenames in os.walk(start_dir):
        fpath = dirpath.replace(start_dir, '')  # 把start_dir代替为空
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename), fpath+filename)
    print('压缩成功')
    z.close()


# 保存持仓信息
def test_trade_save():
    import os
    if os.path.isfile(r"D:\ana\envs\py36\mywagtailone\my_ignore\table.xls"):
        os.remove(r"D:\ana\envs\py36\mywagtailone\my_ignore\table.xls")
    dialog = views.log_on_ht()
    # dialog.window(best_match="资金股票", auto_id="", class_name="", control_type="TreeItem").set_focus()
    dia = dialog.window(best_match="Custom1", auto_id="1047", class_name="CVirtualGridCtrl", control_type="Pane")
    # dia.print_control_identifiers()
    dia.wait("visible", timeout=10, retry_interval=2)
    # pprint(dir(dia.wrapper_object()))
    # print(dia.window_text())
    # dia.set_focus()
    dia.type_keys("^s")
    dia.wait("visible", timeout=5, retry_interval=1)
    save = dialog.window(best_match="保存(S)", auto_id="1", class_name="Button", control_type="Button")
    # save.draw_outline()
    save.type_keys("{VK_RETURN}")


def test_read_xls():
    import pandas as pd
    # df = pd.read_table(r"D:\ana\envs\py36\mywagtailone\my_ignore\table.xls", encoding="gbk")
    df = pd.read_table(r"D:\ana\envs\py36\mywagtailone\my_ignore\table.xls", usecols=["证券代码", "股票余额"], encoding="gbk")
    # dd = df["证券代码"]
    dd = df[(df["证券代码"] == "600011")]
    # dd = df.loc[df['证券代码'].isin(["600011", "600795", "01235"])]
    print(list(dd["股票余额"])[0])
    print(df.head(10))


# 读东财陆股通龙虎榜
def east_dragon_tiger_new(net, size, start, end, f=""):
    import os
    import sqlite3
    from mywagtailone.datatables.tool import mysetting

    dragon_tiger = requests.get(net.format(size, start, end))
    # print(dragon_tiger)
    if dragon_tiger.status_code == 200 and dragon_tiger:
        if f == "institution":
            dr = dragon_tiger.json().get('data', '')
            if dr:
                # Chgradio涨幅,CTypeDes说明,TurnRate换手率，BSL=buy_num买机构数，SSL=sell_num
                if os.path.isfile(mysetting.DATA_TABLE_DB):
                    with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
                        cu = conn.cursor()
                        for i in dr:
                            if float(i.get('PBuy', '0')) > 10000000:
                                # pprint(i)
                                ii = [
                                    i.get('TDate', ''),
                                    i.get('SCode', ''),
                                    i.get('SName', ''),
                                    i.get('BMoney', '0'),
                                    i.get('SMoney', '0'),
                                    i.get('PBuy', '0'),
                                    i.get('CPrice', '0'),
                                    i.get('Chgradio', '0'),
                                    "机构",
                                    i.get('CTypeDes', ''),
                                    i.get('TurnRate', '0'),
                                    i.get('BSL', '0'),
                                    i.get('SSL', '0')
                                ]
                                cu.execute("INSERT INTO dragon_tiger VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", ii)
                        cu.close()
        else:
            dr = dragon_tiger.json().get('Data', '')
            if dr:
                dd = dr[0].get('Data', '')
                if os.path.isfile(mysetting.DATA_TABLE_DB):
                    with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
                        cu = conn.cursor()
                        for item in dd:
                            t = item.split("|")
                            if views.isfloat(t[17]) > 10000000:
                                # pprint(t)
                                ii = [
                                    t[22],
                                    t[10],
                                    t[8],
                                    t[0],
                                    t[14],
                                    t[17],
                                    t[9],
                                    t[7],
                                    t[5],
                                    t[25],
                                    "",
                                    "",
                                    "",
                                ]
                                # pprint(ii)
                                cu.execute("INSERT INTO dragon_tiger VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", ii)
                        cu.close()
                        # return tiger_list
    # return []


# 读数据库龙虎榜 入行情软件
def dragon_tiger_into_tdx():
    import sqlite3
    from mywagtailone.datatables.tool import mysetting
    import os
    if os.path.isfile(mysetting.DATA_TABLE_DB):
        with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
            # conn.text_factory = lambda x: str(x, 'gbk', 'ignore')
            cu = conn.cursor()
            cu.execute("select date, code FROM dragon_tiger order by date limit 0, 5")
            t = []
            for ii in cu:
                # print(ii[0])
                # print(ii[1])
                dragon_tiger_date_mark(ii)
                # code = views.code_add(ii[1]) + '\n'
                # # print(code)
                # t.append(code)
            # print(t)
            cu.close()
            # views.is_write_stock('ths_choice.blk', t, "write")


# 读文件
def read_file(file):
    with open(file, encoding="gbk", errors='ignore') as f:  # 自动关闭
    # with open(file, 'rb') as f:  # 自动关闭
        stock_list = f.readlines()
        print(len(stock_list))
        for i in stock_list:
            print(i)


# 读龙虎榜日期和日k线
def dragon_tiger_date_mark(path):
    import sqlite3
    import os
    from ..tool import tools

    if os.path.isfile(path):
        with sqlite3.connect(path) as conn:
            cu = conn.cursor()
            cu.execute("select date FROM dragon_tiger ORDER BY date ASC;")
            results = [_[0] for _ in cu]
            # print(results)
            new_numbers = list(set(results))
            new_numbers.sort(key=results.index)
            # print(new_numbers[0:1])
            import baostock as bs
            # 登陆系统
            lg = bs.login()
            if lg.error_code == "0":
                for ii in new_numbers:
                    print(ii)
                    # f = "d"返回2021 - 07 - 01,f="t"为2021 - 07 - 01 00：00：00
                    d_add = tools.get_late_trade_day(ii, add_subtract="add", f="t")
                    d_add = tools.get_late_trade_day(d_add, add_subtract="add")
                    cu.execute("select code FROM dragon_tiger where date=?", (ii,))
                    res = {_[0] for _ in cu}
                    for _ in res:
                        code = views.add_sh(_, big="baostock")
                        # print(_)
                        # print(code)
                        """date	交易所行情日期
                    code	证券代码
                    open	开盘价
                    high	最高价
                    low	最低价
                    close	收盘价
                    preclose	前收盘价	见表格下方详细说明
                    volume	成交量（累计 单位：股）
                    amount	成交额（单位：人民币元）
                    adjustflag	复权状态(1：后复权， 2：前复权，3：不复权）
                    turn	换手率	[指定交易日的成交量(股)/指定交易日的股票的流通股总股数(股)]*100%
                    tradestatus	交易状态(1：正常交易 0：停牌）
                    pctChg	涨跌幅（百分比）	日涨跌幅=[(指定交易日的收盘价-指定交易日前收盘价)/指定交易日前收盘价]*100%
                    peTTM	滚动市盈率	(指定交易日的股票收盘价/指定交易日的每股盈余TTM)=(指定交易日的股票收盘价*截至当日公司总股本)/归属母公司股东净利润TTM
                    pbMRQ	市净率	(指定交易日的股票收盘价/指定交易日的每股净资产)=总市值/(最近披露的归属母公司股东的权益-其他权益工具)
                    psTTM	滚动市销率	(指定交易日的股票收盘价/指定交易日的每股销售额)=(指定交易日的股票收盘价*截至当日公司总股本)/营业总收入TTM
                    pcfNcfTTM	滚动市现率	(指定交易日的股票收盘价/指定交易日的每股现金流TTM)=(指定交易日的股票收盘价*截至当日公司总股本)/现金以及现金等价物净增加额TTM
                    isST	是否ST股，1是，0否
                        """
                        rs = bs.query_history_k_data_plus(code,
                                                          "date,code,open,high,low,close,preclose,volume,amount,"
                                                          "adjustflag,turn,"
                                                          "tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
                                                          start_date=ii, end_date=d_add.strftime('%Y-%m-%d'),
                                                          frequency="d", adjustflag="2")
                        # print('query_history_k_data_plus respond error_code:' + rs.error_code)
                        while (rs.error_code == '0') & rs.next():
                            # 获取一条记录，将记录合并在一起
                            d = rs.get_row_data()
                            # print(d[11])
                            # print(d[11] != "1")
                            if d[11] != "1":
                                print(d[0], d[1])
                            cu.execute("INSERT INTO dragon_tiger_k VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                                            d[0],
                                            d[1],
                                            d[2],
                                            d[3],
                                            d[4],
                                            d[5],
                                            d[6],
                                            d[7],
                                            d[8],
                                            d[9],
                                            d[10],
                                            d[11],
                                            d[12],
                                            d[13],
                                            d[14],
                                            d[15],
                                            d[16],
                                            d[17]))

            cu.close()
            bs.logout()


# 读历史龙虎榜代码和日期
def test_tiger_code(path, date):
    import sqlite3
    import os

    if os.path.isfile(path):
        with sqlite3.connect(path) as conn:
            cu = conn.cursor()
            cu.execute("select * FROM dragon_tiger where date=?", (date,))
            columns = [_[0].lower() for _ in cu.description]
            results = [dict(zip(columns, _)) for _ in cu]
            t = []
            for ii in results:
                code = views.add_sh(ii["code"], big="baostock")
                t.append(code)
            t = list({}.fromkeys(t).keys())
            print(t)
            cu.close()
            return t


# 传入code list，获取k线数据
def test_get_k(stock_list, start, end):
    import baostock as bs

    # print(stock_list)
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    # print('login respond error_code:' + lg.error_code)
    par = []
    if lg.error_code == "0":
        for item in stock_list:
            # print(item)
            rs = bs.query_history_k_data_plus(item, "date,code,open,high,low,close,preclose",
                                              start_date=start, end_date=end, frequency="d", adjustflag="3")
            # print('query_history_k_data_plus respond error_code:' + rs.error_code)
            while (rs.error_code == '0') & rs.next():
                # 获取一条记录，将记录合并在一起
                result = rs.get_row_data()
                # print(result)
                if len(result):
                    f = float(result[6])
                    print((float(result[5])-f)/f)
                    op = ((float(result[2])-f)/f)*((float(result[2])-f)/f)
                    cl = ((float(result[5])-f)/f)*((float(result[5])-f)/f)
                    par.append([op, cl])
                else:
                    print("有误", result)
    # 登出系统
    bs.logout()
    return par


# 画散点图
def test_scatter_diagram(sample):
    """使用scatter()绘制散点图"""
    import matplotlib.pyplot as plt

    import numpy as np
    import sklearn.preprocessing as sp
    # scale - ---零均值单位方差,
    # Z-Score(Standardzation)：将原始数据转换为标准正太分布，即原始数据离均值有几个标准差
    # 缺点：对原始数据的分布有要求，要求原始数据数据分布为正太分布计算；在真实世界中，
    # 总体的均值和标准差很难得到，只能用样本的均值和标准差求得
    # x = np.array([[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]])
    # x_scaled = sp.scale(x)
    # print(x_scaled)
    """
    0-1规范化
　　将样本矩阵中的每一列的最小值和最大值设定为相同的区间，统一各列特征值的范围。一般情况下会把特征值缩放至[0, 1]区间。
例如有一列特征值表示年龄： [17, 20, 23]，每个元素减去特征值数组所有元素的最小值，即可使一组特征值的最小值为0
[17−17,20−17,23−17]−−>[0,3,6]把特征值数组的每个元素除以最大值即可使一组特征值的最大值为1
"""
    samples = np.array(sample)
    # samples = np.array([
    #     [100, 90],
    #     [50, 55],
    #     [10, 20],
    #     [80, 88],
    #     [30, 37]])
    # samples = np.array([10, 100, 30, 50])
    # print(samples.reshape(-1, 1))
    # print(samples.reshape(-1, 2))
    # 根据给定范围创建一个范围缩放器
    mms = sp.MinMaxScaler(feature_range=(0, 1))
    # 用范围缩放器实现特征值的范围缩放,.reshape(-1, 1),-1=?,1=1列
    mms_samples = mms.fit_transform(samples.reshape(-1, 1)).reshape(-1, 2)
    print(mms_samples)
    # print(mms_samples[:, 0])
    # print(mms_samples[:, 1])
    """ 归一化 """
    # r = sp.normalize(samples, norm='l1')
    # print(r)
    # r = sp.normalize(samples, norm='l2')
    # print(r)

    # x_values = range(1, 6)
    # y_values = [x * x for x in x_values]
    '''
    scatter()
    x:横坐标 y:纵坐标 s:点的尺寸
    '''
    plt.scatter(mms_samples[:, 0], mms_samples[:, 1], s=50)

    # 设置图表标题并给坐标轴加上标签
    # plt.title('Square Numbers', fontsize=24)
    # plt.xlabel('Value', fontsize=14)
    # plt.ylabel('Square of Value', fontsize=14)

    # 设置刻度标记的大小
    # plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()


# 对龙虎榜表数据添加说明
def dragon_tiger_add_mark(p):
    import os
    import sqlite3

    if os.path.isfile(p):
        with sqlite3.connect(p) as conn:
            cu = conn.cursor()
            cu.execute("select code,caption FROM dragon_tiger ORDER BY date ASC;")
            cuu = [_ for _ in cu]
            # print(_)
            for c, _ in cuu:
                if (("ST、*ST和S证券连续三个交易日内收盘价格涨幅偏离值累计达到15%的证券" == _)
                        or ("非ST、*ST和S证券连续三个交易日内收盘价格涨幅偏离值累计达到20%的证券" == _)
                        or ("非ST、*ST和S证券连续三个交易日内收盘价格涨幅偏离值累计达到30%的证券" == _)
                        or ("有价格涨跌幅限制的连续3个交易日内收盘价格涨幅偏离值累计达到15%的证券" == _)
                        or ("有价格涨跌幅限制的连续3个交易日内收盘价格涨幅偏离值累计达到20%的证券" == _)
                        or ("有价格涨跌幅限制的连续3个交易日内收盘价格涨幅偏离值累计达到30%的证券" == _)
                        or ("连续三个交易日内，涨幅偏离值累计达到15%的证券" == _)
                        or ("连续三个交易日内，涨幅偏离值累计达到20%的证券" == _)
                        or ("连续三个交易日内，涨幅偏离值累计达到30%的证券" == _)
                        or ("异常期间价格涨幅偏离值累计达到" in _)):
                    cu.execute("update dragon_tiger set caption_mark=? where code=? and caption=?", (1, c, _))
                elif (("ST、*ST和S证券连续三个交易日内收盘价格跌幅偏离值累计达到15%的证券" == _)
                        or ("非ST、*ST和S证券连续三个交易日内收盘价格跌幅偏离值累计达到20%的证券" == _)
                        or ("非ST、*ST和S证券连续三个交易日内收盘价格跌幅偏离值累计达到30%的证券" == _)
                        or ("有价格涨跌幅限制的连续3个交易日内收盘价格跌幅偏离值累计达到15%的证券" == _)
                        or ("有价格涨跌幅限制的连续3个交易日内收盘价格跌幅偏离值累计达到20%的证券" == _)
                        or ("有价格涨跌幅限制的连续3个交易日内收盘价格跌幅偏离值累计达到30%的证券" == _)
                        or ("连续三个交易日内，跌幅偏离值累计达到15%的证券" == _)
                        or ("连续三个交易日内，跌幅偏离值累计达到20%的证券" == _)
                        or ("连续三个交易日内，跌幅偏离值累计达到30%的证券" == _)
                        or ("异常期间价格跌幅偏离值累计达到" in _)):
                    cu.execute("update dragon_tiger set caption_mark=? where code=? and caption=?", (2, c, _))
                elif (("日涨幅偏离值达到7%的前5只证券" == _)
                      or ("日价格涨幅偏离值达到" in _)
                      or ("有价格涨跌幅限制的日收盘价格涨幅偏离值达到7%的前三只证券" == _)
                      or ("有价格涨跌幅限制的日收盘价格涨幅达到15%的前五只证券" == _)
                      or ("日涨幅达到15%的前5只证券" == _)
                      or ("日价格涨幅达到20" in _)):
                    cu.execute("update dragon_tiger set caption_mark=? where code=? and caption=?", (3, c, _))
                elif (("日跌幅偏离值达到7%的前5只证券" == _)
                      or ("日跌幅达到15%的前5只证券" == _)
                      or ("有价格涨跌幅限制的日收盘价格跌幅偏离值达到7%的前三只证券" == _)
                      or ("有价格涨跌幅限制的日收盘价格跌幅达到15%的前五只证券" == _)):
                    cu.execute("update dragon_tiger set caption_mark=? where code=? and caption=?", (4, c, _))
                elif (("日换手率达到20%的前5只证券" == _)
                      or ("日换手率达到30%的前5只证券" == _)
                      or ("日换手率达到" in _)
                      or ("有价格涨跌幅限制的日换手率达到20%的前三只证券" == _)):
                    cu.execute("update dragon_tiger set caption_mark=? where code=? and caption=?", (5, c, _))
                elif (("有价格涨跌幅限制的日价格振幅达到15%的前三只证券" == _)
                      or ("日振幅值达到15%的前5只证券" == _)
                      or ("日振幅值达到30%的前5只证券" == _)
                      or ("日价格振幅达到" in _)):
                    cu.execute("update dragon_tiger set caption_mark=? where code=? and caption=?", (6, c, _))
                elif "无价格涨跌幅限制" in _:
                    cu.execute("update dragon_tiger set caption_mark=? where code=? and caption=?", (7, c, _))
            conn.commit()
