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

            cu.execute("select code FROM dragon_tiger_all_inst_lgt2 where date=? and code=?", ("2021-06-17 00:00:00", "605499"))
            row = cu.fetchall()
            print(row.__len__())
            dd = ""
            if dd:
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
    dd = "q"
    if dd:
        caption = "单只标的证券的当日融n资买入数量达到当日该证券总交易量的50％以上"
        # caption = "只标的证券的当日融资买入数量达到当日该证券总"
        # if "退市整理" not in caption:
        if "当日融资买入数量达到" not in caption:
            print(caption)
        print("ee")
    dd = ""
    if dd:
        rows = [('2020-11-19 00:00:00', '600158',
                 275485153.39,
                 182177486.99,
                 93307666.4,
                 2184420403,
                 9.9862,
                 457662640.38,
                 20.95121615562,
                 1,
                 8.9247,
                 4.271506815806,
                 1,
                 0,
                 0,
                 56259952.25,
                 0,
                 56259952.25,
                 0,
                 0,
                 0),
                ('2020-11-19 00:00:00',
                 '002612',
                 86313227.8,
                 56149002.16,
                 30164225.64,
                 754092123,
                 10.0129,
                 142462229.96,
                 18.891886762223,
                 3,
                 17.6107,
                 4.000071704767,
                 1,
                 0,
                 0,
                 16428979,
                 7974938.8,
                 8454040.2,
                 0,
                 0,
                 0)]
        pprint(list(rows[0])[0:2])
        pprint(rows[0][0:2])
        # pprint(rows[:2][0][0:2])
    dd = ""
    if dd:
        y_true = [1, 2, 1, 2, 1]
        y_predict = [1.2, 2.1, 0.7, 2.1, 1.8]
        s = 0
        for index, item in enumerate(y_true):
            print(index, item)
            pp = y_predict[index]
            if item == 1:
                if (pp > 0.75) and (pp < 1.25):
                    s += 1
            elif item == 2:
                if (pp > 1.75) and (pp < 2.25):
                    s += 1
            else:
                pass
        print(s)
    dd = ""
    if dd:
        """无法float [27771779.92, 13511036.92, 14260743, 174553524, 22.48, 2.6016, 41282816.84, 23.650520421461, 6, 1.9018, 8.169839641851, 9138717968, 2, 0, 1, 9667950.16, 0, 9667950.16, '7712355', '', 7712355.0, '22.0892751200']
    无法float [197254368.05, 116061388.51, 81192979.54, 2866279729, 17.36, 10.0127, 313315756.56, 10.931094875004, 1, 21.7542, 2.832695592078, 7076587833.28, 1, 0, 1, 41669308.78, 0, 41669308.78, '33577503.68', '', 33577503.68, '17.0000000000']
    无法float [789976694.87, 740737016.04, 49239678.8300001, 3770269652, 19.14, 3.9652, 1530713710.91, 40.599581785828, 1, 6.7571, 1.305998864136, 31779351552.3, 3, 2, 1, 502912042.74, 309752937.64, 193159105.1, '204455121.36', '', 204455121.36, '19.3300000000']
    无法float [68428647.76, 50075579.98, 18353067.78, 683016797, 21.09, 10.0156, 118504227.74, 17.350119098169, 3, 4.052, 2.687059507264, 17324067967.29, 1, 0, 1, 19865258.33, 0, 19865258.33, '10758389.85', '', 10758389.85, '20.6300000000']
    无法float [86767269.76, 66561313, 20205956.76, 512697613, 6.91, 10.0318, 153328582.76, 29.906240807874, 3, 4.8954, 3.941106072596, 10570627469.05, 2, 0, 1, 20115671.42, 0, 20115671.42, '47312313.34', '', 47312313.34, '7.0000000000']
    """
        line = [27771779.92, 13511036.92, 14260743, 174553524, 22.48, 2.6016, 41282816.84, 23.650520421461, 6, 1.9018, 8.169839641851, 9138717968, 2, 0, 1, 9667950.16, 0, 9667950.16, '7712355', '0', 7712355.0, '22.0892751200']
        values = [float(x) for x in line]
    dd = ""
    if dd:
        li = [10, 8, 9, 26, 72, 6, 28]
        # uu = list(enumerate(li))
        uu = dict(enumerate(li))
        print(uu)
        print(uu.get("3"))
        print(uu.get("0"))
        c = ["退市整理hugu", "整理hugu", "退市", "单只标的证券的当日融资买入数量达到当日该证券总交易量", "hsl", "这个"]
        for caption in c:
            if ("退市整理" not in caption) and ("单只标的证券的当日融资买入数量达到当日该证券总交易量" not in caption):
                print(caption)
        a = [7, 1, 2, 5, 3]
        b = [2, 6, 3, 4]
        ret = [i for i in b if i in a]
        # ret = [i for i in b if i not in a]
        # ret = [i for i in a if i not in b]
        # ret = list(set(a) ^ set(b))
        # ret = list(set(a).difference(set(b)))
        print(ret)
        # import numpy as np
        # a = np.arange(24).reshape(2, 3, 4)
        # print("sa", a[:, -1, 1])
        # print("a", a)


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


# 读东财机构,all,陆股通龙虎榜
def east_dragon_tiger4():
    # start = "2020-11-19"
    start = "2021-12-01"
    end = "2021-12-03"
    # start = "2018-10-1"
    # end = "2018-10-30"
    # end = "2020-11-18"
    # east_dragon_tiger_all(start, end)  # 读东财龙虎榜all
    # east_dragon_tiger_inst(start, end)  # 读东财机构龙虎榜

    # net = "http://datainterface3.eastmoney.com/EM_DataCenter_V3/api/YYBJXMX/GetYYBJXMX?js=&sortfield=&sortdirec=-1&pageSize={}&pageNum={}&tkn=eastmoney&salesCode=80601499&tdir=&dayNum=&startDateTime={}&endDateTime={}&cfg=yybjymx"
    # net = "http://datainterface3.eastmoney.com/EM_DataCenter_V3/api/YYBJXMX/GetYYBJXMX?js=&sortfield=&sortdirec=-1&pageSize={}&pageNum={}&tkn=eastmoney&salesCode=80403915&tdir=&dayNum=&startDateTime={}&endDateTime={}&cfg=yybjymx"
    # east_dragon_tiger_lgt(net, start, end)  # 读东财陆股通龙虎榜 第一种

    net = "https://datacenter-web.eastmoney.com/api/data/v1/get?callback=&sortColumns=TRADE_DATE%2CSECURITY_CODE&sortTypes=-1%2C1&pageSize={}&pageNumber={}&reportName=RPT_OPERATEDEPT_TRADE_DETAILS&columns=ALL&filter=(OPERATEDEPT_CODE%3D%2210634757%22)&source=WEB&client=WEB"
    # net = "https://datacenter-web.eastmoney.com/api/data/v1/get?callback=&sortColumns=TRADE_DATE%2CSECURITY_CODE&sortTypes=-1%2C1&pageSize={}&pageNumber={}&reportName=RPT_OPERATEDEPT_TRADE_DETAILS&columns=ALL&filter=(OPERATEDEPT_CODE%3D%2210434470%22)&source=WEB&client=WEB"
    # east_dragon_tiger_lgt2(net)  # 读东财陆股通龙虎榜 第2种


# 读东财龙虎榜all净买入大于0
def east_dragon_tiger_all(start="", end=""):
    import os
    import sqlite3
    from mywagtailone.datatables.tool import mysetting
    import time
    import demjson
    if os.path.isfile(mysetting.DATA_TABLE_DB):
        with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
            cu = conn.cursor()
            # 龙虎榜净买入排序,一次只能读取500条
            net = "https://datacenter-web.eastmoney.com/api/data/v1/get?callback=&sortColumns=BILLBOARD_NET_AMT%2CTRADE_DATE%2CSECURITY_CODE&sortTypes=-1%2C-1%2C1&pageSize={}&pageNumber={}&reportName=RPT_DAILYBILLBOARD_DETAILS&columns=SECURITY_CODE%2CSECUCODE%2CSECURITY_NAME_ABBR%2CTRADE_DATE%2CEXPLAIN%2CCLOSE_PRICE%2CCHANGE_RATE%2CBILLBOARD_NET_AMT%2CBILLBOARD_BUY_AMT%2CBILLBOARD_SELL_AMT%2CBILLBOARD_DEAL_AMT%2CACCUM_AMOUNT%2CDEAL_NET_RATIO%2CDEAL_AMOUNT_RATIO%2CTURNOVERRATE%2CFREE_MARKET_CAP%2CEXPLANATION%2CD1_CLOSE_ADJCHRATE%2CD2_CLOSE_ADJCHRATE%2CD5_CLOSE_ADJCHRATE%2CD10_CLOSE_ADJCHRATE&source=WEB&client=WEB&filter=(TRADE_DATE%3C%3D%27{}%27)(TRADE_DATE%3E%3D%27{}%27)"
            page = 1
            for t in range(1, 210):
                if t > page:
                    break
                time.sleep(3)
                print("t", t)
                dragon_t = requests.get(net.format(500, t, end, start))  # 日期反
                dragon_tiger = dragon_t.text
                # print(dragon_tiger)
                if dragon_t.status_code == 200 and dragon_tiger:
                    dragon_tiger = demjson.decode(dragon_tiger)
                    """result: {pages: 2, data
                    市场总成交金额 ACCUM_AMOUNT: 3599125579
                    买入金额 BILLBOARD_BUY_AMT: 506971623.79
                    龙虎榜成交额 BILLBOARD_DEAL_AMT: 732885293.85
                    净买入金额 BILLBOARD_NET_AMT: 281057953.73
                    卖出金额 BILLBOARD_SELL_AMT: 225913670.06
                    CHANGE_RATE: 10.0352
                    CLOSE_PRICE: 12.5
                    D1_CLOSE_ADJCHRATE: null
                    D2_CLOSE_ADJCHRATE: null
                    D5_CLOSE_ADJCHRATE: null
                    D10_CLOSE_ADJCHRATE: null
                    成交额占总成交比 DEAL_AMOUNT_RATIO: 20.362870863029
                    净买额占总成交比 DEAL_NET_RATIO: 7.809062161373
                    EXPLAIN: "实力游资买入，成功率43.90%"
                    EXPLANATION: "日涨幅偏离值达到7%的前5只证券"
                    总市值 FREE_MARKET_CAP: 49627862500
                    SECUCODE: "000723.SZ"
                    SECURITY_CODE: "000723"
                    SECURITY_NAME_ABBR: "美锦能源"
                    TRADE_DATE: "2021-11-19 00:00:00"
                    换手率 TURNOVERRATE: 7.4719"""
                    d = dragon_tiger.get('result', '')
                    if t == 1:  # 第一页时获取总页数
                        page = d.get('pages', '')
                        print("page", page)
                    dr = d.get('data', '')
                    print("dr", len(dr))
                    # dr = ""
                    if dr:
                        for i in dr:
                            # print(i.get('BILLBOARD_NET_AMT', '0'))
                            if float(i.get('BILLBOARD_NET_AMT', '0')) > 0 and float(i.get('CHANGE_RATE', '0')) > 0:
                                # print(i.get('NET_BUY_AMT', '0'))
                                caption = i.get('EXPLANATION', ''),  # 上榜原因explanation
                                if "退市整理" not in caption:
                                    if "当日融资买入数量达到" not in caption:
                                        caption, caption_mark = get_caption_mark(caption)  # 上榜原因explanation
                                        ii = [
                                            i.get('TRADE_DATE', ''),
                                            i.get('SECURITY_CODE', ''),
                                            i.get('SECURITY_NAME_ABBR', ''),  # 名字security_name_abbr
                                            i.get('BILLBOARD_BUY_AMT', '0'),  # 卖入
                                            i.get('BILLBOARD_SELL_AMT', '0'),  # 卖出
                                            i.get('BILLBOARD_NET_AMT', '0'),  # 净买入
                                            i.get('ACCUM_AMOUNT', '0'),  # 市场总成交金额'accum_amount'
                                            i.get('CLOSE_PRICE', '0'),
                                            i.get('CHANGE_RATE', '0'),  # 涨幅
                                            i.get('BILLBOARD_DEAL_AMT', '0'),  # 龙虎榜成交额 billboard_deal_amt,1122
                                            i.get('DEAL_AMOUNT_RATIO', '0'),  # 成交额占总成交比 deal_amount_ratio 1122
                                            caption,  # 上榜原因
                                            caption_mark,  # 上榜原因标记
                                            i.get('TURNOVERRATE', '0'),  # 换手率'turnoverrate'
                                            i.get('DEAL_NET_RATIO', '0'),  # 净买额占总成交比 deal_net_ratio  1122
                                            i.get('FREE_MARKET_CAP', '0')  # 总市值 free_market_cap  1122
                                        ]
                                        cu.execute("INSERT INTO dragon_tiger_all VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ii)
                                        # cu.execute("INSERT INTO dragon_tiger_all VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ii)
            cu.close()


# 上榜原因explanation
def get_caption_mark(caption):
    if isinstance(caption, tuple):
        # print(caption)
        caption = caption[0]
        # print(caption)
    if (("ST、*ST和S证券连续三个交易日内收盘价格涨幅偏离值累计达到" in caption)
            or ("非ST、*ST和S证券连续三个交易日内收盘价格涨幅偏离值累计达到" in caption)
            or ("有价格涨跌幅限制的连续3个交易日内收盘价格涨幅偏离值累计达到" in caption)
            or ("连续三个交易日内，涨幅偏离值累计达到" in caption)
            or ("连续三个交易日内,涨幅偏离值累计达到" in caption)
            or ("异常期间价格涨幅偏离值累计达到" in caption)):
        caption_mark = 1
    elif (("ST、*ST和S证券连续三个交易日内收盘价格跌幅偏离值累计达到" in caption)
          or ("非ST、*ST和S证券连续三个交易日内收盘价格跌幅偏离值累计达到" in caption)
          or ("有价格涨跌幅限制的连续3个交易日内收盘价格跌幅偏离值累计达到" in caption)
          or ("连续三个交易日内，跌幅偏离值累计达到" in caption)
          or ("异常期间价格跌幅偏离值累计达到" in caption)):
        caption_mark = 2
    elif (("日涨幅偏离值达到" in caption)
          or ("日价格涨幅偏离值达到" in caption)
          or ("有价格涨跌幅限制的日收盘价格涨幅偏离值达到" in caption)
          or ("有价格涨跌幅限制的日收盘价格涨幅达到" in caption)
          or ("日涨幅达到" in caption)
          or ("日价格涨幅达到" in caption)):
        caption_mark = 3
    elif (("日跌幅偏离值达到" in caption)
          or ("日跌幅达到" in caption)
          or ("日价格跌幅偏离值达到" in caption)
          or ("有价格涨跌幅限制的日收盘价格跌幅偏离值达到" in caption)
          or ("有价格涨跌幅限制的日收盘价格跌幅达到" in caption)):
        caption_mark = 4
    elif (("日换手率达到" in caption)
          or ("连续三个交易日内，日均换手率" in caption)
          or ("连续三个交易日内的日均换手率" in caption)
          or ("异常期间日均换手率" in caption)
          or ("有价格涨跌幅限制的日换手率达到" in caption)):
        caption_mark = 5
    elif (("有价格涨跌幅限制的日价格振幅达到" in caption)
          or ("日振幅值达到" in caption)
          or ("日价格振幅达到" in caption)):
        caption_mark = 6
    elif "无价格涨跌幅限制" in caption:
        caption_mark = 7
    else:
        caption_mark = 8  # 需要修改数据库
    return caption, caption_mark


# 读东财机构龙虎榜
def east_dragon_tiger_inst(start, end):
    import os
    import sqlite3
    from mywagtailone.datatables.tool import mysetting
    import time
    import demjson
    if os.path.isfile(mysetting.DATA_TABLE_DB):
        with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
            cu = conn.cursor()
            # 读东财龙虎榜 机构龙虎榜净买入 日期排序,一次只能读取500
            net = "https://datacenter-web.eastmoney.com/api/data/v1/get?callback=&sortColumns=NET_BUY_AMT%2CTRADE_DATE%2CSECURITY_CODE&sortTypes=-1%2C-1%2C1&pageSize={}&pageNumber={}&reportName=RPT_ORGANIZATION_TRADE_DETAILS&columns=ALL&source=WEB&client=WEB&filter=(TRADE_DATE%3E%3D%27{}%27)(TRADE_DATE%3C%3D%27{}%27)"
            page = 1
            for t in range(1, 150):
                if t > page:
                    break
                print("t", t)
                # print("t", net.format(500, t, start, end))
                organization_dragon_tiger = requests.get(net.format(500, t, start, end))
                if organization_dragon_tiger and organization_dragon_tiger.status_code == 200:
                    """{"version":"9471f341535aed1787fa9300a3c35475","result":{"pages":1,"data":
                    [{"SECUCODE":"603606.SH",
                    "SECURITY_NAME_ABBR":"东方电缆",
                    "SECURITY_CODE":"603606",
                    "TRADE_DATE":"2021-11-18 00:00:00",
                    "CLOSE_PRICE":60.1,
                    涨幅"CHANGE_RATE":1.1955,
                    "BUY_TIMES":3,
                    "SELL_TIMES":2,
                    买入金额"BUY_AMT":583387049.87,
                    卖出金额"SELL_AMT":167391913.66,
                    净买入金额 "NET_BUY_AMT":415995136.21,
                    市场总成交金额"ACCUM_AMOUNT":5500295190,
                    净买额占总成交比"RATIO":7.563142010384,   2222
                    换手率"TURNOVERRATE":5.199,
                    总市值"FREECAP":404.17,
                    "EXPLANATION":"非ST、*ST和S证券连续三个交易日内收盘价格涨幅偏离值累计达到20%的证券","D1_CLOSE_ADJCHRATE":-2.11314476,"D2_CLOSE_ADJCHRATE":null,"D3_CLOSE_ADJCHRATE":null,"D5_CLOSE_ADJCHRATE":null,"D10_CLOSE_ADJCHRATE":null,"MARKET":"沪市","TRADE_MARKET_CODE":"069001001001"},
"""
                    o = organization_dragon_tiger.text
                    # print(o)
                    org = demjson.decode(o).get('result', '')
                    if t == 1:  # 第一页时获取总页数
                        page = org.get('pages', '')
                        print("page", page)
                    da = org.get('data', '')
                    print("da", len(da))
                    for i in da:
                        try:
                            net_buy = float(i.get('NET_BUY_AMT', '0'))
                            up_range = float(i.get('CHANGE_RATE', '0'))  # 涨幅
                        except ValueError:
                            print("转换float失败" + i)
                        if net_buy > 0 and up_range > 0:  # 净买入,涨幅大于0
                            # print(i.get('NET_BUY_AMT', '0'))
                            caption = i.get('EXPLANATION', ''),  # 上榜原因explanation
                            if ("退市整理" not in caption) and ("单只标的证券的当日融资买入数量达到当日该证券总交易量" not in caption):
                                caption, caption_mark = get_caption_mark(caption)  # 上榜原因explanation
                                """"SECURITY_NAME_ABBR":"东方电缆",
                    "SECURITY_CODE":"603606",
                    "TRADE_DATE":"2021-11-18 00:00:00",
                    "CLOSE_PRICE":60.1,
                    涨幅"CHANGE_RATE":1.1955,
                    "BUY_TIMES":3,
                    "SELL_TIMES":2,
                    买入金额"BUY_AMT":583387049.87,
                    卖出金额"SELL_AMT":167391913.66,
                    净买入金额 "NET_BUY_AMT":415995136.21,
                    市场总成交金额"ACCUM_AMOUNT":5500295190,
                    净买额占总成交比"RATIO":7.563142010384,   2222
                    换手率"TURNOVERRATE":5.199,
                    流通市值"FREECAP":404.17,
                    "EXPLANATION":"非ST、*ST和S证券连续三个交易日内收盘价格涨幅偏离值累计达到20%的证券"""
                                ii = [
                                    i.get('TRADE_DATE', ''),
                                    i.get('SECURITY_CODE', ''),
                                    i.get('SECURITY_NAME_ABBR', ''),  # 名字security_name_abbr
                                    i.get('BUY_AMT', '0'),  # 卖入
                                    i.get('SELL_AMT', '0'),  # 卖出
                                    net_buy,  # 净买入
                                    i.get('ACCUM_AMOUNT', '0'),  # 市场总成交金额'accum_amount'
                                    i.get('CLOSE_PRICE', '0'),
                                    up_range,  # 涨幅
                                    i.get('RATIO', '0'),  # 净买额占总成交比 RATIO
                                    # i.get('BILLBOARD_DEAL_AMT', '0'),  # 龙虎榜成交额 billboard_deal_amt,
                                    # i.get('DEAL_AMOUNT_RATIO', '0'),  # 成交额占总成交比 deal_amount_ratio
                                    caption,  # 上榜原因
                                    caption_mark,  # 上榜原因标记
                                    i.get('TURNOVERRATE', '0'),  # 换手率'turnoverrate'
                                    i.get('BUY_TIMES', '0'),  # 买入机构数量
                                    i.get('SELL_TIMES', '0'),  # 卖出机构数量
                                    i.get('FREECAP', '0')  # 流通市值 free_market_cap
                                ]
                                cu.execute("INSERT INTO dragon_tiger_inst VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ii)
                time.sleep(3)
            cu.close()


# 读东财陆股通龙虎榜 第一种
def east_dragon_tiger_lgt(net, start, end):
    import os
    import sqlite3
    from mywagtailone.datatables.tool import mysetting
    import time
    import demjson
    if os.path.isfile(mysetting.DATA_TABLE_DB):
        with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
            cu = conn.cursor()
            page = 1
            for t in range(1, 150):
                if t > page:
                    break
                print("t", t)
                # print("t", net.format(500, t, start, end))
                organization_dragon_tiger = requests.get(net.format(500, t, start, end))
                if organization_dragon_tiger and organization_dragon_tiger.status_code == 200:
                    """{'Data': [{'ConsumeMSecond': 24,
           'Data': ['255802070.97|44.94857834||||深股通专用||5.9708|许继电气|23.96|000400||||177984704.55|260947220.75|82.968433|-5145149.78|||186627266.97||2021-11-19||28.8172043|连续三个交易日内，涨幅偏离值累计达到20%的证券||||80601499|||53.05344206',
                    '8002767.36|25.10638298||||深股通专用||10.1124|利欧股份|2.94|002131||||39355858.1|39355858.1|25.6479581|-31353090.74|||8002767.36||2021-11-18||25.64102564|连续三个交易日内，涨幅偏离值累计达到20%的证券||||80601499|0.3401||-11.06245011'],
           'FieldName': 'BMoney,RChange3M,RChange20DC,RChange15DC,RChange10DC,SalesName,RChange3DC,ChgRadio,SName,CPrice,SCode,RChange5DC,RChange20DO,RChange2DC,ActSellNum,SMoney,RChange6M,PBuy,RChange30DC,RChange15DO,ActBuyNum,RChange3DO,TDate,RChange10DO,RChange1M,CTypeDes,RChange5DO,RChange1DO,RChange30DO,SalesCode,RChange1DC,RChange2DO,RChange1Y',
           'SplitSymbol': '|',
           'TableName': 'RptLHBYYBJYMXMap',
           'TotalPage': 1}],"""
                    # o = organization_dragon_tiger.text
                    dr = demjson.decode(organization_dragon_tiger.text)
                    # pprint(dr)
                    if dr:
                        dd = dr.get('Data', '')[0]
                    if t == 1:  # 第一页时获取总页数
                        page = dd.get('TotalPage', '')
                        print("page", page)
                    da = dd.get('Data', '')
                    print("每页条数", len(da))
                    """{0: '255802070.97', 1: '44.94857834', 2: '', 3: '', 4: '', 5: '深股通专用', 6: '',
                    涨幅 7: '5.9708', 8: '许继电气', 9: '23.96', 10: '000400', 11: '', 12: '', 13: '', 14: '177984704.55', 15: '260947220.75', 16: '82.968433',
                    净买入 17: '-5145149.78', 18: '', 19: '', 20: '186627266.97', 21: '', 22: '2021-11-19', 23: '', 24: '28.8172043',
                    25: '连续三个交易日内，涨幅偏离值累计达到20%的证券', 26: '', 27: '', 28: '', 29: '80601499', 30: '', 31: '', 32: '53.05344206'}"""
                    for ii in da:
                        i = ii.split("|")
                        # i = dict(enumerate(i))
                        # print("i", i)
                        try:
                            # print("17", i.get('17', '0'))
                            net_buy = float(i[17])
                            # print("net_buy", net_buy)
                            up_range = float(i[7])  # 涨幅
                        except ValueError:
                            print("转换float失败" + i)
                        if net_buy > 0 and up_range > 0:  # 净买入,涨幅大于0
                            # print("25", i.get('25', '0'))
                            caption = i[25],  # 上榜原因explanation
                            if ("退市整理" not in caption) and ("单只标的证券的当日融资买入数量达到当日该证券总交易量" not in caption):
                                caption, caption_mark = get_caption_mark(caption)  # 上榜原因explanation
                                # if isinstance(caption, tuple):
                                #     caption = caption[0]
                                #     # print(caption)
                                # if (("ST、*ST和S证券连续三个交易日内收盘价格涨幅偏离值累计达到" in caption)
                                #         or ("非ST、*ST和S证券连续三个交易日内收盘价格涨幅偏离值累计达到" in caption)
                                #         or ("有价格涨跌幅限制的连续3个交易日内收盘价格涨幅偏离值累计达到" in caption)
                                #         or ("连续三个交易日内，涨幅偏离值累计达到" in caption)
                                #         or ("连续三个交易日内,涨幅偏离值累计达到" in caption)
                                #         or ("异常期间价格涨幅偏离值累计达到" in caption)):
                                #     caption_mark = 1
                                # elif (("ST、*ST和S证券连续三个交易日内收盘价格跌幅偏离值累计达到" in caption)
                                #       or ("非ST、*ST和S证券连续三个交易日内收盘价格跌幅偏离值累计达到" in caption)
                                #       or ("有价格涨跌幅限制的连续3个交易日内收盘价格跌幅偏离值累计达到" in caption)
                                #       or ("连续三个交易日内，跌幅偏离值累计达到" in caption)
                                #       or ("异常期间价格跌幅偏离值累计达到" in caption)):
                                #     caption_mark = 2
                                # elif (("日涨幅偏离值达到" in caption)
                                #       or ("日价格涨幅偏离值达到" in caption)
                                #       or ("有价格涨跌幅限制的日收盘价格涨幅偏离值达到" in caption)
                                #       or ("有价格涨跌幅限制的日收盘价格涨幅达到" in caption)
                                #       or ("日涨幅达到" in caption)
                                #       or ("日价格涨幅达到" in caption)):
                                #     caption_mark = 3
                                # elif (("日跌幅偏离值达到" in caption)
                                #       or ("日跌幅达到" in caption)
                                #       or ("日价格跌幅偏离值达到" in caption)
                                #       or ("有价格涨跌幅限制的日收盘价格跌幅偏离值达到" in caption)
                                #       or ("有价格涨跌幅限制的日收盘价格跌幅达到" in caption)):
                                #     caption_mark = 4
                                # elif (("日换手率达到" in caption)
                                #       or ("连续三个交易日内，日均换手率" in caption)
                                #       or ("有价格涨跌幅限制的日换手率达到" in caption)):
                                #     caption_mark = 5
                                # elif (("有价格涨跌幅限制的日价格振幅达到" in caption)
                                #       or ("日振幅值达到" in caption)
                                #       or ("日价格振幅达到" in caption)):
                                #     caption_mark = 6
                                # elif "无价格涨跌幅限制" in caption:
                                #     caption_mark = 7
                                # else:
                                #     caption_mark = 8  # 需要修改数据库
                                """{卖入 0: '255802070.97', 1: '44.94857834', 2: '', 3: '', 4: '',
                                 5: '深股通专用', 6: '',
                                涨幅 7: '5.9708',
                                名字 8: '许继电气',
                                价格 9: '23.96',
                                代码 10: '000400', 11: '', 12: '', 13: '', 14: '177984704.55',
                                卖出 15: '260947220.75', 16: '82.968433',
                                净买入 17: '-5145149.78', 18: '', 19: '', 20: '186627266.97', 21: '',
                                日期 22: '2021-11-19', 23: '', 24: '28.8172043',
                                25: '连续三个交易日内，涨幅偏离值累计达到20%的证券', 26: '', 27: '', 28: '', 29: '80601499', 30: '', 31: '', 32: '53.05344206'}"""
                                ii = [
                                    i[22],  # 日期
                                    i[10],  # 代码
                                    i[8],  # 名字
                                    i[0],  # 卖入
                                    i[15],  # 卖出
                                    net_buy,  # 净买入
                                    # i.get('ACCUM_AMOUNT', '0'),  # 市场总成交金额'accum_amount'
                                    i[9],  # 价格
                                    up_range,  # 涨幅
                                    caption,  # 上榜原因
                                    caption_mark,  # 上榜原因标记
                                    i[5]  # 陆股通mark
                                ]
                                cu.execute("INSERT INTO dragon_tiger_lgt VALUES(?,?,?,?,?,?,?,?,?,?,?)", ii)
                time.sleep(3)
            cu.close()


# 读东财陆股通龙虎榜 第2种
def east_dragon_tiger_lgt2(net):
    import os
    import sqlite3
    from mywagtailone.datatables.tool import mysetting
    import time
    import demjson
    if os.path.isfile(mysetting.DATA_TABLE_DB):
        with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
            cu = conn.cursor()
            page = 1
            for t in range(1, 150):
                if t > page:
                    break
                print("t", t)
                # print("t", net.format(500, t, start, end))
                organization_dragon_tiger = requests.get(net.format(500, t))
                if organization_dragon_tiger and organization_dragon_tiger.status_code == 200:
                    # o = organization_dragon_tiger.text
                    dr = demjson.decode(organization_dragon_tiger.text)
                    # pprint(dr)
                    if dr:
                        dd = dr.get('result', '')
                        if t == 1:  # 第一页时获取总页数
                            page = dd.get('pages', '')
                            print("page", page)
                        da = dd.get('data', '')
                        print("每页条数", len(da))
                        """ACT_BUY: 113239969.6
                        ACT_SELL: 63940041.78
                        CHANGE_RATE: -9.9952
                        D1_CLOSE_ADJCHRATE: null
                        D2_CLOSE_ADJCHRATE: null
                        D3_CLOSE_ADJCHRATE: null
                        D5_CLOSE_ADJCHRATE: null
                        D10_CLOSE_ADJCHRATE: null
                        D20_CLOSE_ADJCHRATE: null
                        D30_CLOSE_ADJCHRATE: null
                        EXPLANATION: "日跌幅偏离值达到7%的前5只证券"
                        NET_AMT: 49299927.82
                        OPERATEDEPT_CODE: "10634757"
                        OPERATEDEPT_CODE_OLD: "80601499"
                        OPERATEDEPT_NAME: "深股通专用"
                        ORG_NAME_ABBR: "深股通专用"
                        SECUCODE: "000657.SZ"
                        SECURITY_CODE: "000657"
                        SECURITY_NAME_ABBR: "中钨高新"
                        TRADE_DATE: 2021-11-30 00:00:00"""
                        for ii in da:
                            try:
                                net_buy = float(ii.get('NET_AMT', '0'))
                                # print("net_buy", net_buy)
                                up_range = float(ii.get('CHANGE_RATE', '0'))  # 涨幅
                            except ValueError:
                                print("转换float失败" + ii)
                            if net_buy > 0 and up_range > 0:  # 净买入,涨幅大于0
                                caption = ii.get('EXPLANATION', ''),  # 上榜原因explanation
                                if ("退市整理" not in caption) and ("单只标的证券的当日融资买入数量达到当日该证券总交易量" not in caption):
                                    caption, caption_mark = get_caption_mark(caption)  # 上榜原因explanation
                                    ii = [
                                        ii.get('TRADE_DATE', ''),  # 日期
                                        ii.get('SECURITY_CODE', ''),  # 代码
                                        ii.get('SECURITY_NAME_ABBR', ''),  # 名字
                                        ii.get('ACT_BUY', '0'),  # 卖入
                                        ii.get('ACT_SELL', '0'),  # 卖出
                                        net_buy,  # 净买入
                                        0,  # 价格
                                        up_range,  # 涨幅
                                        caption,  # 上榜原因
                                        caption_mark,  # 上榜原因标记
                                        ii.get('ORG_NAME_ABBR', '')  # 陆股通mark
                                    ]
                                    cu.execute("INSERT INTO dragon_tiger_lgt VALUES(?,?,?,?,?,?,?,?,?,?,?)", ii)
                time.sleep(3)
            cu.close()


# 读东财机构,all,陆股通龙虎榜表入robot stock dragon_tiger_all_inst_lgt2数据表 f == "inst"
def read_dragon_tiger_robot_stock(f):
    import os
    import sqlite3
    from mywagtailone.datatables.tool import mysetting
    if os.path.isfile(mysetting.DATA_TABLE_DB):
        with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
            cu = conn.cursor()
            cu2 = conn.cursor()
            if f == "inst":
                cu3 = conn.cursor()
                cu.execute("select date,code,caption_mark,buy_num,sell_num,buy,sell,net_buy from dragon_tiger_inst where up_range<11 and buy_num>0")
                # cu.execute("select date,code,caption_mark,buy_num,sell_num,buy,sell,net_buy from dragon_tiger_inst where up_range<11 and buy_num>0 limit 0, 10")
                for ii in cu:
                    # print("ii", ii)
                    cu2.execute("select * from dragon_tiger_all where date=? and code=? and caption_mark=?", (ii[0], ii[1], ii[2]))
                    # cu2.execute("select date,code,caption_mark from dragon_tiger_all_181001_201118 where date=? and code=? and caption_mark=?", (ii[0], ii[1], ii[2]))
                    for iii in cu2:
                        iii = list(iii)
                        # print(iii)
                        iii += [
                            ii[3],  # buy_num
                            ii[4],  # sell_num
                            0,  # lgt_mark
                            ii[5],  # buy_inst
                            ii[6],  # sell_inst
                            ii[7],  # net_inst
                            0,  # buy_lgt
                            0,  # sell_lgt
                            0,  # net_lgt
                        ]
                        # 25
                        cu3.execute("INSERT INTO dragon_tiger_all_inst_lgt2 VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", iii)
                cu3.close()
            elif f == "lgt":
                cu.execute("select date,code,caption_mark,buy,sell,net_buy from dragon_tiger_lgt where up_range<11 AND date<'2021-12-01' AND date>'2020-11-18'")
                # cu.execute("select date,code,caption_mark,buy,sell,net_buy from dragon_tiger_lgt where up_range<11 AND date<'2020-11-19' AND date>'2018-10-01' limit 0, 10")
                for ii in cu:
                    # print("ii", ii)
                    tt = [
                        1,  # lgt_mark
                        ii[3],  # buy_lgt
                        ii[4],  # sell_lgt
                        ii[5],  # net_lgt
                        ii[0],  # date
                        ii[1],  # code
                        ii[2]  # caption_mark
                    ]
                    # print("tt", tt)
                    cu2.execute("update dragon_tiger_all_inst_lgt2 set lgt_mark=?,buy_lgt=?,sell_lgt=?,net_lgt=? where date=? and code=? and caption_mark=?", tt)
            cu.close()
            cu2.close()


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


# 读dragon_tiger_all_inst_lgt2龙虎榜日期和baostock日k线,把k数据插入dragon_tiger_all_inst_lgt2k
def dragon_tiger_date_mark(path):
    import sqlite3
    import os
    from ..tool import tools
    if os.path.isfile(path):
        with sqlite3.connect(path) as conn:
            cu = conn.cursor()
            # cu2 = conn.cursor()
            cu.execute("select id,date,code FROM dragon_tiger_all_inst_lgt2_181001_211130 ORDER BY date ASC")
            # cu.execute("select id,date,code FROM dragon_tiger_all_inst_lgt2_181001_211130 ORDER BY date ASC limit 300,5")
            import baostock as bs
            # 登陆系统
            # lg = bs.login()
            if lg.error_code == "0":
                stock_empty = []
                rows = cu.fetchall()
                # print(len(rows))
                for ii in rows:
                    trade_date = ii[1]
                    # print(ii)
                    # print(trade_date)
                    if len(trade_date) > 10:
                        trade_date = trade_date[:10]
                    # 如果当天+1为交易日则返回,否则返回两周内最近一个交易日add_subtract="subtract"后退
                    # f = "d"返回2021 - 07 - 01,f="t"为2021 - 07 - 01 00：00：00.取当天num=0，最近一天num=1
                    start = tools.get_late_trade_day_n(trade_date, add_subtract="sub",  num=5, f="t").strftime('%Y-%m-%d')
                    end = tools.get_late_trade_day_n(trade_date, add_subtract="add", num=3, f="t").strftime('%Y-%m-%d')
                    code = views.add_sh(ii[2], big="baostock")
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
                                                      start_date=start, end_date=end,
                                                      frequency="d", adjustflag="2")
                    rr = rs.error_code
                    if rr != '0':
                        print(trade_date + "error_code", code)
                    if rs.data.__len__() == 0:
                        stock_empty.append([trade_date, start, code])
                        print(trade_date + "-" + start, code + "空")
                    hh = 0
                    while (rr == '0') & rs.next():
                        # d = rs.get_row_data()
                        hh += 1
                        ss = [ii[0], hh]
                        ss += rs.get_row_data()
                        # print(ss)
                        try:
                            pass
                            # cu.execute("INSERT INTO dragon_tiger_all_inst_lgt2k_181001_211130_5 (dragon_id,son_id,date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ss)
                        except:
                            print(trade_date, ss)
            print("空", stock_empty)
            cu.close()
            # cu2.close()
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
    from ..tool import tools
    import sqlite3
    # print(stock_list)
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    # print('login respond error_code:' + lg.error_code)
    par = []
    end = tools.get_late_trade_day(start, add_subtract="add", f="t").strftime('%Y-%m-%d')
    print("end", end)
    if lg.error_code == "0":
        for c in stock_list:
            print(start, c)
            rs = bs.query_history_k_data_plus(c, "date,code,open,high,low,close,preclose,volume,amount,"
                                              "adjustflag,turn,"
                                              "tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
                                              start_date=start, end_date=end, frequency="d", adjustflag="3")
            # print('query_history_k_data_plus respond error_code:' + rs.error_code)
            rr = rs.error_code
            if rr != '0':
                print("error_code", c)
            if rs.data.__len__() == 0:
                print("hh", c + "空")
            while (rr == '0') & rs.next():
                # 获取一条记录，将记录合并在一起
                result = rs.get_row_data()
                print("gg", result)
                dd = ""
                if dd:
                    if len(result):
                        if result[11] != "1":
                            print(result[0], result[1] + "停牌")
                        path = r"D:\ana\envs\py36\mywagtailone\datatables\datatable.db"
                        with sqlite3.connect(path) as conn:
                            cu = conn.cursor()
                            print(result)
                            cu.execute(
                                "INSERT INTO dragon_tiger_all_inst_lgt2k_181001_211130_copy VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                result)
                            cu.close()

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


# 读dragon_tiger_all_inst_lgt2龙虎榜日期和dragon_tiger_all_inst_lgt2k日k线,把k数据插入dragon_tiger_all_inst_lgt2k_0 or 1,2
def dragon_tiger_date_mark_0(path, number=5):  # number=3取几条数据
    import sqlite3
    import os
    from ..tool import tools
    if os.path.isfile(path):
        with sqlite3.connect(path) as conn:
            cu = conn.cursor()
            cu.execute("select id,date,code FROM dragon_tiger_all_inst_lgt2_181001_211130 ORDER BY date ASC")
            # cu.execute("select id,date,code FROM dragon_tiger_all_inst_lgt2_181001_211130 ORDER BY date ASC limit 300,30")
            rows = cu.fetchall()
            # print(len(rows))
            for ii in rows:
                trade_date = ii[1]
                # print(ii)
                if len(trade_date) > 10:
                    trade_date = trade_date[:10]
                # 如果当天+1为交易日则返回,否则返回两周内最近一个交易日add_subtract="subtract"后退
                # f = "d"返回2021 - 07 - 01,f="t"为2021 - 07 - 01 00：00：00.取当天num=0，最近一天num=1
                code = views.add_sh(ii[2], big="baostock")
                dat = tools.get_late_trade_day_n(trade_date, add_subtract="add", num=2, f="t").strftime('%Y-%m-%d')
                kk = 0
                data_list = []
                for yy in range(0, 12):
                    aa = (ii[0], code, dat)
                    sql = "select * FROM dragon_tiger_all_inst_lgt2k_181001_211130_5 where dragon_id=? and code=? and date=? ORDER BY date ASC"
                    cu.execute(sql, aa)
                    r2 = cu.fetchall()
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
                    if r2.__len__() == 1:
                        qq = r2[-1]  # 取最后一天数据
                        # print(trade_date, qq)
                        pre_close = float(qq[8])
                        if pre_close and pre_close != 0:
                            r_up = abs((float(qq[7]) - pre_close) / pre_close)  # print(d[5], d[6])  # close, pre_close
                            if r_up < 0.11:
                                for d in r2:
                                    if d[13] == "0":
                                        print("停牌", (trade_date, d[2], d[3]))
                                    elif d[13] == "1":
                                        kk += 1
                                        d = list(d)
                                        d[1] = kk
                                        # st_id = [ii[0], kk]
                                        # st_id += list(d)
                                        # print(st_id)
                                        data_list.append(d)
                                    else:
                                        print("error,不为0，1", (trade_date, d[0], d[1]))
                            else:
                                print("涨幅r_up > 0.11:", r2[0][2:4])
                        else:
                            print("pre_close没有 or pre_close = 0:", r2)
                    else:
                        # pass
                        if yy > 10:
                            print("查询数量error" + trade_date, (dat, code, r2.__len__()))
                    dat = tools.get_late_trade_day_n(dat, add_subtract="sub", num=1, f="t").strftime(
                        '%Y-%m-%d')
                    # print("dat", dat)
                    if kk == number:
                        break
                if len(data_list) == number:
                    for rr in data_list:
                        oo = ""
                        if oo:
                            sql2 = "INSERT INTO dragon_tiger_all_inst_lgt2k_181001_211130_4 (dragon_id,son_id,date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM, isST) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                            cu.execute(sql2, rr)
                else:
                    print("数量不足", [trade_date,  len(data_list), data_list[0][2:4]])
            cu.close()
