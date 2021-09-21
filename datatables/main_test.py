from pprint import pprint


def test_sq_lite():
    import sqlite3
    from mywagtailone.datatables.tool import mysetting
    from .tool import tools
    import os
    trade_date = tools.get_date()  # 获取那一天数据
    print(trade_date)
    print(os.path.isfile(mysetting.DATA_TABLE_DB))
    with sqlite3.connect(mysetting.DATA_TABLE_DB) as conn:
        conn.text_factory = lambda x: str(x, 'gbk', 'ignore')
        cu = conn.cursor()
        # cu.execute("select * FROM ymd_1280194006")
        # columns = [_[0].lower() for _ in cu.description]
        # results = [dict(zip(columns, _)) for _ in cu]
        # for ii in results:
        #     if "发送成功" in ii['xd_2105']:
        #         ii['xd_2105'] = "已发送"
        # pprint(results)
        # sq = "delete FROM ymd_1280194006 WHERE dat!='{}'".format(str(trade_date))
        # print(sq)
        # cu.execute("delete FROM ymd_1280194006 WHERE dat!='{}'".format(str(trade_date)))
        stock_dict = [{
            'dat': '2021-09-07',
            'xd_2102': '600688',
            'xd_2103': '上海石化',
            'xd_2105': '',
            'xd_2106': '',
            'xd_2108': '',
            'xd_2109': '',
            'xd_2126': '',
            'xd_2127': '350.44',
            'xd_3630': ''},
         {
            'dat': '',
            'xd_2102': '002821',
            'xd_2103': '凯莱英',
            'xd_2105': '已发送',
            'xd_2106': '0280719034',
            'xd_2108': '深圳Ａ股',
            'xd_2109': '买入',
            'xd_2126': '100',
            'xd_2127': '350.44',
            'xd_3630': '64.000000'}]
        # for t in stock_dict:
        #     cu.execute(
        #         "INSERT INTO ymd_1280194006 (dat,xd_2102,xd_2103,xd_2106,xd_2109,xd_2127,xd_2126,xd_2108,xd_2105,xd_3630) VALUES(?,?,?,?,?,?,?,?,?,?)",
        #         (
        #             t["dat"],
        #             t["xd_2102"],
        #             t["xd_2103"].encode(encoding='utf8'),
        #             t["xd_2106"],
        #             t["xd_2109"].encode(encoding='gbk'),
        #             t["xd_2127"],
        #             t["xd_2126"],
        #             t["xd_2108"].encode(encoding='gbk'),
        #             t["xd_2105"],
        #             t["xd_3630"],
        #         )
        #     )

        cu.close()


def test_my():
    a = [7, 1, 2, 5, 3]
    b = [2, 6, 3, 4]
    ret = [i for i in a if i not in b]
    # ret = list(set(a) ^ set(b))
    # ret = list(set(a).difference(set(b)))
    print(ret)


