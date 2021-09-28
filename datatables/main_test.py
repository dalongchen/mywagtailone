from pprint import pprint
import requests
from . import views


def test_sq_lite(path):
    import sqlite3
    from mywagtailone.datatables.tool import mysetting
    from .tool import tools
    import os
    print(os.path.isfile(path))
    if os.path.isfile(path):
        with sqlite3.connect(path) as conn:
            conn.text_factory = lambda x: str(x, 'gbk', 'ignore')
            cu = conn.cursor()
            # cu.execute("select * FROM ymd_1280194006")
            # cu.execute("select * FROM HDConfig")
            cu.execute("select * FROM HDProp")
            columns = [_[0].lower() for _ in cu.description]
            results = [dict(zip(columns, _)) for _ in cu]
            for ii in results:
                print(ii)
            # sq = "delete FROM ymd_1280194006 WHERE dat!='{}'".format(str(trade_date))
            # print(sq)
            # cu.execute("delete FROM ymd_1280194006 WHERE dat!='{}'".format(str(trade_date)))

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
            conn.text_factory = lambda x: str(x, 'gbk', 'ignore')
            cu = conn.cursor()
            cu.execute("select code FROM dragon_tiger order by date")
            t = []
            for ii in cu:
                # print(ii[0])
                code = views.code_add(ii[0]) + '\n'
                # print(code)
                t.append(code)
            print(t)
            cu.close()
            views.is_write_stock('ths_choice.blk', t, "write")
