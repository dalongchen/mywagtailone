from .. import views
import os


# 查询当天收盘价并构建数据给交易软件。查询收盘价和名字
def inquiry_close(stock_list, d_trade):
    import baostock as bs
    # from ..tool import mysetting
    # 登陆系统
    lg = bs.login()
    stock = []
    if lg.error_code == "0":
        for item in stock_list:
            # if item.startswith("sz"):
            #     xd_2106 = mysetting.HOLDER_CODE[0][0]  # 股东代码
            #     xd_2108 = mysetting.HOLDER_CODE[0][1]
            # elif item.startswith("sh"):
            #     xd_2106 = mysetting.HOLDER_CODE[1][0]
            #     xd_2108 = mysetting.HOLDER_CODE[1][1]
            # else:
            #     print(item, "error")
            #     xd_2106 = ""
            st = {}
            # print(item)
            rs = bs.query_history_k_data_plus(item, "close,", start_date=d_trade, frequency="d", adjustflag="3")
            # if not rs.next() or not rs:  # 如果bookstock没有则从新浪或其他获取
            #     # d_today = d_now.date()
            #     # iss = is_workday(d_now.date())
            #     # if is_workday(d_now.date()) and 9 >= d_now.hour >= 15:  # 如果是交易日且为9点到15点，从其他获取
            #     #     print(item, "book stock无数据，检查输入是否为未上市新股或日期（每天收盘5点后）或代码格式（sz,000001,sh.600000）")
            #     #     d = ""
            #     # else:
            #     #     d = views.sina_real_time(item[3:])  # 从新浪获取收盘价
            #     if d and xd_2106 != "":
            #         # print(d[3])
            #         # st["da"] = d_trade  # 日期
            #         st["xd_2102"] = item[3:]  # 代码 000001
            #         st["xd_2127"] = d[3]  # 收盘价
            #
            #         st["xd_2103"] = d[2]  # 名称
            #         st["xd_2106"] = xd_2106
            #         st["xd_2108"] = xd_2108
            #         st["xd_2109"] = buy
            #         st["xd_2126"] = "100"
                    # st["xd_2105"] = ""
                    # st["xd_3630"] = "0.000000"
            # print('respond error_code:' + rs.error_code)
            while (rs.error_code == '0') & rs.next():
                # 获取一条记录，将记录合并在一起
                result = rs.get_row_data()
                # print("item", result)
                if len(result):
                    # print(result[0])
                    if result[0] != "":
                        st["da"] = d_trade  # 日期
                        st["xd_2102"] = item[3:]  # 代码
                        st["xd_2127"] = result[0]  # 收盘价
                    else:
                        print("有误，空", item)
                else:
                    print("有误", item)

            # 获取证券基本资料
            res = bs.query_stock_basic(code=item)
            while (res.error_code == '0') & res.next():
                # 获取一条记录，将记录合并在一起
                result = res.get_row_data()
                # print(result)
                # print(result[1])
                if len(result):
                    if result[1] != "":
                        st["xd_2103"] = result[1]
                        # st["xd_2106"] = xd_2106
                        # st["xd_2108"] = xd_2108
                        # st["xd_2109"] = buy
                        # st["xd_2126"] = "100"
                        # st["xd_2105"] = ""
                        # st["xd_3630"] = "0.000000"
                    else:
                        print("有误，空", item)
                else:
                    print("有误", item)

            if st:
                stock.append(st)
    # 登出系统
    bs.logout()
    # print(stock)
    return stock


# 另存持仓数据
def trade_save():
    if os.path.isfile(r"D:\ana\envs\py36\mywagtailone\my_ignore\table.xls"):
        os.remove(r"D:\ana\envs\py36\mywagtailone\my_ignore\table.xls")
    dialog = views.log_on_ht()
    dia = dialog.window(best_match="Custom1", auto_id="1047", class_name="CVirtualGridCtrl", control_type="Pane")
    # dia.print_control_identifiers()
    dia.wait("visible", timeout=10, retry_interval=1)
    dia.type_keys("^s")
    dia.wait("visible", timeout=5, retry_interval=1)
    save = dialog.window(best_match="保存(S)", auto_id="1", class_name="Button", control_type="Button")
    # save.draw_outline()
    save.type_keys("{VK_RETURN}")
