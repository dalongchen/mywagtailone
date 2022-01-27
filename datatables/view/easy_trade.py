from .. import views
import os
import baostock as bs


# 查询当天收盘价并构建数据给交易软件。查询收盘价和名字
def inquiry_close(stock_list, d_trade):
    lg = bs.login()
    stock = []
    if lg.error_code == "0":
        for item in stock_list:
            st = {}
            print(item)
            rs = bs.query_history_k_data_plus(item, "close,", start_date=d_trade, frequency="d", adjustflag="3")
            if (rs.error_code != '0') or not rs.next():
                st["da"] = d_trade  # 日期
                st["xd_2102"] = item[3:]   # 代码
                st["xd_2127"] = 0  # 收盘价
                # print("result", rs.next())
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
            if (res.error_code != '0') or not res.next():
                st["xd_2103"] = "有误"
                # print("result", res.next())
            while (res.error_code == '0') & res.next():
                # 获取一条记录，将记录合并在一起
                result = res.get_row_data()
                # print(result[1])
                if len(result):
                    if result[1] != "":
                        st["xd_2103"] = result[1]
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
