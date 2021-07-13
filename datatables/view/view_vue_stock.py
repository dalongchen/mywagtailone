from pprint import pprint
from time import sleep
import requests
import json
from datetime import date as d_date
from datetime import datetime, timedelta
from .. import views


# 东财公告利好 vue stock
def good_notice_parent(headers, start):
    d = []
    w = []
    c = []
    url = "http://np-anotice-stock.eastmoney.com/api/security/ann?cb=&sr=-1&page_size=100&page_index={}&ann_type=A&client_source=web&f_node=1&s_node=0"
    dc_notice_go = dc_notice_good(headers, url, start[0], "业绩")
    d += dc_notice_go.get("lgt")
    w += dc_notice_go.get("new_date")
    c += dc_notice_go.get("stock_code")
    url = "http://np-anotice-stock.eastmoney.com/api/security/ann?cb=&sr=-1&page_size=100&page_index={}&ann_type=A&client_source=web&f_node=6&s_node=0"
    dc_notice_go = dc_notice_good(headers, url, start[1], "回购")
    d += dc_notice_go.get("lgt")
    w += dc_notice_go.get("new_date")
    c += dc_notice_go.get("stock_code")
    url = "http://np-anotice-stock.eastmoney.com/api/security/ann?cb=&sr=-1&page_size=100&page_index={}&ann_type=A&client_source=web&f_node=7&s_node=0"
    dc_notice_go = dc_notice_good(headers, url, start[2], "增持")
    d += dc_notice_go.get("lgt")
    w += dc_notice_go.get("new_date")
    c += dc_notice_go.get("stock_code")
    with open(r"D:\ana\envs\py36\mywagtailone\datatables\static\store.txt", "w") as f:  # 自动关闭
        f.writelines(w)
    p = views.is_not_path("GOOD_NOTICE.blk")
    with open(p, "w") as f:  # 自动关闭
        f.writelines(c)
    # print("oi", dc_notice_go)
    # print("oi", d)
    # print("oi", w)
    # print("oi", c)
    return d


# 东财公告利好 vue stock
def dc_notice_good(headers, url, start, re):
    lgt = []
    new_date = []
    stock_code = []
    n = 0
    f = 0
    # 公告日，类型，标题
    for i in range(1, 7):
        if f == 1:
            print("break")
            break
        sleep(1.93)
        vv = requests.get(url.format(i), headers=headers)
        # print(vv.status_code)
        text = vv.text
        if vv.status_code == 200 and text:
            detail = json.loads(text).get("data", "")
            # print(detail)
            # detail = ""
            if detail:
                li = detail.get("list", "")
                # print(len(li))
                # li = ""
                if li:
                    n += len(li)
                    if i == 1:
                        if li[0].get("notice_date", ""):
                            new_date.append(li[0].get("notice_date", "") + li[0].get("title", "") + "\n")
                            # new_date.append(li[0].get("title", ""))
                        else:
                            new_date.append(li[1].get("notice_date", "") + li[1].get("title", "") + "\n")
                            # new_date.append(li[1].get("title", ""))

                    for v in li:
                        # print(v)
                        # v = ""
                        if v:
                            notice_date = v.get("notice_date", "")
                            # print(notice_date)
                            if notice_date:
                                da = datetime.strptime(notice_date, "%Y-%m-%d %H:%M:%S")
                                # da = datetime.strptime(notice_date, "%Y-%m-%d %H:%M:%S:%f")
                                notice_date = notice_date[0:10]
                                if (da - start).days < 0:
                                    f = 1
                                    break
                            dis_time = v.get("display_time", "")
                            # print(dis_time)
                            col = v.get("columns", "")
                            col_type = ""
                            if col:
                                col_type = col[0].get("column_name", "")
                            # print(col_type)
                            # title = v.get("title", "")
                            # print(title)
                            if col_type.find(re) != -1:
                                t = v.get("title", "")
                                c = v.get("codes", "")
                                for cc in c:
                                    if cc.get("ann_type").find("A,") != -1:
                                        st = cc.get("stock_code", "")
                                if re == "回购":
                                    if t.find("注销部分") == -1:
                                        lg = {"notice_date": notice_date, "dis_time": dis_time, "col_type": col_type, "title": t}
                                        lgt.append(lg)
                                        stock_code.append(views.code_add(st + "\n"))
                                        # if len(c) == 1:
                                        #     stock_code.append(views.code_add(c[0].get("stock_code", "") + "\n"))
                                        # elif len(c) == 2:
                                        #     stock_code.append(views.code_add(c[1].get("stock_code", "") + "\n"))
                                        # else:
                                        #     print("有误", c)
                                else:
                                    lg = {"notice_date": notice_date, "dis_time": dis_time, "col_type": col_type, "title": t}
                                    lgt.append(lg)
                                    stock_code.append(views.code_add(st + "\n"))
                                    # if len(c) == 1:
                                    #     stock_code.append(views.code_add(c[0].get("stock_code", "") + "\n"))
                                    # elif len(c) == 2:
                                    #     stock_code.append(views.code_add(c[1].get("stock_code", "") + "\n"))
                                    # else:
                                    #     print("有误", c)
    print(n, "条公告")
    return {"lgt": lgt, "new_date": new_date, "stock_code": stock_code}






