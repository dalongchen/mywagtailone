from pprint import pprint
from time import sleep
import time
import requests
import json
from datetime import date as d_date
from datetime import datetime, timedelta
from .. import views
from ..tool import tools


# 东财公告利好 vue stock    此文件是vue为前端的功能
def good_notice_parent(headers, start):
    d = []  # 页面显示集合
    w = []  # 存储最新日期资料
    c = []  # 写入通达信
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
    ths_lis = []
    if re == "业绩":
        views.open_chrome()
        li = []
        for i in range(1, 30):
            li.append(tools.time_increase(str(start)[0:10], i))
        # print(li)
        t = datetime.strptime(time.strftime("%Y-%m-%d"), "%Y-%m-%d")  # 今天
        for ii in li:
            ii = datetime.strptime(ii, "%Y-%m-%d")
            if (t - ii).days < -1:
                print("break date", ii)
                break
            ths_lis += views.ths_choice("预告日期为{}日；公告业绩预减或预亏".format(str(ii)[0:10]), t="2")
            sleep(3)
        print("ths_lis22", len(ths_lis))
    # 公告日，类型，标题
    for i in range(1, 8):
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
                        else:
                            new_date.append(li[1].get("notice_date", "") + li[1].get("title", "") + "\n")
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
                                lg = {"notice_date": notice_date, "dis_time": dis_time, "col_type": col_type, "title": t}
                                if re == "回购":
                                    if t.find("注销部分") == -1:
                                        lgt.append(lg)
                                        stock_code.append(views.code_add(st + "\n"))
                                elif re == "业绩":
                                    if st not in ths_lis:
                                        lgt.append(lg)
                                        stock_code.append(views.code_add(st + "\n"))
                                else:
                                    lgt.append(lg)
                                    stock_code.append(views.code_add(st + "\n"))
    print(n, "条公告")
    return {"lgt": lgt, "new_date": new_date, "stock_code": stock_code}






