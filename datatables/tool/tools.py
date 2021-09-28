from baiduspider import BaiduSpider
from pprint import pprint
from time import sleep
import time
import datetime
import re
import pandas as pd
from .. import views
import os
spider = BaiduSpider()


# 百度个股负面
def bai_du(kw):
    bai = spider.search_web(query=kw, pn=1, exclude=['tieba', 'video'])
    # print(type(bai))
    b = bai.get("results", "")
    to = bai.get("total", "")  # 页数
    # print(to)
    # pprint(b)
    if b:
        if to == (0 or ""):
            return
        elif to == 1:
            return b[2:]
        elif to == 2:
            t = [2]
        else:
            t = [2, 3]
        bai_d = b[2:]
        for i in t:
            sleep(1)
            # pprint(i)
            sp = spider.search_web(query=kw, pn=i, exclude=['tieba', 'video'])
            sp = sp.get("results", "")
            if sp:
                bai_d.extend(sp[2:])
        b = [["时间", "来源", "标题", "描述", "url"]]
        for ii in bai_d:
            if ii.get("origin", "") != "股吧" and ii.get("type", "") != "baike":
                l = list(ii.values())
                if l and len(l) >= 3:
                    if l[2]:  # 取中文
                        l[2] = ''.join(re.findall(re.compile(u'[\u4e00-\u9fa5-\，\。]'), l[2])).replace("-", "")
                    else:
                        l[2] = ""
                    b.append([l[4], l[2], l[0], l[1], l[3]])
        # print(b)
        return b
    return


# 日期自增
def time_increase(begin_time, days):
    ts = time.strptime(str(begin_time), "%Y-%m-%d")
    ts = time.mktime(ts)
    dateArray = datetime.datetime.utcfromtimestamp(ts)
    date_increase = (dateArray + datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    # print("日期：{}".format(date_increase))
    return date_increase


# 输入代码和时间段获取股票k线数据. adjustflag(1：后复权， 2：前复权，3：不复权）
def history_k_data(code="sh.000001", start_date='2021-07-01', end_date='2021-07-31', frequency="d", adjustflag="1"):
    import baostock as bs
    lg = bs.login()
    if not code.startswith("sh.") or not code.startswith("sz."):
        if code.startswith("SH.") or not code.startswith("SZ."):
            code = code.lower()
        else:
            code = views.add_sh(code, big="baostock")
    # 显示登陆返回信息
    print('login respond error_code:' + lg.error_code)
    print('login respond  error_msg:' + lg.error_msg)
    # “分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    rs = bs.query_history_k_data_plus(code,
                                      "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                                      start_date, end_date, frequency, adjustflag)
    print('query_history_k_data_plus respond error_code:' + rs.error_code)
    print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    # adjustflag	指数没有复权?	不复权、前复权、后复权
    # turn	换手率	精度：小数点后6位；单位：%
    # tradestatus	交易状态	1：正常交易 0：停牌
    # pctChg	涨跌幅（百分比）	精度：小数点后6位
    # peTTM	滚动市盈率	精度：小数点后6位
    # psTTM	滚动市销率	精度：小数点后6位
    # pcfNcfTTM	滚动市现率	精度：小数点后6位
    # pbMRQ	市净率	精度：小数点后6位
    # isST	是否ST	1是，0否
    result.to_csv("D:\\history_A_stock_k_data.csv", index=False)
    # print(result)
    bs.logout()


# 压缩
def create_zip(start_dir, tagger, f="local"):
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
        fpath = fpath and fpath + os.sep or ''  # 这句话理解我也点郁闷，实现当前文件夹以及包含的所有文件的压缩
        for filename in filenames:
            z.write(os.path.join(dirpath, filename), fpath+filename)
    print('压缩成功')
    z.close()


# 判断返回那一天
def get_date():
    from chinese_calendar import is_workday
    now = datetime.datetime.now()
    my_date = now.date()
    iss = is_workday(my_date)
    if not iss:
        my_da = (now + datetime.timedelta(-1)).date()
        iss = is_workday(my_da)
        i = 2
        while not iss:
            my_da = (now + datetime.timedelta(-i)).date()
            iss = is_workday(my_da)
            i += 1
        return my_da
    elif iss and (15 >= now.hour >= 0):  # 交易日且15点前取昨天
        my_da = (now + datetime.timedelta(-1)).date()
        iss = is_workday(my_da)
        # print(not iss)
        i = 2
        while not iss:
            my_da = (now + datetime.timedelta(-i)).date()
            iss = is_workday(my_da)
            i += 1
        return my_da
    else:
        return my_date


# 如果当天为交易日则返回否则返回两周内最近一个交易日
def get_late_trade_day(d_now):
    from chinese_calendar import is_workday
    timedelta = datetime.timedelta
    week_day = d_now.date().strftime("%A")
    if week_day == "Sunday":  # 后退2天为周五，3= 4, 4=3, 5=2, 6=1,7,8天为周六日跳过,循环13，第13为周五
        for i in range(2, 14):
            if i != 7 and i != 8:
                if is_workday(d_now + timedelta(-i)):
                    trade_day = (d_now + timedelta(-i)).date()
                    break
    elif week_day == "Monday":  # 后退1天为周7，2= 6, 3=5, 4=4, 5=3, 6=2, 7=1, 8=7, 1,2,8,9天为周六日跳过,循环11，第11为周五
            for i in range(3, 15):
                if i != 8 and i != 9:
                    if is_workday(d_now + timedelta(-i)):
                        trade_day = (d_now + timedelta(-i)).date()
                        print(trade_day)
                        break
    # 后退1天为周1，2= 周7, 3=周6, 4= 周5, 5=周4, 6= 周3, 7=周2, 8= 周1, 9=周7, 2,3,9,10天为周六日跳过,循环12，第12为周五
    elif week_day == "Tuesday":
            for i in range(1, 16):
                if i != 2 and i != 3 and i != 9 and i != 10:
                    if is_workday(d_now + timedelta(-i)):
                        trade_day = (d_now + timedelta(-i)).date()
                        break
    # 后退1天为周2，2= 周1, 3=周7, 4= 周6, 5=周5, 6= 周4, 7=周3, 8= 周2, 9=周1, 3, 4, 10,11天为周六日跳过
    elif week_day == "Wednesday":
            for i in range(1, 17):
                if i != 3 and i != 4 and i != 10 and i != 11:
                    if is_workday(d_now + timedelta(-i)):
                        trade_day = (d_now + timedelta(-i)).date()
                        break
    # 后退1天为周3，2= 周2, 3=周1, 4= 周7, 5=周6, 6= 周5, 7=周4, 8= 周3, 9=周2, 10=周1, 4, 5, 11,12天为周六日跳过
    elif week_day == "Thursday":
            for i in range(1, 18):
                if i != 4 and i != 5 and i != 11 and i != 12:
                    if is_workday(d_now + timedelta(-i)):
                        trade_day = (d_now + timedelta(-i)).date()
                        break
    # 后退1天为周4，2= 周3, 3=周2, 4= 周1, 5=周7, 6= 周6, 7=周5, 8= 周4, 9=周3, 10=周2, 11=周1, 5, 6, 12,13天为周六日跳过
    elif week_day == "Friday":
            for i in range(1, 19):
                if i != 5 and i != 6 and i != 12 and i != 13:
                    if is_workday(d_now + timedelta(-i)):
                        trade_day = (d_now + timedelta(-i)).date()
                        break
    elif week_day == "Saturday":  # 后退一天为周五，2= 4, 3=3, 4=2, 5=1,6,7天为周六日跳过,循环12，第12为周五
        for i in range(1, 13):
            if i != 6 and i != 7:
                if is_workday(d_now + timedelta(-i)):
                    trade_day = (d_now + timedelta(-i)).date()
                    break
    return trade_day


# 传入路径，读取csv数据,
def read_data():
    from ..tool import mysetting
    # ss = os.path.isfile(mysetting.TRADE_CACHE_PATH)
    # d_trade = ['600163', '002774', '002797', '600011', '002202', '002326', '000683', '300891', '601800']
    # pd_csv = pd.read_csv(mysetting.TRADE_CACHE_PATH, header=0).to_dict()
    pd_csv = pd.read_csv(mysetting.TRADE_CACHE_PATH, dtype={'xd_2102': object}).to_dict(orient='records')
    # pd_csv = pd.read_csv(mysetting.TRADE_CACHE_PATH, dtype={'xd_2102': object})
    # pprint(ss)
    # df_li = list(pd_csv.xd_2102.values)
    # for i in df_li:
    #     print(i)
    # if d_trade == df_li:
    #     print("dd")
    # else:
    #     pass
    # df['da'] = pd.to_datetime(df.xd_2102, format='%Y-%m-%d')
    return ""




