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


# 日期自增days=1为当天
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
    from chinese_calendar import is_workday, is_holiday
    now = datetime.datetime.now()
    my_date = now.date()
    # print(my_date)
    iss = is_workday(my_date)
    iss2 = is_holiday(my_date)
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


# 如果当天为交易日则返回否则返回两周内最近一个交易日add_subtract="subtract"后退. d_now = "%Y-%m-%d"),
def get_late_trade_day(d_now, add_subtract="subtract", f="d"):
    if not isinstance(d_now, datetime.datetime):
        d_now = datetime.datetime.strptime(d_now, "%Y-%m-%d")
    week_day = d_now.date().strftime("%A")  # % A 周几
    if week_day == "Sunday":
        if add_subtract == "subtract":
            trade_day = get_late_trade_day_son(d_now, fl="sub")
        elif add_subtract == "add":
            trade_day = get_late_trade_day_son(d_now)
        else:
            print("error;", add_subtract)
    elif week_day == "Monday":
        if add_subtract == "subtract":
            trade_day = get_late_trade_day_son(d_now, fl="sub")
        elif add_subtract == "add":
            trade_day = get_late_trade_day_son(d_now)
        else:
            print("error;", add_subtract)
    elif week_day == "Tuesday":
        if add_subtract == "subtract":
            trade_day = get_late_trade_day_son(d_now, fl="sub")
        elif add_subtract == "add":
            trade_day = get_late_trade_day_son(d_now)
        else:
            print("error;", add_subtract)
    elif week_day == "Wednesday":
        if add_subtract == "subtract":
            trade_day = get_late_trade_day_son(d_now, fl="sub")
        elif add_subtract == "add":
            trade_day = get_late_trade_day_son(d_now)
        else:
            print("error;", add_subtract)
    elif week_day == "Thursday":
        if add_subtract == "subtract":
            trade_day = get_late_trade_day_son(d_now, fl="sub")
        elif add_subtract == "add":
            trade_day = get_late_trade_day_son(d_now)
        else:
            print("error;", add_subtract)
    elif week_day == "Friday":
        if add_subtract == "subtract":
            trade_day = get_late_trade_day_son(d_now, fl="sub")
        elif add_subtract == "add":
            trade_day = get_late_trade_day_son(d_now)
        else:
            print("error;", add_subtract)
    elif week_day == "Saturday":  # 后退一天为周五，2= 4, 3=3, 4=2, 5=1,6,7天为周六日跳过,循环12，第12为周五
        if add_subtract == "subtract":
            trade_day = get_late_trade_day_son(d_now, fl="sub")
        elif add_subtract == "add":
            trade_day = get_late_trade_day_son(d_now)
        else:
            print("error;", add_subtract)
    # f = "d"返回2021 - 07 - 01否则为2021 - 07 - 01 00：00：00
    if f == "d":  # trade_day.date().strftime('%Y-%m-%d')
        return trade_day.date()
    else:
        return trade_day


def get_late_trade_day_son(d_now, fl="add"):
    from chinese_calendar import is_workday
    timedelta = datetime.timedelta
    if fl == "add":
        for i in range(1, 19):
            tt = d_now + timedelta(+i)
            if is_workday(tt):
                ww = tt.strftime("%A")  # % A 周几
                if ww != "Sunday" and ww != "Saturday":
                    trade_day = tt
                    break
    if fl == "sub":
        dd = d_now.strftime("%A")  # % A 周几
        if is_workday(d_now) and dd != "Sunday" and dd != "Saturday":
                trade_day = d_now
        else:
            for i in range(1, 19):
                tt = d_now + timedelta(-i)
                if is_workday(tt):
                    ww = tt.strftime("%A")  # % A 周几
                    if ww != "Sunday" and ww != "Saturday":
                        trade_day = tt
                        break
    return trade_day


# 返回n天前或n天后最近一个交易日add_subtract="subtract"后退. d_now = "%Y-%m-%d"),取当天num=0，最近一天num=1
# def get_late_trade_day_n(d_now, add_subtract="subtract", start=2, end=3, f="d"):
def get_late_trade_day_n(d_now, add_subtract="subtract", num=1, f="d"):
    from chinese_calendar import is_workday
    timedelta = datetime.timedelta
    if not isinstance(d_now, datetime.datetime):
        d_now = datetime.datetime.strptime(d_now, "%Y-%m-%d")
    if add_subtract == "add":
        aa = 0
        for i in range(0, 100):
            tt = d_now + timedelta(+i)
            if is_workday(tt):
                ww = tt.strftime("%A")  # % A 周几
                if ww != "Sunday" and ww != "Saturday":
                    if aa == num:
                        trade_day = tt
                        break
                    aa += 1
    if add_subtract == "sub":
        aa = 0
        for i in range(0, 100):
            tt = d_now + timedelta(-i)
            if is_workday(tt):
                ww = tt.strftime("%A")  # % A 周几
                if ww != "Sunday" and ww != "Saturday":
                    if aa == num:
                        trade_day = tt
                        break
                    aa += 1
    # f = "d"返回2021 - 07 - 01否则为2021 - 07 - 01 00：00：00
    if f == "d":  # trade_day.date().strftime('%Y-%m-%d')
        return trade_day.date()
    else:
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


# big="baostock"加(sh. or sz.)code加(sh or sz) or (SZ or SH)
def add_sh(code, big=""):
    if big == "":
        if code.startswith("0") or code.startswith("3") or code.startswith("2"):
            code = "sz" + code
        elif code.startswith("5") or code.startswith("6") or code.startswith("9"):
            code = "sh" + code
        else:
            print(code)
    elif big == "baostock":
        if code.startswith("0") or code.startswith("3") or code.startswith("2"):
            code = "sz." + code
        elif code.startswith("5") or code.startswith("6") or code.startswith("9"):
            code = "sh." + code
        else:
            print(code)
    else:
        if code.startswith("0") or code.startswith("3") or code.startswith("2"):
            code = "SZ" + code
        elif code.startswith("5") or code.startswith("6") or code.startswith("9"):
            code = "SH" + code
        else:
            print(code)
    return code





