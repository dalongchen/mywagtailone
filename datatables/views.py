import os
import sys
# sys.path.insert(0, os.path.dirname(os.getcwd()))
# print("rte", os.path.dirname(os.getcwd()))
import re
import time
from time import sleep
import requests
import sqlite3
from django.shortcuts import render
from django.utils.safestring import mark_safe
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
# from .models import ShgtDf2021
# from .models import MResearchReport as mrr
from django.http import HttpResponse, JsonResponse
from django.db import connection
import json
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import win32crypt
import win32api
import win32com
import win32gui
import win32con
import psutil
import pprint
import baostock as bs
import pandas as pd
import easytrader
import pywinauto
from pywinauto.controls.hwndwrapper import HwndWrapper
from selenium.webdriver.support import expected_conditions as EC
import string
# import datetime
from datetime import date as d_date
from datetime import datetime, timedelta
from chinese_calendar import is_workday, is_holiday
import chinese_calendar as calendar
import shutil
import dateutil
from dateutil import parser
from dateutil.relativedelta import relativedelta
from django.db import transaction
import demjson
from .tool import tools
from .tool import mysetting
from .view import view_vue_stock as vi_vu
from pprint import pprint
from ratelimit.decorators import ratelimit
import numpy as np
import pandas as pd
import unicodedata
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from treeinterpreter import treeinterpreter as ti
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def index(request):
    # latest_question_list = shgtDf2021()
    # context = {
    #     "latest_question_list": latest_question_list,
    # }
    # context = HttpResponse(context, content_type='application/json; charset=utf-8')
    # print(context)
    return render(request, 'datatables/bootstrap.html')
    # return render(request, 'datatables/index.html', context)


# 各种选股
@ratelimit(key='ip', rate='1/10s', block=True)
def east_money_lgt(request):
    re_get = request.GET
    date = re_get.get("date", "")
    s = re_get.get("s", "")
    # print(s)
    d_t = re_get.get("d_t", "")
    if d_t:
        gmt_format = '%a %b %d %Y %H:%M:%S GMT+0800 (中国标准时间)'
        d_t = str(datetime.strptime(d_t, gmt_format))[0:10]
    # 丢弃，和下面合并
    if re_get.get("ths_fund_inflow", "") == "ths_fund_inflow":
        open_chrome()
        # cookie = get_cookie()
        # date = "2021-04-30"
        search = '资金净买入大于2500万; 涨幅大于1.6%'
        # return JsonResponse({'number': ""})
        return JsonResponse({'number': ths_fund_inflow(date, search, '400', "ths_fund_inflow")})

    # 同花顺资金净买入大于10万; 涨幅大于1% 以及  大于2500和涨幅1.6%
    if s == "ths_fund_inflow0":
        if d_t:
            open_chrome()
            s = ths_fund_inflow(date, '资金净买入大于10万; 涨幅大于1%', '2000', "ths_fund_inflow0")
            # s = ""
            sleep(2)
            if s:
                ss = ths_fund_inflow(date, '资金净买入大于2500万; 涨幅大于1.6%', '400', "ths_fund_inflow")
            return JsonResponse({'number': "大10:" + s + "大2500:" + ss})
        return JsonResponse({'number': "缺日期"})

    # 读东财龙虎榜
    if s == "east_dragon":
        if d_t:
            return JsonResponse({'number': east_dragon_tiger(d_t)})
        return JsonResponse({'number': '缺日期'})

    # 读东财陆股通
    if s == "east_lgt":
        if d_t:
            return JsonResponse({'number': east_lgt_finance(d_t, '400', "east_lgt")})
        return JsonResponse({'number': '缺日期'})

    # 读东财上海融资
    if s == "east_finance_sh":
        if d_t:
            return JsonResponse({'number': east_lgt_finance(d_t, '300', "east_finance_sh")})
        return JsonResponse({'number': "缺日期"})

    # 读东财深圳融资
    if s == "east_finance_sz":
        if d_t:
            return JsonResponse({'number': east_lgt_finance(d_t, '200', "east_finance_sz")})
        return JsonResponse({'number': "缺日期"})

    # 读东财融资股票数量
    if s == "east_finance_number":
        if d_t:
            return JsonResponse({'number': east_lgt_finance(d_t, '500', "east_finance_number")})
        return JsonResponse({'number': "缺日期"})

    # 读东财陆股通股票数量
    if s == "east_lgt_number":
        if d_t:
            return JsonResponse({'number': east_lgt_finance(d_t, '1800', "east_lgt_number")})
        return JsonResponse({'number': "缺日期"})

    # 读东财研究报告和机构调研股票数量
    if s == "research_report":
        s = research_report(re_get.get("start_date", ""), re_get.get("end_date", ""), '100', "add")
        ss = research_organization(re_get.get("start_date", ""), re_get.get("end_date", ""), '10000')
        return JsonResponse({'number': s + ss})

    # 读东财机构调研股票数量 被上面研报代替。
    if re_get.get("research_organization", "") == "research_organization":
        return JsonResponse({'number': research_organization(re_get.get("start_date", ""), re_get.get("end_date", ""), '10000', "research_organization")})

    # 交集和并集股票数量
    if s == "combine":
        return JsonResponse({'number': combine()})

    # 加雪球和自选 新版不需要显示choice板块个股
    if s == "shown_choice":
        write_self_hai_tong()  # 读choice写自选和海通自选
        return JsonResponse({'number': "choice板块" + str(len(read_choice("choice.blk")))})

    # 废弃，已经和上面合并。读choice写自选和海通自选
    if re_get.get("read_self_choice", "") == "read_self_choice":
        # print("read_self_choice")
        write_self_hai_tong()  # 读choice写自选和海通自选
        return JsonResponse({'is_success': "成功"})

    # 读dragon板块龙虎榜页面
    if s == "open_dragon":
        return JsonResponse({'number': read_dragon("DRAGON_TIGER.blk")})

    # 预埋单
    if s == "pre_paid":
        # 读取choice板块买入
        stock_list = read_choice_code("choice.blk")
        if len(stock_list):
            stock_dict = inquiry_close(stock_list, date, f="2")  # f="2" 获取收盘价和名字
        stock_sell = read_choice_code("choice_sell.blk")
        if len(stock_sell):
            stock_dict += inquiry_close(stock_sell, date, buy="卖出", f="2")  # f="2" 获取收盘价和名字
        # print(stock_dict)
        # stock_dict = ""
        if len(stock_dict):
            pre_paid(stock_dict, t="backstage")  # t="backstage"时从交易软件后台数据库添加
            log_on_ht()  # 登录海通
            return JsonResponse({'number': u"成功"+str(len(stock_dict))})
        else:
            return JsonResponse({'number': u"失败"})

    #  同花顺陆股通  # 需要改cookie
    if s == "ths_lgt":
        if d_t:
            return JsonResponse({'number': ths_lgt(d_t)})
        return JsonResponse({'number': "缺日期 "})

    # 同花顺公告利好
    if re_get.get("ths_notice", "") == "ths_notice":
        open_chrome()
        number = ths_notice_good(date)
        # number = ""
        # print(number)
        return JsonResponse({'number': number})

    # 东财公告利好
    if re_get.get("dc_notice", "") == "dc_notice":
        number = ths_notice_good(date)
        # number = ""
        # print(number)
        return JsonResponse({'number': number})

    # 同花顺涨停或大于5
    if re_get.get("ths_rise", "") == "ths_rise":
        open_chrome()
        up_rise = re_get.get("up_rise", "")
        # print(up_rise)
        # date = "2021-04-28"
        number = ths_rise(date, up_rise)
        # number = ""
        # print(number)
        return JsonResponse({'number': number})

    # 同花顺选股
    if re_get.get("ths_choice", "") == "ths_choice":
        open_chrome()
        t = re_get.get("t", "")
        if t:
            if not d_t:
                return JsonResponse({'number': "缺日期"})
            if t == "ths_lgt02":
                s = "{}陆股通净买入大于1000万；{}陆股通净买入占流通股市值的比例大于0.2%".format(d_t, d_t)
            return JsonResponse({'number': len(ths_choice(s, f="lgtda02.blk"))})
        return JsonResponse({'number': len(ths_choice(re_get.get("ths_in", "")))})

    # 打开个股详情雪球,东财
    if re_get.get("open_stock_detail", "") == "open_stock_detail":
        code_name = re_get.get("code_name", "")
        # print(code_name)
        # code = re.sub("\D", "", code_name)
        # if code.startswith("6"):
        #     xue_code = "SH" + code
        #     # print(code)
        # else:
        #     xue_code = "SZ" + code
        #     # print(code)
        name = code_name.strip(string.digits)
        # print(name)
        dri = selenium_open()
        # xue_code = "SH600026"
        # code = "600026"
        # name = "共达电声"
        # dri.get("https://xueqiu.com/S/" + xue_code)
        # open_xue_qiu_able(dri, xue_code, 2)
        # open_xue_qiu_able(dri, xue_code, 4)
        # open_xue_qiu_able(dri, xue_code, 5)
        # open_xue_qiu_able(dri, xue_code, 6)
        sleep(1)
        dri.execute_script('window.open("https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=1&rsv_idx=1&tn=baidu&wd={}负面&fenlei=256&rsv_pq=acca74440006348f&rsv_t=a491ttWdgaJ1ft%2FlyX1Xfp6kRoKFHqCpFwCHag6p8%2FZaID1SxzUCu6YSQKI&rqlang=cn&rsv_enter=1&rsv_dl=tb&rsv_sug3=6&rsv_sug1=5&rsv_sug7=100&rsv_sug2=0&rsv_btype=i&inputT=14546&rsv_sug4=22545")'.format(name))
        # sleep(3)  # 个股详情
        # dri.execute_script('window.open("http://data.eastmoney.com/stockdata/{}.html")'.format(code))
        # sleep(4)  # 资金流入
        # dri.execute_script('window.open("http://data.eastmoney.com/zjlx/{}.html")'.format(code))
        # sleep(4)  # 融资融券
        # dri.execute_script('window.open("http://data.eastmoney.com/rzrq/detail/{}.html")'.format(code))
        # sleep(4)  # 大宗交易
        # dri.execute_script('window.open("http://data.eastmoney.com/dzjy/detail/{}.html")'.format(code))
        # sleep(4)  # 高管减持
        # dri.execute_script('window.open("http://data.eastmoney.com/executive/{}.html")'.format(code))
        # sleep(4)  # 股东减持
        # dri.execute_script('window.open("http://data.eastmoney.com/executive/gdzjc/{}.html")'.format(code))
        # sleep(4)  # 限售
        # dri.execute_script('window.open("http://data.eastmoney.com/dxf/q/{}.html")'.format(code))
        # sleep(4)  # 股东户数
        # dri.execute_script('window.open("http://data.eastmoney.com/gdhs/detail/{}.html")'.format(code))
        # sleep(4)  # 互动
        # dri.execute_script('window.open("http://guba.eastmoney.com/qa/qa_search.aspx?company={}")'.format(code))
        # sleep(4) # lgt
        # dri.execute_script('window.open("http://data.eastmoney.com/hsgtcg/StockHdStatistics/{}.html")'.format(code))

    # 打开360
    if re_get.get("browser", "no") == "360":
        path_folder = "D:/mySoft/360Chrome/Chrome/Application/"
        path_exe = "360chrome.exe"
        run_app(path_folder, path_exe, "360", "")
    # return HttpResponse()
    return render(request, 'datatables/east_money_lgt.html')
xq_dis = 0


# 单个股票详情 vue为前端
@ratelimit(key='ip', rate='1/10s', block=True)
def stock_details(request):
    global xq_dis  # 雪球讨论取不到是把其置为1，再取一次
    re_get = request.GET
    st = re_get.get("st", "")
    # tail页面没有股票代码时去获取choice板块代码
    if st == "":
        stock_li = read_choice("choice.blk", xue_qiu="")
        # print(stock_li)
        return HttpResponse(json.dumps(stock_li))
    # print("df")
    # return HttpResponse(json.dumps("oo"))
    number = re.sub("\D", "", st)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36',
    }
    sina = [sina_real_time(number)]  # 新浪实时
    # sina = ""
    finance = stock_finance(number, headers)  # 财务指标
    # finance = ""
    sleep(1.1)
    st_ach = stock_achievement(number, headers)  # 业绩
    # st_ach = ""
    sleep(1)
    per_for = performance_forecast(number, headers)  # 业绩预告
    # per_for = ""
    sleep(1.5)
    ins_pos = institution_position(number, headers)  # 机构持仓 股东 解禁
    # ins_pos = ""
    sleep(1.05)
    lgt = east_lgt_detail(number, headers)  # 陆股通
    # lgt = ""
    sleep(1.2)
    # lift_ban = east_lift_ban(number)  # 东财解禁详细 暂时丢弃
    # sleep(0.5)
    add_subtract = east_add_subtract(number, headers)  # 股东减持
    # add_subtract = ""
    sleep(1.1)
    manager_a = manager_add(number, headers)  # 高管减持
    # manager_a = ""
    sleep(1.8)
    rz = east_rz(number, headers)  # 融资
    # rz = ""
    sleep(1.)
    capital_inflow = east_zllr(number, headers)  # 资金流入
    # capital_inflow = ""
    sleep(1.6)
    institution_res = institution_research(number, headers)  # 机构调研
    # institution_res = ""
    sleep(1.3)
    share_num = shareholder_number(number, headers)  # 股东户数
    # share_num = ""
    sleep(1.)
    dragon_tiger = per_dragon_tiger(number, headers)  # 龙虎榜
    # dragon_tiger = ""
    sleep(1.19)
    ins_re_re = ins_research_report(number, headers)  # 机构研究报告
    # ins_re_re = ""
    xq_discuss = xiu_qiu_discuss(number)  # 雪球讨论
    # xq_discuss = ""
    sleep(1.1)
    # print("jdjd", xq_dis == 1)
    if xq_dis == 1:  # 为1时打开浏览器，读cookie，再来一次
        xq_discuss = xiu_qiu_discuss(number)  # 雪球讨论
        # xq_discuss = ""
    xq_new = xiu_qiu_new(number)  # 雪球资信
    # xq_new = ""
    bai = tools.bai_du(re.sub("\d", "", st) + '负面')
    # bai = ""
    st_notice = stock_notice(number, headers)  # 个股公告
    # st_notice = ""
    d = {
        "sina": sina,
        "finance": finance,
        "st_ach": st_ach,
        "per_for": per_for,
        "ins_pos": ins_pos,
        "lgt": lgt,
        # #"lift_ban": lift_ban,
        "add_subtract": add_subtract,
        "manager_a": manager_a,
        "rz": rz,
        "capital_inflow": capital_inflow,
        "institution_res": institution_res,
        "share_num": share_num,
        "dragon_tiger": dragon_tiger,
        "ins_re_re": ins_re_re,
        "xq_discuss": xq_discuss,
        "xq_new": xq_new,
        "bai": bai,
        "st_notice": st_notice,
    }
    return HttpResponse(json.dumps(d))


# 东财选股数据 新 vue stock
@ratelimit(key='ip', rate='1/10s', block=True)
def east_data(request):
    re_get = request.GET
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36',
    }
    # 东财公告利好 vue stock
    if re_get.get("dc_notice", "") == "dc_notice":
        d_t = re_get.get("d_t", "")
        if d_t:
            gmt_format = '%a %b %d %Y %H:%M:%S GMT+0800 (中国标准时间)'
            start = datetime.strptime(d_t, gmt_format)
            # print(start)
            d = vi_vu.good_notice_parent(headers, [start, start, start])
        else:
            r = read_file(r"D:\ana\envs\py36\mywagtailone\datatables\static\store.txt", f="1")
            # print(r)
            # r = ""
            if r:
                start1 = datetime.strptime(r[0][0:19], "%Y-%m-%d %H:%M:%S")
                start2 = datetime.strptime(r[1][0:19], "%Y-%m-%d %H:%M:%S")
                start3 = datetime.strptime(r[2][0:19], "%Y-%m-%d %H:%M:%S")
                d = vi_vu.good_notice_parent(headers, [start1, start2, start3])
            else:
                d = []
                print("store文件有误")
        return JsonResponse({'dc_notice_go': d})

    return JsonResponse("")


# 人工智能
@ratelimit(key='ip', rate='1/10s', block=True)
def artificial_intelligence(request):
    re_get = request.GET
    df_stocks = pd.read_pickle('data/pickled_ten_year_filtered_data.pkl')
    df_stocks['prices'] = df_stocks['adj close'].apply(np.int64)
    df_stocks = df_stocks[['prices', 'articles']]
    df_stocks['articles'] = df_stocks['articles'].map(lambda x: x.lstrip('.-'))
    print("yt")
    return JsonResponse({"number": "ok"})


# 网易 历史行情
def w163_history(code):
    code = code_add(code, param="param")
    print(code, "poy")
    v = requests.get( "http://quotes.money.163.com/service/chddata.html?code={}&start={}&end={}&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP".format(code, "20210506", "20210506"))
    text = v.text
    # 日期 代码 名称 收盘价 最高价	最低价	开盘价 前收盘 涨跌额 涨跌幅	换手率	成交量 成交金额 总市值	流通市值 成交笔数
    #  TCLOSE收盘价 ;HIGH最高价;LOW最低价;TOPEN开盘价;LCLOSE前收盘价;CHG涨跌额;PCHG涨跌幅;TURNOVER换手率;VOTURNOVER成交量;VATURNOVER成交金额;TCAP总市值;MCAP流通市值
    if v.status_code == 200 and text:
        try:
            t = text.split("\r")[1].split(",")
            # print(t[3], t[9])
            # print(t[10], t[12])
            # print(t[13], t[14])
            # print(t, "piu")
            return {
                "date": t[0].strip(),
                "name": t[2],
                "close": t[3],
                "up": t[9],
                "turnover": t[10],
                "va": t[12],
                "total": t[13],
                "circulate": t[14]
            }
        except:
            print("没有数据")
            return ""
    return ""


# one 新浪即时行情 # 75.730,75.400,79.000,现价 79.480, 75.500, 79.000, 79.010, 3197647,总量 250231897.790,成交金额 2021-05-31,15:00:00,
def sina_real_time(code):  # 日期，时间 名字，现价，成交量，成交金额，
    # add_sh(code)
    vv = requests.get("http://hq.sinajs.cn/list={}".format(add_sh(code)))
    text = vv.text
    if vv.status_code == 200 and text:
        detail = text.split("\"")[1].split(",")
        if detail[8]:
            detail[8] = '{:.2f}亿'.format(float(detail[8]) / 100000000)
        if detail[9]:
            detail[9] = '{:.2f}亿'.format(float(detail[9]) / 100000000)
        return [detail[30], detail[31], detail[0], detail[3], detail[8], detail[9]]
        # return {
        #     "name": detail[0],
        #     "now_price": detail[3],
        #     "now_vo": detail[8],
        #     "now_va": detail[9],
        #     "d_date": detail[-3],
        #     "timely": detail[-2],
        # }
    return ""


# two 财务指标 http://f10.eastmoney.com/f10_v2/FinanceAnalysis.aspx?code=SZ000785  # 这个应该是表格的90度翻转
def stock_finance(code, headers):
    code = add_sh(code, big="big")
    # print(code)
    # return
    url = 'http://f10.eastmoney.com/NewFinanceAnalysis/ZYZBAjaxNew?type=0&code={}'
    vv = requests.get(url.format(code), headers=headers)
    # print(vv.status_code)
    text = vv.text
    # print(text)
    if vv.status_code == 200 and text:
        detail = json.loads(text).get("data", "")
        # print(detail)
        # detail = ""
        if detail:
            d = [
                '季度',
                '公告日',
                '基本每股收益(元)',
                '扣非每股收益(元)',
                '稀释每股收益(元)',
                '每股净资产(元)',
                '每股公积金(元)',
                '每股未分配利润(元)',
                '每股经营现金流(元)',
                '营业总收入',

                '归属净利润',
                '扣非净利润',
                '营业总收入同比增长',
                '归属净利润同比增长',
                '扣非净利润同比增长',
                '营业总收入滚动环比增长',
                '归属净利润滚动环比增长',
                '扣非净利润滚动环比增长',
                '净资产收益率(加权)',
                '净资产收益率(扣非)',

                '总资产收益率(加权)',
                '毛利率',
                '净利率',
                '流动比率',
                '速动比率',
                '现金流量比率',
                '资产负债率',
                '权益乘数',
                '产权比率',
                '总资产周转天数(天)',

                '存货周转天数(天)',
                '应收账款周转天数(天)',
                '总资产周转率(次)',
                '存货周转率(次)',
                '应收账款周转率(次)'
            ]
            lgt = [d]
            # print(lgt)
            # 季度21 - 03 - 31  REPORT_DATE
            # 公告日  NOTICE_DATE
            # 基本每股收益(元)0.0900 # EPSJB: 0.09
            # 扣非每股收益(元) - -    EPSKCJB
            # 稀释每股收益(元) 0.0900 # EPSXS: 0.09
            # 每股净资产(元) 2.5168 # BPS: 2.516781312035
            # 每股公积金(元)1.2719 # MGZBGJ: 1.271914156284
            # 每股未分配利润(元) 0.1920 # MGWFPLR: 0.192008056476
            # 每股经营现金流(元) 0.1820 # MGJYXJJE: 0.182031308981
            # 营业总收入(元) 33.42 亿 # TOTALOPERATEREVE: 3342213825.15

            # 归属净利润(元)6.153 亿 # PARENTNETPROFIT: 615286108.76
            # 扣非净利润(元)6.143 亿 # KCFJCXSYJLR: 614335543.35
            # 营业总收入同比增长( %)    39.05   # TOTALOPERATEREVETZ: 39.0546531976
            # 归属净利润同比增长( %)    144.58  # PARENTNETPROFITTZ: 144.5771006343
            # 扣非净利润同比增长( %)    131.23  # KCFJCXSYJLRTZ: 131.2267552973
            # 营业总收入滚动环比增长( %)    10.44  # YYZSRGDHBZC: 10.438095592011
            # 归属净利润滚动环比增长( %)    26.69  # NETPROFITRPHBZC: 26.690892283152
            # 扣非净利润滚动环比增长( %)    27.48  # KFJLRGDHBZC: 27.482618509451
            # 净资产收益率(加权)( %)    3.41  # ROEJQ: 3.41
            # 净资产收益率(扣非 / 加权)( %)    -- ROEKCJQ   8.05

            # 总资产收益率(加权)( %)    1.34  # ZZCJLL: 1.3379151284
            # 毛利率( %)    46.17  # XSMLL: 46.165063003734
            # 净利率( %)    18.85  # XSJLL: 18.8528443353
            # 流动比率 1.231  # LD: 1.230816445912
            # 速动比率 1.197  # SD: 1.19748128541
            # 现金流量比率 0.140  # XJLLB: 0.139949401809
            # 资产负债率( %)    68.43  # ZCFZL: 68.4349099218
            # 权益乘数 3.168  # QYCS: 3.16805685497
            # 产权比率 2.287  # CQBL: 2.28684843613
            # 总资产周转天数(天)1268  # ZZCZZTS: 1268.20898739016

            # 存货周转天数(天)11.58  # CHZZTS: 11.584537446216
            # 应收账款周转天数(天)26.54  # YSZKZZTS: 26.53880464387
            # 总资产周转率(次) 0.071  # TOAZZL: 0.070966221573
            # 存货周转率(次) 7.769  # CHZZL: 7.768976570523
            # 应收账款周转率(次)3.391  # YSZKZZL: 3.391260503543

            for v in detail:
                # print(v)
                # v = ""
                if v:
                    report = v.get("REPORT_DATE", "")
                    notice = v.get("NOTICE_DATE", "")
                    eps = v.get("EPSJB", "")
                    deduct_eps = v.get("EPSKCJB", "")
                    dilution_eps = v.get("EPSXS", "")
                    p_net_asset = v.get("BPS", "")
                    p_public = v.get("MGZBGJ", "")
                    p_undistributed = v.get("MGWFPLR", "")
                    p_cash = v.get("MGJYXJJE", "")
                    income = v.get("TOTALOPERATEREVE", "")

                    profit = v.get("PARENTNETPROFIT", "")
                    deduct_profit = v.get("KCFJCXSYJLR", "")
                    income_y_o_y = v.get("TOTALOPERATEREVETZ", "")
                    profit_y_o_y = v.get("PARENTNETPROFITTZ", "")
                    deduct_profit_y_o_y = v.get("KCFJCXSYJLRTZ", "")
                    income_q_o_q = v.get("YYZSRGDHBZC", "")
                    profit_q_o_q = v.get("NETPROFITRPHBZC", "")
                    deduct_profit_q_o_q = v.get("KFJLRGDHBZC", "")
                    roe = v.get("ROEJQ", "")
                    deduct_roe = v.get("ROEKCJQ", "")

                    total_roe = v.get("ZZCJLL", "")
                    gross_profit = v.get("XSMLL", "")
                    net_profit = v.get("XSJLL", "")
                    current_rate = v.get("LD", "")
                    quick_rate = v.get("SD", "")
                    cash_rate = v.get("XJLLB", "")
                    debt_rate = v.get("ZCFZL", "")
                    equity_multiplier = v.get("QYCS", "")
                    equity_rate = v.get("CQBL", "")
                    asset_turnover = v.get("ZZCZZTS", "")

                    inventory_turnover = v.get("CHZZTS", "")
                    receivable_turnover = v.get("YSZKZZTS", "")
                    asset_rate = v.get("TOAZZL", "")
                    inventory_rate = v.get("CHZZL", "")
                    receivable_rate = v.get("YSZKZZL", "")
                    # print(base, deduct)
                    if report:
                        report = report[0:10]
                    if report is None:
                        report = "-"
                    if notice:
                        notice = notice[0:10]
                    if notice is None:
                        notice = "-"

                    if eps:
                        eps = '{:.2f}'.format(eps)
                    if eps is None:
                        eps = "-"

                    if deduct_eps:
                        deduct_eps = '{:.2f}'.format(deduct_eps)
                    if deduct_eps is None:
                        deduct_eps = "-"

                    if dilution_eps:
                        dilution_eps = '{:.2f}'.format(dilution_eps)
                    if dilution_eps is None:
                        dilution_eps = "-"

                    if p_net_asset:
                        p_net_asset = '{:.2f}'.format(p_net_asset)
                    if p_net_asset is None:
                        p_net_asset = "-"

                    if p_public:
                        p_public = '{:.2f}'.format(p_public)
                    if p_public is None:
                        p_public = "-"

                    if p_undistributed:
                        p_undistributed = '{:.2f}'.format(p_undistributed)
                    if p_undistributed is None:
                        p_undistributed = "-"

                    if p_cash:
                        p_cash = '{:.2f}'.format(p_cash)
                    if p_cash is None:
                        p_cash = "-"

                    if income:
                        income = '{:.2f}亿'.format(income/100000000)
                    if income is None:
                        income = "-"

                    if profit:
                        profit = '{:.2f}亿'.format(profit/100000000)
                    if profit is None:
                        profit = "-"

                    if deduct_profit:
                        deduct_profit = '{:.2f}亿'.format(deduct_profit/100000000)
                    if deduct_profit is None:
                        deduct_profit = "-"

                    if income_y_o_y:
                        income_y_o_y = '{:.2f}%'.format(income_y_o_y)
                    if income_y_o_y is None:
                        income_y_o_y = "-"

                    if profit_y_o_y:
                        profit_y_o_y = '{:.2f}%'.format(profit_y_o_y)
                    if profit_y_o_y is None:
                        profit_y_o_y = "-"

                    if deduct_profit_y_o_y:
                        deduct_profit_y_o_y = '{:.2f}%'.format(deduct_profit_y_o_y)
                    if deduct_profit_y_o_y is None:
                        deduct_profit_y_o_y = "-"

                    if income_q_o_q:
                        income_q_o_q = '{:.2f}%'.format(income_q_o_q)
                    if income_q_o_q is None:
                        income_q_o_q = "-"

                    if profit_q_o_q:
                        profit_q_o_q = '{:.2f}%'.format(profit_q_o_q)
                    if profit_q_o_q is None:
                        profit_q_o_q = "-"

                    if deduct_profit_q_o_q:
                        deduct_profit_q_o_q = '{:.2f}%'.format(deduct_profit_q_o_q)
                    if deduct_profit_q_o_q is None:
                        deduct_profit_q_o_q = "-"

                    if roe:
                        roe = '{:.2f}%'.format(roe)
                    if roe is None:
                        roe = "-"

                    if deduct_roe:
                        deduct_roe = '{:.2f}%'.format(deduct_roe)
                    if deduct_roe is None:
                        deduct_roe = "-"

                    if total_roe:
                        total_roe = '{:.2f}%'.format(total_roe)
                    if total_roe is None:
                        total_roe = "-"

                    if gross_profit:
                        gross_profit = '{:.2f}%'.format(gross_profit)
                    if gross_profit is None:
                        gross_profit = "-"

                    if net_profit:
                        net_profit = '{:.2f}%'.format(net_profit)
                    if net_profit is None:
                        net_profit = "-"

                    if current_rate:
                        current_rate = '{:.2f}%'.format(current_rate)
                    if current_rate is None:
                        current_rate = "-"

                    if quick_rate:
                        quick_rate = '{:.2f}%'.format(quick_rate)
                    if quick_rate is None:
                        quick_rate = "-"

                    if cash_rate:
                        cash_rate = '{:.2f}%'.format(cash_rate)
                    if cash_rate is None:
                        cash_rate = "-"

                    if debt_rate:
                        debt_rate = '{:.2f}%'.format(debt_rate)
                    if debt_rate is None:
                        debt_rate = "-"

                    if equity_multiplier:
                        equity_multiplier = '{:.2f}'.format(equity_multiplier)
                    if equity_multiplier is None:
                        equity_multiplier = "-"

                    if equity_rate:
                        equity_rate = '{:.2f}'.format(equity_rate)
                    if equity_rate is None:
                        equity_rate = "-"

                    if asset_turnover:
                        asset_turnover = '{:.2f}'.format(asset_turnover)
                    if asset_turnover is None:
                        asset_turnover = "-"

                    if inventory_turnover:
                        inventory_turnover = '{:.2f}'.format(inventory_turnover)
                    if inventory_turnover is None:
                        inventory_turnover = "-"

                    if receivable_turnover:
                        receivable_turnover = '{:.2f}'.format(receivable_turnover)
                    if receivable_turnover is None:
                        receivable_turnover = "-"

                    if asset_rate:
                        asset_rate = '{:.2f}'.format(asset_rate)
                    if asset_rate is None:
                        asset_rate = "-"

                    if inventory_rate:
                        inventory_rate = '{:.2f}'.format(inventory_rate)
                    if inventory_rate is None:
                        inventory_rate = "-"

                    if receivable_rate:
                        receivable_rate = '{:.2f}'.format(receivable_rate)
                    if receivable_rate is None:
                        receivable_rate = "-"
                    # print(income)
                    lg = [
                        report,
                        notice,
                        eps,
                        deduct_eps,
                        dilution_eps,
                        p_net_asset,
                        p_public,
                        p_undistributed,
                        p_cash,
                        income,

                        profit,
                        deduct_profit,
                        income_y_o_y,
                        profit_y_o_y,
                        deduct_profit_y_o_y,
                        income_q_o_q,
                        profit_q_o_q,
                        deduct_profit_q_o_q,
                        roe,
                        deduct_roe,

                        total_roe,
                        gross_profit,
                        net_profit,
                        current_rate,
                        quick_rate,
                        cash_rate,
                        debt_rate,
                        equity_multiplier,
                        equity_rate,
                        asset_turnover,

                        inventory_turnover,
                        receivable_turnover,
                        asset_rate,
                        inventory_rate,
                        receivable_rate
                    ]
                    lgt.append(lg)
            return list(map(list, zip(*lgt)))  # 这个应该是表格的90度翻转
    return ""


# three 业绩
def stock_achievement(code, headers):
    url = 'http://datacenter-web.eastmoney.com/api/data/get?callback=&st=REPORTDATE&sr=-1&ps=9&p=1&sty=ALL&filter=(SECURITY_CODE%3D%22{}%22)&token=894050c76af8597a853f5b408b759f5d&type=RPT_LICO_FN_CPD'
    # print(url.format(code))
    vv = requests.get(url.format(code), headers=headers)
    # print(vv.status_code)
    text = vv.text
    # print(text)
    if vv.status_code == 200 and text:
        detail = json.loads(text).get("result", "")
        # print(detail)
        # detail = ""
        if detail:
            dd = detail.get("data", "")
            if dd:
                d = [
                    '公告日',
                    '截至',
                    '收益元',
                    '扣非元',
                    '营收',
                    '同比%',
                    '环比%',
                    '利润%',
                    '同比%',
                    '环比%',

                    '净资产元',
                    '收益率%',
                    '现金流元',
                    '毛利%',
                    '分配',
                ]
                lgt = [d]
                # 公告日，截至，收益(元)，扣非(元)，营收，同比%，环比%，利润，同比%，环比%，净资产(元)，收益率%，现金流，毛利%，分配
                for v in dd:
                    # print(v)
                    # v = ""
                    if v:
                        notice = v.get("EITIME", "")
                        report = v.get("REPORTDATE", "")
                        base = v.get("BASIC_EPS", "")
                        deduct = v.get("DEDUCT_BASIC_EPS", "")
                        income = v.get("TOTAL_OPERATE_INCOME", "")
                        in_y_one_y = v.get("YSTZ", "")
                        in_q_on_q = v.get("YSHZ", "")
                        profit = v.get("PARENT_NETPROFIT", "")
                        p_y_one_y = v.get("SJLTZ", "")
                        p_q_on_q = v.get("SJLHZ", "")
                        asset = v.get("BPS", "")
                        roe = v.get("WEIGHTAVG_ROE", "")
                        cash_flow = v.get("MGJYXJJE", "")
                        gross_pro = v.get("XSMLL", "")
                        assign = v.get("ASSIGNDSCRPT", "")
                        # print(notice, report)
                        # print(base, deduct)
                        # vv = ""
                        if report:
                            report = report[0:10]
                        # print(report)
                        if income:
                            income = '{:.2f}亿'.format(income/100000000)
                        if in_y_one_y:
                            in_y_one_y = '{:.2f}'.format(in_y_one_y)
                        if in_q_on_q:
                            in_q_on_q = '{:.2f}'.format(in_q_on_q)
                        if profit:
                            profit = '{:.2f}亿'.format(profit/100000000)
                        if p_y_one_y:
                            p_y_one_y = '{:.2f}'.format(p_y_one_y)
                        if p_q_on_q:
                            p_q_on_q = '{:.2f}'.format(p_q_on_q)
                        if asset:
                            asset = '{:.2f}'.format(asset)
                        if cash_flow:
                            cash_flow = '{:.2f}'.format(cash_flow)
                        if gross_pro:
                            gross_pro = '{:.2f}'.format(gross_pro)
                        # print(income)
                        lg = [notice, report, base, deduct, income, in_y_one_y, in_q_on_q, profit,
                              p_y_one_y, p_q_on_q, asset, roe, cash_flow, gross_pro, assign]
                        lgt.append(lg)
                # print(code, lgt)
                return list(map(list, zip(*lgt)))
    print(code, "无业绩")
    return ""


# four 业绩预告
def performance_forecast(code, headers):
    url = 'http://datacenter-web.eastmoney.com/securities/api/data/v1/get?callback=&sortColumns=NOTICE_DATE&sortTypes=-1&pageSize=7&pageNumber=1&sty=ALL&filter=(SECURITY_CODE%3D%22{}%22)&token=894050c76af8597a853f5b408b759f5d&type=RPT_PUBLIC_OP_NEWPREDICT&st=REPORT_DATE&reportName=RPT_PUBLIC_OP_NEWPREDICT&columns=ALL'
    vv = requests.get(url.format(code), headers=headers)
    # print(vv.status_code)
    text = vv.text
    # print(text)
    if vv.status_code == 200 and text:
        detail = json.loads(text).get("result", "")
        # print(detail)
        # detail = ""
        if detail:
            dd = detail.get("data", "")
            if dd:
                lgt = [[
                    '公告日',
                    '截至',
                    '指标',
                    '预告类型',
                    '预测值',
                    '幅度',
                    '业绩',
                    '原因',
                ]]
                # 公告日，截至，指标，预告类型,预测值，幅度，业绩，原因
                for v in dd:
                    # print(v)
                    # v = ""
                    if v:
                        notice = v.get("NOTICE_DATE", "")
                        report = v.get("REPORT_DATE", "")
                        fin_type = v.get("PREDICT_FINANCE", "")
                        ty = v.get("PREDICT_TYPE", "")
                        profit_low = v.get("PREDICT_AMT_LOWER", "")
                        profit_up = v.get("PREDICT_AMT_UPPER", "")
                        mg_low = v.get("ADD_AMP_LOWER", "")
                        mg_up = v.get("ADD_AMP_UPPER", "")
                        content = v.get("PREDICT_CONTENT", "")
                        reason = v.get("CHANGE_REASON_EXPLAIN", "")
                        # print(base, deduct)
                        if notice:
                            notice = notice[0:10]
                            e = (datetime.strptime(notice, '%Y-%m-%d') - datetime.strptime(
                                str(d_date.today() + timedelta(-365)), '%Y-%m-%d'))
                            # print(e.days)
                            if e.days < 0:
                                break
                        if report:
                            report = report[0:10]
                        # print("11q", profit_low)
                        if profit_low:
                            if fin_type == "每股收益":
                                profit_low = '{:.2f}元'.format(profit_low)
                            else:
                                profit_low = '{:.2f}亿'.format(profit_low/100000000)
                        else:
                            profit_low = ""
                        if profit_up:
                            if fin_type == "每股收益":
                                profit_up = '{:.2f}元'.format(profit_up)
                            else:
                                profit_up = '{:.2f}亿'.format(profit_up/100000000)
                        else:
                            profit_up = ""
                        if mg_low:
                            mg_low = '{:.2f}%'.format(mg_low)
                        else:
                            mg_low = ""
                        if mg_up:
                            mg_up = '{:.2f}%'.format(mg_up)
                        else:
                            mg_up = ""
                        lg = [notice, report, fin_type, ty, profit_low + "-" + profit_up,
                              mg_low + "-" + mg_up, content, reason]
                        lgt.append(lg)
                if len(lgt) >= 2:
                    return lgt
    return ""


# 机构持仓 股东 解禁 http://emweb.securities.eastmoney.com/ShareholderResearch/Index?type=web&code=SZ000063#
def institution_position(code, headers):
    code = add_sh(code, big="big")
    # print(code)
    url_f = 'http://emweb.securities.eastmoney.com/ShareholderResearch/ShareholderResearchAjax?code={}'
    # print(url_f.format(code))
    vv_f = requests.get(url_f.format(code), headers=headers)
    text_f = vv_f.text
    # print(text_f)
    if vv_f.status_code == 200 and text_f:
        detail_f = json.loads(text_f)
        # pprint(detail_f)
        # detail_f = ""
        if detail_f:
            return {
                "ten_share": ten_big_share(detail_f),  # 十大股东
                "ten_current_share": ten_big_current_share(detail_f),  # 十大流通股东
                "institution_position_son": institution_position_son(detail_f, code),  # 机构持仓
                "lift_ban": lift_a_ban(detail_f)  # 解禁
            }
    return ""


# five 十大股东
def ten_big_share(detail_f):
    ten_share1 = detail_f.get("sdgd", "")
    # print(ten_share1)
    # ten_share = ""
    if ten_share1:
        ten_share = []
        for i in ten_share1:
            shareholder = i.get("sdgd", "")
            rq = i.get("rq", "")
            # print("poi", rq)
            # shareholder = ""
            if shareholder:
                share = {}
                ss = [["股东", "类型", "数量", "占总比", "增减", "增减比例"]]
                t = 0
                for ii in shareholder:
                    # print(ii)
                    gdmc = ii.get("gdmc", "")
                    gflx = ii.get("gflx", "")
                    cgs = ii.get("cgs", "")
                    if cgs:
                        cgs = float(cgs.replace(",", ""))
                        if cgs > 1000000:
                            cgs = '{:.2f}亿'.format(cgs/100000000)
                        else:
                            cgs = '{}股'.format(cgs)
                    zltgbcgbl = ii.get("zltgbcgbl", "")
                    if zltgbcgbl:
                        if zltgbcgbl.endswith("%"):
                            t += float(zltgbcgbl.replace("%", ""))
                        else:
                            t += float(zltgbcgbl)
                    zj = ii.get("zj", "")
                    bdbl = ii.get("bdbl", "")
                    ss.append([gdmc, gflx, cgs, zltgbcgbl, zj, bdbl])
                ss.append(["合计", "", "", '{:.2f}%'.format(t), "", ""])
                share["rq"] = rq
                share["sdgd"] = ss
                ten_share.append(share)
    # pprint(ten_share)
    return ten_share


# six 十大流通股东
# {'rq': '2021-03-31', 'mc': '1', 'gdmc': '中国铁路工程集团有限公司', 'gdxz': '其它', 'gflx': 'A股,H股',
# 'cgs': '11,598,764,390', 'zltgbcgbl': '47.21%', 'zj': '不变', 'bdbl': '--'}
def ten_big_current_share(detail_f):
    ten_share1 = detail_f.get("sdltgd", "")  # 流通股东
    # pprint(ten_share)
    # ten_share = ""
    if ten_share1:
        ten_share = []
        for i in ten_share1:
            shareholder = i.get("sdltgd", "")
            rq = i.get("rq", "")
            # print("poi", rq)
            # print("poi", shareholder)
            # shareholder = ""
            if shareholder:
                share = {}
                ss = [["股东", "股东性质", "类型", "数量", "占流通比", "增减", "增减比例"]]
                t = 0
                for ii in shareholder:
                    # print(ii)
                    gdmc = ii.get("gdmc", "")
                    gdxz = ii.get("gdxz", "")
                    gflx = ii.get("gflx", "")
                    cgs = ii.get("cgs", "")
                    if cgs:
                        cgs = float(cgs.replace(",", ""))
                        if cgs > 1000000:
                            cgs = '{:.2f}亿'.format(cgs / 100000000)
                        else:
                            cgs = '{}股'.format(cgs)
                    zltgbcgbl = ii.get("zltgbcgbl", "")
                    if zltgbcgbl:
                        if zltgbcgbl.endswith("%"):
                            t += float(zltgbcgbl.replace("%", ""))
                        else:
                            t += float(zltgbcgbl)
                    zj = ii.get("zj", "")
                    bdbl = ii.get("bdbl", "")
                    ss.append([gdmc, gdxz, gflx, cgs, zltgbcgbl, zj, bdbl])
                ss.append(["合计", "", "", "", '{:.2f}%'.format(t), "", ""])
                share["rq"] = rq
                share["sdgd"] = ss
                ten_share.append(share)
    # pprint(ten_share)
    return ten_share


# seven 机构持仓son {"rq":"2021-06-30","jglx":"基金","ccjs":"19","ccgs":"151888747","zltgbl":"0.75%","zltgbbl":"0.62%"}
def institution_position_son(detail_f, code):
    quarter_n = detail_f.get("zlcc_rz", "")
    # print(quarter_n[0:3])
    if quarter_n:
        lgt = []
        for v in quarter_n:
            s = {}
            share_current = [["类型", "家数", "数量", "占流通比", "占总比"]]
            url = 'http://emweb.securities.eastmoney.com/ShareholderResearch/MainPositionsHodlerAjax?date={}&code={}'
            vv = requests.get(url.format(v, code))
            # print(vv.status_code)
            text = vv.text
            # print(text)
            if vv.status_code == 200 and text:
                detail = json.loads(text)
                # print(detail)
                for d in detail:
                    jglx = d.get("jglx", "")
                    ccjs = d.get("ccjs", "")
                    ccgs = d.get("ccgs", "")
                    if ccgs:
                        if ccgs != "--":
                            ccgs = float(ccgs)
                            if ccgs > 1000000:
                                ccgs = '{:.2f}亿'.format(ccgs / 100000000)
                            else:
                                ccgs = '{}股'.format(ccgs)
                    zltgbl = d.get("zltgbl", "")
                    zltgbbl = d.get("zltgbbl", "")
                    if ccjs != "--":
                        share_current.append([jglx, ccjs, ccgs, zltgbl, zltgbbl])
                s["rq"] = v
                # print(v)
                s["sdgd"] = share_current
                lgt.append(s)
    # pprint(lgt)
    return lgt


# night 未来解禁 {'jjsj': '2022-05-28', 'jjsl': '96.63万', 'jjgzzgbbl': '0.25%', 'jjgzltgbbl': '0.43%', 'gplx': '股权激励限售股份'}
def lift_a_ban(detail_f):
    quarter_n = detail_f.get("xsjj", "")
    if quarter_n:
        s = [["时间", "数量", "占总比", "占流通比", "类型"]]
        for i in quarter_n:
            s.append(list(i.values()))
        # print(s)
        return s
    return ""


# ten 东财个股陆股通详细 ,
def east_lgt_detail(code, headers):
    url = "http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?callback=&st=HDDATE&sr=-1&ps=10&p=1&type=HSGTHDSTA&token=894050c76af8597a853f5b408b759f5d&js=%7B%22data%22%3A(x)%2C%22pages%22%3A(tp)%2C%22font%22%3A(font)%7D&filter=(SCODE%3D%27{}%27)"
    vv = requests.get(url.format(code), headers=headers)
    # print(vv.status_code)
    text = vv.text
    # print(text)
    if vv.status_code == 200 and text:
        detail = json.loads(text).get("data", "")
        # print(detail)
        if detail:
            lgt = [["日期", "占总比", "市值"]]
            for v in detail:
                date = v.get("HDDATE", "")
                zb = v.get("SHARESRATE", "")
                hold = v.get("SHAREHOLDPRICE", "")
                if zb:
                    zb = '{:.2f}%'.format(zb)
                # print(zb)
                if hold:
                    hold = '{:.2f}亿'.format(hold/100000000)
                # print(hold)
                if date:
                    date = date[0:10]
                lgt.append([date, zb, hold])
            # print(lgt)
            return lgt
    return ""


# eleven 东财解禁详细
def east_lift_ban(code, headers):
    url = "http://dcfm.eastmoney.com/em_mutisvcexpandinterface/api/js/get?callback=&st=ltsj&sr=-1&ps=50&p=1&token=70f12f2f4f091e459a279469fe49eca5&type=XSJJ_NJ_PC&js=%7B%22data%22%3A(x)%2C%22pages%22%3A(tp)%2C%22font%22%3A(font)%7D&filter=(gpdm%3D%27{}%27)"
    vv = requests.get(url.format(code), headers=headers)
    # print(vv.status_code)
    text = vv.text
    # print(text)
    if vv.status_code == 200 and text:
        detail = json.loads(text).get("data", "")
        # print(detail)
        if detail:
            lgt = {}
            for v in detail:
                zb = v.get("zb", "")  # 流通比
                total_zb = v.get("zzb", "")  # 总比
                date = v.get("ltsj", "")
                t_type = v.get("xsglx", "")  # 限售股类型
                if zb:
                    zb = '{:.2f}%'.format(zb)
                # print(zb)
                if total_zb:
                    total_zb = '{:.2f}'.format(total_zb)
                # print(hold)
                if date:
                    date = date[0:10]
                    e = (datetime.strptime(date, '%Y-%m-%d') - datetime.strptime(str(d_date.today() + timedelta(-180)), '%Y-%m-%d'))
                    # print(e.days)
                    if e.days < 0:
                        break
                lgt[date] = zb + " / " + total_zb + " / " + t_type
                # print(date)
            return lgt
    return ""


# twelve 东财股东增持 半年 http://data.eastmoney.com/executive/gdzjc/000785.html
def east_add_subtract(code, headers):
    url = "http://datainterface3.eastmoney.com/EM_DataCenter_V3/api/GDZC/GetGDZC?js=&pageSize=50&pageNum=1&tkn=eastmoney&cfg=gdzc&secucode={}&fx=&sharehdname=&sortFields=BDJZ&sortDirec=1&startDate=&endDate=&_=1622589484998"
    vv = requests.get(url.format(code), headers=headers)
    # print(vv.status_code)
    text = vv.text
    # print(text)
    # 名称 , 增减,方式 变动占总股比 , 变动占流通比 ,剩股占总股比 ,剩股占流通比, 开始日 , 截至日,公告日
    if vv.status_code == 200 and text:
        detail = json.loads(text).get("Data", "")[0].get("Data", "")
        # print(detail)
        if detail:
            lgt = [["名称", "增减", "方式", "变动占总股比%", "变动占流通比%", "剩股占总股比%", "剩股占流通比%", "开始日", "截至日", "公告日"]]
            for v in detail:
                if v:
                    de = v.split("|")
                    # print(de)
                    e = (datetime.strptime(de[-1], '%Y-%m-%d') - datetime.strptime(str(d_date.today() + timedelta(-190)), '%Y-%m-%d'))
                    # print(e.days)
                    if e.days < 0:
                        break
                    if de[9]:
                        de[9] = '{:.2f}'.format(float(de[9]))
                    if de[10]:
                        de[10] = '{:.2f}'.format(float(de[10]))
                    lgt.append([de[6], de[7], de[11], de[10], de[9], de[-6], de[-4], de[-3], de[-2], de[-1]])
            # print(lgt)
            if len(lgt) >= 2:
                return lgt
    return ""


# thirteen 东财高管增持 1年 http://data.eastmoney.com/executive/000785.html
def manager_add(code, headers):
    url = 'http://datainterface.eastmoney.com/EM_DataCenter/JS.aspx?cb=&type=GG&sty=GGC&p=1&ps=30&code={}&name=&js=%7B"pages"%3A(pc)%2C"data"%3A%5B(x)%5D%7D&_=1622603931514'
    # print(url.format(code))
    vv = requests.get(url.format(code), headers=headers)
    # print(vv.status_code)
    text = vv.text
    # print(text)
    if vv.status_code == 200 and text:
        detail = demjson.decode(text, encoding='utf-8').get("data", "")  # 把json字符串变json对象
        # print(detail)
        if detail:
            lgt = [['日期', '变动人', '变动数亿', '均价', '比例', '变动后亿', '原因', '类型', '懂监高', '关系']]
            # 日期，变动人，变动数亿，均价，比例，变动后亿，原因，类型，懂监高，关系
            for v in detail:  # '{"pages": 0, "data": [{stats: false}]}' 不是字符串{'stats': False}
                if v:
                    # print(v)
                    if not isinstance(v, str):
                        if not v.get("stats", ""):
                            break
                    de = v.split(",")
                    # print(de)
                    e = (datetime.strptime(de[5], '%Y-%m-%d') - datetime.strptime(str(d_date.today() + timedelta(-365)), '%Y-%m-%d'))
                    # print(e.days)
                    if e.days < 0:
                        break
                    # print(float(de[6]))
                    if de[6]:
                        de[6] = '{:.2f}'.format(float(de[6]) / 100000000)
                    if de[7]:
                        de[7] = '{:.2f}'.format(float(de[7]) / 100000000)
                    if de[0]:
                        de[0] = '{:.2f}%'.format(float(de[0]) / 10)
                    lg = [de[5], de[1], de[6], de[8], de[0], de[7], de[-3], de[4], de[-1], de[-5]]
                    lgt.append(lg)
            # print(lgt)
            if len(lgt) >= 2:
                return lgt
    return ""


# fourteen 东财个股融资融券 10天 http://data.eastmoney.com/rzrq/detail/000785.html
def east_rz(code, headers):
    url = 'http://datacenter-web.eastmoney.com/api/data/get?type=RPTA_WEB_RZRQ_GGMX&sty=ALL&source=WEB&st=DATE&sr=-1&p=1&ps=240&filter=(scode={})&callback=&_=1622688470305'
    vv = requests.get(url.format(code), headers=headers)
    # print(vv.status_code)
    text = vv.text
    # print(text)
    if vv.status_code == 200 and text:
        detail = json.loads(text).get("result", "")
        # print(detail)
        # detail = ""
        if detail:
            d = detail.get("data", "")
            # print(d)
            # d = ""
            if d:
                lgt = [['日期', '融资余额', '余额占流通', '融资尽买入', '融券余额', '融资融券差值']]
                # 日期DATE，融资余额RZYE，流通比RZYEZB，'RZJME，'RQYE，融资融券差值 RZRQYECZ
                for v in d:
                    # print(v)
                    # v = ""
                    if v:
                        dat = v.get("DATE", "")
                        # print(dat1)
                        if dat:
                            da = datetime.strptime(dat, "%Y-%m-%d %H:%M:%S")
                            # print(da.date())
                            e = (da - datetime.strptime(str(d_date.today() + timedelta(-10)), '%Y-%m-%d'))
                            # print(e.days)
                            if e.days < 0:
                                break
                            rzye = v.get("RZYE", "")
                            rzzb = v.get("RZYEZB", "")
                            rzjmr = v.get("RZJME", "")
                            rqye = v.get("RQYE", "")
                            rzrqcz = v.get("RZRQYECZ", "")
                            if rzye:
                                rzye = '{:.2f}亿'.format(float(rzye) / 100000000)
                            if rzzb:
                                rzzb = '{:.2f}%'.format(float(rzzb))
                            if rzjmr:
                                rzjmr = '{:.2f}亿'.format(float(rzjmr) / 100000000)
                            if rqye:
                                rqye = '{:.2f}亿'.format(float(rqye) / 100000000)
                            if rzrqcz:
                                rzrqcz = '{:.2f}亿'.format(float(rzrqcz) / 100000000)
                            lg = [str(da.date()), rzye, rzzb, rzjmr, rqye, rzrqcz]
                            lgt.append(lg)
                            # print(lg)
                if len(lgt) >= 2:
                    return lgt
    return ""


# fifteen 东财主力资金流入
def east_zllr(code, headers):
    # code = code_add(code, param="1.")
    # print(code)
    url = 'http://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get?cb=&lmt=0&klt=101&fields1=f1%2Cf2%2Cf3%2Cf7&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61%2Cf62%2Cf63%2Cf64%2Cf65&ut=b2884a393a59ad64002292a3e90d46a5&secid={}&_=1622756188471'
    vv = requests.get(url.format(code_add(code, param="1.")), headers=headers)
    # print(url.format(code_add(code, param="1.")))
    # print(vv.status_code)
    text = vv.text
    # print(text)
    if vv.status_code == 200 and text:
        detail = json.loads(text).get("data", "")
        # print(detail)
        # detail = ""
        if detail:
            d = detail.get("klines", "")[::-1]
            # print(d)
            # d = ""
            if d:
                lgt = [["日期", "流入净额"]]
                # 日期DATE，流入净额RZYE
                for v in d:
                    # print(v)
                    # v = ""
                    if v:
                        vv = v.split(",")
                        # print(vv)
                        # vv = ""
                        if vv:
                            da = datetime.strptime(vv[0], "%Y-%m-%d")
                            # print(da.date())
                            e = (da - datetime.strptime(str(d_date.today() + timedelta(-10)), '%Y-%m-%d'))
                            # print(e.days)
                            if e.days < 0:
                                break
                            if vv[1]:
                                mf = '{:.2f}亿'.format(float(vv[1]) / 100000000)
                            lg = [str(da.date()), mf]
                            lgt.append(lg)
                            # print(lg)
                if len(lgt) >= 2:
                    return lgt
    return ""


# sixteen 东财机构调研 90天 http://data.eastmoney.com/jgdy/gsjsdy/002913.html
def institution_research(code, headers):
    url = 'http://datainterface3.eastmoney.com/EM_DataCenter_V3/api/JGDYHZ/GetJGDYMX?js=&tkn=eastmoney&secuCode={}&sortfield=1&sortdirec=1&pageNum=1&pageSize=700&cfg=jgdyhz&_=1622808241272'
    vv = requests.get(url.format(code), headers=headers)
    # print(vv.status_code)
    text = vv.text
    # print(text)
    if vv.status_code == 200 and text:
        detail = json.loads(text).get("Data", "")
        # print(detail)
        # detail = ""
        if detail:
            # d = detail[0].get("Data", "")
            d = detail[0].get("Data", "")[::-1]
            # print(d)
            # d = ""
            if d:
                lgt = [["公告日", "调研日", "数量", "方式"]]
                # 公告日，调研日，数量，方式
                for v in d:
                    # print(v)
                    # v = ""
                    if v:
                        vv = v.split("|")
                        # print(vv)
                        # vv = ""
                        if vv:
                            da = datetime.strptime(vv[7], "%Y-%m-%d")
                            # print(da.date())
                            e = (da - datetime.strptime(str(d_date.today() + timedelta(-90)), '%Y-%m-%d'))
                            # print(e.days)
                            if e.days < 0:
                                break
                            lg = [vv[7], vv[8], vv[4], vv[11]]
                            lgt.append(lg)
                            # print(lg)
                if len(lgt) >= 2:
                    return lgt
    return ""


# seventeen 个股研究报告 半年内
def ins_research_report(code, headers):
    d = datetime.strptime(str(d_date.today() + timedelta(-190)), '%Y-%m-%d')  # 半年去日期
    url = 'http://reportapi.eastmoney.com/report/list?cb=&pageNo=1&pageSize=600&code={}&industryCode=*&industry=*&rating=*&ratingchange=*&beginTime={}&endTime={}&fields=&qType=0&_=1622819583383'
    vv = requests.get(url.format(code, d.date(), d_date.today()), headers=headers)
    # print(vv.status_code)
    text = vv.text
    # print(text)
    if vv.status_code == 200 and text:
        detail = json.loads(text).get("data", "")
        # print(detail)
        # detail = ""
        if detail:
            lgt = [["日期", "标题"]]
            # 公告日，标题
            for v in detail:
                # print(v)
                # v = ""
                if v:
                    vv = v.get("publishDate", "")
                    tit = v.get("title", "")
                    # print(tit)
                    # vv = ""
                    if vv:
                        lg = [vv[0:10], tit]
                    else:
                        lg = ["", tit]
                    lgt.append(lg)
                    # print(lg)
            return lgt
    return ""


# eighteen 股东户数 http://data.eastmoney.com/gdhs/detail/601179.html
def shareholder_number(code, headers):
    url = 'http://datacenter-web.eastmoney.com/api/data/v1/get?callback=&sortColumns=END_DATE&sortTypes=-1&pageSize=100&pageNumber=1&reportName=RPT_HOLDERNUM_DET&columns=SECURITY_CODE%2CSECURITY_NAME_ABBR%2CCHANGE_SHARES%2CCHANGE_REASON%2CEND_DATE%2CINTERVAL_CHRATE%2CAVG_MARKET_CAP%2CAVG_HOLD_NUM%2CTOTAL_MARKET_CAP%2CTOTAL_A_SHARES%2CHOLD_NOTICE_DATE%2CHOLDER_NUM%2CPRE_HOLDER_NUM%2CHOLDER_NUM_CHANGE%2CHOLDER_NUM_RATIO%2CEND_DATE%2CPRE_END_DATE&quoteColumns=f2%2Cf3&filter=(SECURITY_CODE%3D%22{}%22)&source=WEB&client=WEB'
    vv = requests.get(url.format(code), headers=headers)
    # print(vv.status_code)
    text = vv.text
    # print(text)
    if vv.status_code == 200 and text:
        detail = json.loads(text).get("result", "")
        # print(detail)
        # detail = ""
        if detail:
            li = detail.get("data", "")
            if li:
                lgt = [["公告日", "截至", "股东人数", "增减比", "总市值", "股本数量", "股本变动", "原因"]]
                # 公告日，截至，增减比,总市值，股东数量，变化股，原因
                for v in li:
                    # print(v)
                    # v = ""
                    if v:
                        notice_dat = v.get("HOLD_NOTICE_DATE", "")
                        if notice_dat:
                            da = datetime.strptime(notice_dat[0:10], "%Y-%m-%d")
                            # print(da.date())
                            e = (da - datetime.strptime(str(d_date.today() + timedelta(-365)), '%Y-%m-%d'))
                            # print(e.days)
                            if e.days < 0:
                                break
                        end_da = v.get("END_DATE", "")
                        if end_da:
                            end_da = end_da[0:10]
                        holder_num = v.get("HOLDER_NUM", "")
                        add_rate = v.get("HOLDER_NUM_RATIO", "")
                        total = v.get("TOTAL_MARKET_CAP", "")
                        share_num = v.get("TOTAL_A_SHARES", "")
                        change_share = v.get("CHANGE_SHARES", "")
                        # change_reason = v.get("CHANGE_REASON", "")
                        if add_rate:
                            add_rate = '{:.2f}%'.format(add_rate)
                        if total:
                            total = '{:.2f}亿'.format(total/100000000)
                        if share_num:
                            share_num = '{:.2f}亿'.format(share_num/100000000)
                        if change_share:
                            change_share = '{:.2f}亿'.format(change_share/100000000)
                        # print(tit)
                        lg = [str(da.date()), end_da, holder_num, add_rate, total, share_num,
                              change_share, v.get("CHANGE_REASON", "")]
                        lgt.append(lg)
                        # print(lg)
                if len(lgt) >= 2:
                    return lgt
    return ""


# nineteen 龙虎榜
def per_dragon_tiger(code, headers):
    date_li = per_dragon_tiger1(code, headers)
    if len(date_li) > 2:
        date_li = date_li[0:2]
    print(date_li)
    return per_dragon_tiger2(code, date_li, headers)


#  龙虎榜son1
def per_dragon_tiger1(code, headers):
    url = 'http://datainterface3.eastmoney.com/EM_DataCenter_V3/api/LHBGGSBRQ/GetLHBGGSBRQ?tkn=eastmoney&scode={}&dayNum=100&startDateTime=&endDateTime=&sortField=1&sortDirec=1&pageNum=1&pageSize=1000&cfg=ggsbrq&js='
    vv = requests.get(url.format(code), headers=headers)
    # print(vv.status_code)
    text = vv.text
    # print(text)
    lgt = []
    if vv.status_code == 200 and text:
        detail = json.loads(text).get("Data", "")
        # print(detail)
        # detail = ""
        if detail:
            li = detail[0].get("Data", "")
            if li:
                for v in li:
                    # print(v)
                    # v = ""
                    if v:
                        notice_dat = v.split("|")
                        if notice_dat:
                            lgt.append(notice_dat[1])
                            # print(notice_dat[1])
    # print(lgt)
    return lgt


#  龙虎榜son2
def per_dragon_tiger2(code, date_li, headers):
    url = 'http://datainterface3.eastmoney.com/EM_DataCenter_V3/api/LHBMMMX/GetLHBMXKZ?tkn=eastmoney&Code={}&dateTime={}&pageNum=1&pageSize=50&cfg=lhbmxkz&js='
    lgt = []
    if date_li and code:
        for d in date_li:
            vv = requests.get(url.format(code, d), headers=headers)
            # print(vv.status_code)
            text = vv.text
            # print(text)
            if vv.status_code == 200 and text:
                detail = json.loads(text).get("Data", "")
                # print(detail)
                # detail = ""
                if detail:
                    li = detail[0].get("Data", "")
                    if li:
                        # 13日期,14类型,排名2， 19交易营业， 21上榜次数，22胜率,11买入，17卖入占总成交比,卖出9，卖出占总成交比16，净额20
                        lg = []
                        flag = ""
                        for v in li:
                            # print(v)
                            # v = ""
                            if v:
                                n = v.split("|")
                                if n:
                                    if n[12]:  # 13日期
                                        n[12] = n[12].split(" ")[0]
                                    # print(flag != n[13])
                                    if flag != n[13]:  # 类型
                                        flag = n[13]
                                        lg.append(["", n[13], n[12], "", "", "", "", "", ""])
                                        lg.append(["", "营业部", "上榜次数", "胜率%", "买入", "买入占成交%", "卖出", "卖出占成交%", "余额"])
                                    if n[10]:
                                        n[10] = '{:.2f}亿'.format(float(n[10]) / 100000000)
                                    if n[8]:
                                        n[8] = '{:.2f}亿'.format(float(n[8]) / 100000000)
                                    if n[19]:
                                        n[19] = '{:.2f}亿'.format(float(n[19]) / 100000000)
                                    lg.append([n[1], n[18], n[20], n[21], n[10], n[16], n[8], n[15], n[19]])
                        lgt.append(lg)
    # pprint(lgt)
    return lgt


# twenty one雪球讨论
def xiu_qiu_discuss(code):
    code = add_sh(code, big="big")
    # print(code)
    # return ""
    url = 'https://xueqiu.com/query/v1/symbol/search/status?u=1654388911&uuid=1401490525072289792&count=30&comment=0&symbol={}&hl=0&source=all&sort=alpha&page=1&q=&type=11&session_token=null&access_token=d974567d4ecd24e8208ec9590638a9bcc706f5ed'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36',
        'Cookie': get_cookie(wet="xueqiu.com")
    }
    vv = requests.get(url.format(code), headers=headers)
    # print(vv.status_code)
    text = vv.text
    # print(text)
    if vv.status_code == 200 and text:
        detail = json.loads(text).get("list", "")
        # print(detail)
        # detail = ""
        if detail:
            lgt = [["日期", "标题", "描述"]]
            # 公告日，标题,描述
            for v in detail:
                # print(v)
                # v = ""
                if v:
                    # dis_time = v.get("timeBefore", "")
                    # title = v.get("title", "")
                    description = v.get("description", "")
                    if description:
                        description = re.sub('<[^<]+?>', '', description).replace('\n', '').strip()
                        # print(c)
                    # lg = {"dis_time": v.get("timeBefore", ""), "title": v.get("title", ""), "description": description}
                    lgt.append([v.get("timeBefore", ""), v.get("title", ""), description])
            # print(lgt)
            if len(lgt) >= 2:
                return lgt
    print(code, "雪球cookie有误")
    open_chrome(url="https://xueqiu.com/S/SH600693")
    xq_dis += 1
    return ""


# twenty two 雪球资信 https://xueqiu.com/S/SH600693
def xiu_qiu_new(code):
    code = add_sh(code, big="big")
    # print(code)
    # return ""
    url = 'https://xueqiu.com/statuses/stock_timeline.json?symbol_id={}&count=30&source=%E8%87%AA%E9%80%89%E8%82%A1%E6%96%B0%E9%97%BB&page=1'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36',
        'Cookie': get_cookie(wet="xueqiu.com")
    }
    vv = requests.get(url.format(code), headers=headers)
    # print(vv.status_code)
    text = vv.text
    # print(text)
    if vv.status_code == 200 and text:
        detail = json.loads(text).get("list", "")
        # print(detail)
        # detail = ""
        if detail:
            lgt = [["日期", "标题", "描述"]]
            # 公告日，标题,描述
            for v in detail:
                # print(v)
                # v = ""
                if v:
                    # dis_time = v.get("timeBefore", "")
                    # title = v.get("rawTitle", "")
                    description = v.get("description", "")
                    if description:
                        description = re.sub('<[^<]+?>', '', description).replace('\n', '').strip()
                    # print(tit)
                    lg = [v.get("timeBefore", ""), v.get("rawTitle", ""), description]
                    # lg = {"dis_time": v.get("timeBefore", ""), "title": v.get("rawTitle", ""), "description": description}
                    lgt.append(lg)
                    # print(lg)
            # print(lgt)
            if len(lgt) >= 2:
                return lgt
    print(code, "资信有误")
    return ""


# twenty three 个股公告 个股详情最后一项
def stock_notice(code, headers):
    url = 'http://np-anotice-stock.eastmoney.com/api/security/ann?cb=&sr=-1&page_size=300&page_index=1&ann_type=A&client_source=web&stock_list={}&f_node=0&s_node=0'
    vv = requests.get(url.format(code), headers=headers)
    # print(vv.status_code)
    text = vv.text
    # print(text)
    if vv.status_code == 200 and text:
        detail = json.loads(text).get("data", "")
        # print(detail)
        # detail = ""
        if detail:
            li = detail.get("list", "")
            if li:
                lgt = [["日期", "类型", "标题"]]
                # 公告日，类型，标题
                for v in li:
                    # print(v)
                    # v = ""
                    if v:
                        dis_time = v.get("display_time", "")
                        if dis_time:
                            da = datetime.strptime(dis_time[0:10], "%Y-%m-%d")
                            # print(da.date())
                            e = (da - datetime.strptime(str(d_date.today() + timedelta(-90)), '%Y-%m-%d'))
                            # print(e.days)
                            if e.days < 0:
                                break
                        col = v.get("columns", "")
                        col_type = ""
                        if col:
                            col_type = col[0].get("column_name", "")
                        lgt.append([dis_time, col_type, v.get("title", "")])
                if len(lgt) >= 2:
                    return lgt
    return ""


# 交集和并集
def combine():
    # folder = r"D:\myzq\axzq\T0002\blocknew"
    east_finance_number = read_file("east_finance_number.blk")
    east_lgt_number = read_file("east_lgt_number.blk")
    research_report = read_file("research_report.blk")
    research_organization = read_file("research_organization.blk")
    east_finance_number.extend(east_lgt_number)
    east_finance_number.extend(research_report)
    east_finance_number.extend(research_organization)
    print(len(east_finance_number))
    east_finance_number = list(set(east_finance_number))
    le = len(east_finance_number)
    print(le, "并集")
    pp = is_not_path("combine.blk")
    with open(pp, "w") as f:  # 自动关闭
        f.writelines(east_finance_number)
    tmp = list(set(east_finance_number).intersection(set(east_lgt_number)))
    print(len(tmp), "交集1")
    tmp = list(set(tmp).intersection(set(research_report)))
    print(len(tmp), "交集2")
    tmp = list(set(tmp).intersection(set(research_organization)))
    print(len(tmp), "")
    p = is_not_path("contraction.blk")
    with open(p, "w") as f:  # 自动关闭
        f.writelines(tmp)
    # return ""
    return "并" + str(le) + "交" + str(len(tmp))


# 涨幅 'Zdf': 1.1566,占流通比 'LTZB': 0.060944376156863  净买入 'ShareSZ_Chg_One': 535678176.0,
def east_lgt_finance(date, page, choice):
    if choice == "east_lgt":
        lgt = requests.get("http://dcfm.eastmoney.com/EM_MutiSvcExpandInterface/api/js/get?type=HSGT20_GGTJ_SUM"
                           "&token=894050c76af8597a853f5b408b759f5d&st=ShareSZ_Chg_One&sr=-1&p=1&ps=" + page +
                           "&json={pages:(tp),data:(x)}&filter=(DateType=%271%27%20and%20HdDate=%27" + date + "%27)"
                           "&rt=53887112")
        if lgt.status_code == 200:
            stock_list = []
            for item in lgt.json():
                if item.get('ShareSZ_Chg_One', '0') > 10000000 and item.get('Zdf', '0') > 1 and item.get('LTZB', '0') > 0.001:
                    code = code_add(item.get('SCode', ''))
                    stock_list.append(code+'\n')
                    # print(code)
                    # print(item.get('LTZB', '0'))
            print(stock_list)
            le = len(stock_list)
            print(le, "交前")
            ths_fund_inflow0 = read_file("ths_fund_inflow0.blk")
            if ths_fund_inflow0 and le:
                fund_inflow_list = sorted(set(stock_list).intersection(ths_fund_inflow0),
                                          key=lambda x: stock_list.index(x))
                print(fund_inflow_list, "交集后")
                print(len(fund_inflow_list), "交集后股数")
                is_write_stock('lgt.blk', fund_inflow_list, "write")
                return "陆股通交前" + str(le) + "交后" + str(len(fund_inflow_list))
    elif choice == "east_lgt_number":
        # if not date:
        #     date = str(d_date.today())  # 今天
        #     print(date)
        lgt = requests.get("http://dcfm.eastmoney.com/EM_MutiSvcExpandInterface/api/js/get?type=HSGT20_GGTJ_SUM"
                           "&token=894050c76af8597a853f5b408b759f5d&st=ShareSZ_Chg_One&sr=-1&p=1&ps=" + page +
                           "&json={pages:(tp),data:(x)}&filter=(DateType=%271%27%20and%20HdDate=%27" + date + "%27)"
                                                                                                              "&rt=53887112")
        if lgt.status_code == 200:
            stock_list = []
            for item in lgt.json():
                code = code_add(item.get('SCode', ''))
                stock_list.append(code + '\n')
            # print(stock_list)
            print(len(stock_list))
            is_write_stock('east_lgt_number.blk', stock_list, "write")
            return "陆股通数量" + str(len(stock_list))
    elif choice == "east_finance_sh":
        finance = requests.get("http://datacenter.eastmoney.com/api/data/get?callback=&type=RPTA_WEB_RZRQ_GGMX&sty=ALL&source=WEB&p=1&ps={}&st=RZJME&sr=-1&filter=(TRADE_MARKET_CODE+in+(%22069001001001%22%2C%22069001001006%22))(date%3D%27{}%27)&pageNo=1&_=1620278889766".format(page, date))
        print(finance)
        text = finance.text
        if finance.status_code == 200 and text:
            result = json.loads(text).get("result", "")
            if result:
                data = result.get("data", "")
                if data:
                    stock_list = []
                    for item in data:
                        # print(item)
                        if item.get('RZJME', '0') > 10000000 and item.get('ZDF', '0') > 1:
                            code = code_add(item.get('SCODE', ''))
                            stock_list.append(code + '\n')
                    print(stock_list)
                    le = len(stock_list)
                    print(le)
                    ths_fund_inflow0 = read_file("ths_fund_inflow0.blk")
                    if ths_fund_inflow0 and le:
                        fund_inflow_list = sorted(set(stock_list).intersection(ths_fund_inflow0),
                                                  key=lambda x: stock_list.index(x))
                        print(fund_inflow_list, "交集后股数")
                        print(len(fund_inflow_list), "交集后股数")
                        is_write_stock('east_finance_sh.blk', fund_inflow_list, "write")
                        return "上海交前" + str(le) + "交后" + str(len(fund_inflow_list))
        return ""
    elif choice == "east_finance_number":
        stock_list = []
        pages = 1
        for i in range(1, 7):
            # print(i)
            # print(pages)
            # print(i > pages)
            if i > pages:
                break
            finance = requests.get("http://datacenter.eastmoney.com/api/data/get?type=RPTA_WEB_RZRQ_GGMX&sty=ALL&source=WEB&p={}&ps={}&st=rzjme&sr=-1&filter=(date=%27{}%27)&_=1620158762614".format(i, page, date))
            sleep(2.5)
            print(finance)
            data = json.loads(finance.text).get("result", "")
            if finance.status_code == 200 and data:
                if i == 1:
                    pages = data.get("pages", "7")
                    # print(pages)
                for item in data.get("data", ""):
                    code = code_add(item.get('SCODE', ''))
                    stock_list.append(code + '\n')
                    # print(stock_list)
        print(len(stock_list))
        is_write_stock('east_finance_number.blk', stock_list, "write")
        return "两融" + str(len(stock_list))
    elif choice == "east_finance_sz":
        finance = requests.get("http://datacenter.eastmoney.com/api/data/get?callback=&type=RPTA_WEB_RZRQ_GGMX&sty=ALL&source=WEB&p=1&ps={}&st=RZJME&sr=-1&filter=(TRADE_MARKET_CODE+in+(%22069001002001%22%2C%22069001002002%22%2C%22069001002003%22))(date%3D%27{}%27)&pageNo=1&_=1620265586428".format(page, date))
        print(finance)
        text = finance.text
        if finance.status_code == 200 and text:
            result = json.loads(text).get("result", "")
            if result:
                data = result.get("data", "")
                if data:
                    stock_list = []
                    for item in data:
                        # print(item)
                        if item.get('RZJME', '0') > 10000000 and item.get('ZDF', '0') > 1:
                            code = code_add(item.get('SCODE', ''))
                            stock_list.append(code + '\n')
                    # print(stock_list)
                    le = len(stock_list)
                    print(le)
                    ths_fund_inflow0 = read_file("ths_fund_inflow0.blk")
                    if ths_fund_inflow0 and le:
                        fund_inflow_list = sorted(set(stock_list).intersection(ths_fund_inflow0),
                                                  key=lambda x: stock_list.index(x))
                        print(fund_inflow_list, "交集")
                        print(len(fund_inflow_list), "交集后股数")
                        is_write_stock('east_finance_sz.blk', fund_inflow_list, "write")
                        return "深圳交前" + str(le) + "交后" + str(len(fund_inflow_list))
        return ""


# 读东财研究报告股票数量  all模式有问题，有空再优化
def research_report(start_date="", end_date="", page_size="50", choice="add"):
    # start_date = "2021-05-09"
    # end_date = "2021-05-09"
    # page_size = 100
    # choice = "add"
    if not start_date or not end_date:
        start_date = d_date.today() + timedelta(-365)  # 一年前
        end_date = str(d_date.today())  # 今天
        # print(start_date)
        # print(end_date)
    cursor = connection.cursor()
    if choice == "add":
        # 查新语句
        cursor.execute("SELECT date FROM m_research_report ORDER BY date DESC LIMIT 0,1")
        rows = cursor.fetchone()
        # rows = cursor.fetchall()
        if len(rows):
            print('rows', rows[0])
            start_date = rows[0]
            stock_list = research_report_son(start_date, end_date, page_size, cursor)
            # print(stock_list)
            le = len(stock_list)
            # print(le)
            if le:
                with transaction.atomic():  # 都在事物中，要么都成功，要么都失败
                    # 删除语句
                    ccc = cursor.execute("delete from m_research_report where date=%s", [start_date])
                    num = ccc.rowcount
                    # print("删除数量", num)
                    if ccc.rowcount:
                        # 插入语句
                        sql = "insert into m_research_report(code,name,date) values ( %s, %s, %s)"
                        for item in stock_list:
                            c = cursor.execute(sql, list(item))
                            # print(c.rowcount)
                            if not c.rowcount:
                                print(name, "无插入")
                        end_date_one = d_date.strftime(parser.isoparse(end_date) - relativedelta(months=12),
                                                       '%Y-%m-%d')
                        # print(end_date_one)
                        cursor.execute("SELECT distinct code FROM m_research_report as m WHERE m.date BETWEEN %s AND %s", [end_date_one, end_date])
                        # rows = cursor.fetchone()
                        rows = cursor.fetchall()
                        if len(rows):
                            stock_li = []
                            for item in rows:
                                if len(item):
                                    code1 = code_add(item[0]) + '\n'
                                    stock_li.append(code1)
                            if stock_li:
                                # print("", stock_li)
                                # print("写入数", len(stock_li))
                                is_write_stock('research_report.blk', stock_li, "write")
                                cursor.close()
                                return "研报" + str(len(stock_li))
        cursor.close()
        return ""
    elif choice == "all":
        # is_write_stock('research_report.blk', stock_list, "write")
        return l


# 读东财研究报告股票数量子方法
def research_report_son(start_date, end_date, page_size, cursor):
    stock_list = []
    total_page = 1
    for i in range(1, 450):
        # if i > 2:
        if i > total_page:
            break
        lgt = requests.get("http://reportapi.eastmoney.com/report/list?cb=&industryCode=*&pageSize={}"
                           "&industry=*&rating=*&ratingChange=*&beginTime={}&endTime={}&pageNo={}&fields=&qType=0"
                           "&orgCode=&code=*&rcode=&p=656&pageNum=656&_=1620218614382"
                           .format(page_size, start_date, end_date, i))
        sleep(2)
        status_code = lgt.status_code
        print(str(status_code) + ":" + str(i))
        text = lgt.text
        # print(text)
        if status_code == 200 and text:
            js = json.loads(text)
            data = js.get("data", "")
            if data:
                if i == 1:
                    total_page = js.get("TotalPage", "")
                    print("总页数", total_page)
                for item in data:
                    code = item.get('stockCode', '')
                    name = item.get('stockName', '')
                    if code or name:
                        dat = item.get('publishDate', '')
                        # print(dat)
                        if dat and len(dat) >= 10:
                            dat = dat[0:10]
                            t = (code, name, dat)
                            stock_list.append(t)
                            # print(dat)
                        else:
                            print(name, "日期有误")
                    else:
                        print("数据有误")
    return stock_list


# 读东财机构调研股票数量
def research_organization(start_date="", end_date="", page_size="50"):
    # print(start_date)
    # print(end_date)
    # start_date = "2020-05-05"
    # end_date = "2021-05-05"
    # page_size = "10000"
    if not start_date or not end_date:
        start_date = str(d_date.today() + timedelta(-365))  # 一年前
        end_date = str(d_date.today())  # 今天
    stock_list = []
    total_page = 1
    format_pattern = '%Y-%m-%d'
    for i in range(1, 200):
        # print(pages)
        # print(i > pages)
        # if i > total_page:
        # if i > 250:
        if i > 1:
            break
        lgt = requests.get("http://datainterface3.eastmoney.com/EM_DataCenter_V3/api/JGDYHZ/GetJGDYMX?js=&tkn=eastmoney&secuCode=&sortfield=0&sortdirec=1&pageNum={}&pageSize={}&cfg=jgdyhz&p={}&pageNo={}&_=1620232702240".format(i, page_size, i, i))
        sleep(1.5)
        status_code = lgt.status_code
        print(str(status_code) + ":" + str(i))
        text = lgt.text
        # print(text)
        if status_code == 200 and text:
            data = json.loads(text).get("Data", "")
            # print(js.get("size", "abc"))
            # print(data)
            if data:
                data0 = data[0]
                data = data0.get("Data", "")
                # print(data)
                if data:
                    if i == 1:
                        total_page = data0.get("TotalPage", "")
                        print(total_page)
                    for item in data:
                        # print(item)
                        lis = item.split("|")
                        # print(lis)
                        # print(lis[7])
                        # 将 'time' 类型时间通过格式化模式转换为 'str' 时间
                        end_difference = (datetime.strptime(lis[7], format_pattern) - datetime.strptime(end_date, format_pattern))
                        start_difference = (datetime.strptime(lis[7], format_pattern) - datetime.strptime(start_date, format_pattern))
                        # print(end_difference.days, '在当')
                        if start_difference.days < 0:
                            print(lis[7], '公告日期小于输入开始日期')
                            break
                        if end_difference.days <= 0:
                            # print(lis[7])
                            code = code_add(lis[5]) + '\n'
                            if code not in stock_list:
                                stock_list.append(code)
                                # print(code)
    # print(stock_list)
    l = len(stock_list)
    print(l)
    is_write_stock('research_organization.blk', stock_list, "write")
    return "机调" + str(l)


# 读东财龙虎榜 # 净买入 'JmMoney': '63519965.72',涨幅 'Chgradio': '9.98',
def east_dragon_tiger(date):
    dragon_tiger = requests.get("http://data.eastmoney.com/DataCenter_V3/stock2016/TradeDetail/pagesize=200,page=1,"
                                "sortRule=-1,sortType=,startDate=" + date + ",endDate=" + date + ",gpfw=0,"
                                                                                                 "js=.html?rt=26947717")
    #  读入机构龙虎榜尽买入
    organization_dragon_tiger = requests.get("http://data.eastmoney.com/DataCenter_V3/stock2016/DailyStockListStatistics/pagesize=100,page=1,sortRule=-1,sortType=PBuy,startDate=" + date + ",endDate=" + date + ",gpfw=0,js=.html?rt=26985157")
    print(organization_dragon_tiger)
    if dragon_tiger.status_code == 200 and organization_dragon_tiger.status_code == 200:
        dragon_tiger_list = {}
        for item in dragon_tiger.json().get('data', ''):
            if isfloat(item.get('JmMoney', '0')) > 10000000:
                code = code_add(item.get('SCode', ''))
                # print(code)
                dragon_tiger_list[code+'\n'] = float(item.get('JmMoney', '0'))
        tiger_list = dict(sorted(dragon_tiger_list.items(), key=lambda x: x[1], reverse=True)).keys()  # 返回的是list 按字典集合中，每一个元组的第二个元素排列。
        tiger_list = list(tiger_list)
        # print(tiger_list)
        print(len(tiger_list))

        dragon_tiger_list2 = {}
        for item in organization_dragon_tiger.json().get('data', ''):
            if isfloat(item.get('PBuy', '0')) > 10000000:
                code = code_add(item.get('SCode', ''))
                dragon_tiger_list2[code + '\n'] = float(item.get('PBuy', '0'))
        dragon_tiger_list2['1' + date + '\n'] = 0
        tiger_list2 = dict(sorted(dragon_tiger_list2.items(), key=lambda x: x[1], reverse=True)).keys()  # 返回的是list 按字典集合中，每一个元组的第二个元素排列。
        tiger_list2 = list(tiger_list2)
        # print(tiger_list2)
        print(len(tiger_list2)-1)
        i = 0
        for each in tiger_list2:
            if each not in tiger_list:
                tiger_list.append(each)
                i += 1
        # print(tiger_list)
        is_write_stock('DRAGON_TIGER.blk', tiger_list, "write")
        # combine_code = read_file(r"D:\myzq\axzq\T0002\blocknew\combine.blk")
        # if combine_code:
        #     dragon_tiger_contract = sorted(set(tiger_list).intersection(combine_code), key=lambda x: tiger_list.index(x))
        #     print(len(dragon_tiger_contract), "交集后股数")
        #     # print(dragon_tiger_contract)
        #     is_write_stock('dragon_tiger_contract.blk', dragon_tiger_contract, "write")
        #     return "总=" + str(len(tiger_list)-1) + "机构=" + str(i-1) + "交集=" + str(len(dragon_tiger_contract))
        return "总" + str(len(tiger_list)-1) + "机构" + str(i-1)
    return ""


# 同花顺资金流入, 资金净买入大于10万; 涨幅大于1%
def ths_fund_inflow(date, search, number, choice):
    cookie = get_cookie()
    headers = {
        "Cookie": "chat_bot_session_id=7bcebb62fb52a2de11bf41ec05073886; "
                  "other_uid=Ths_iwencai_Xuangu_dz8du2owtuaf0fcmkdy2h8ewto46jpsh; "
                  "cid=b39c75faa8d86f5eee3a5502e26c76f81617695835; "
                  "v=" + cookie,
        "Connection": "keep-alive",
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36"
    }
    ths_data = {
        'question': date + search,
        'perpage': number,
        'page': '1',
        'log_info': '{"input_type":"typewrite"}',
        'source': 'Ths_iwencai_Xuangu',
        'version': '2.0',
        'add_info': '{"urp":{"scene": 1, "company": 1, "business": 1}, "contentType":"json"}',
    }
    fund_inflow = requests.post("http://x.10jqka.com.cn/unifiedwap/unified-wap/v2/result/get-robot-data",
                                headers=headers, data=ths_data)
    if fund_inflow.status_code == 200:
        fund_inflow_data = fund_inflow.json().get('data', '').get('answer', '')[0].get('txt', '')[0].get('content', '') \
            .get('components', '')[0].get('data', '').get('datas', '')
        fund_inflow_list = []
        for item in fund_inflow_data:
            code = code_add(item.get('code', ''))
            fund_inflow_list.append(code+'\n')
        # print(fund_inflow_list)
        print(len(fund_inflow_list))
        if choice == "ths_fund_inflow":
            # p = is_not_path("combine.blk", flag="1")
            # print(p)
            combine_code = read_file("combine.blk")
            # print(combine_code)
            if combine_code:
                fund_inflow_list = sorted(set(fund_inflow_list).intersection(combine_code), key=lambda x: fund_inflow_list.index(x))
                print(len(fund_inflow_list), "交集后股数")
                # print(fund_inflow_list)
            is_write_stock('fund_inflow.blk', fund_inflow_list, "write")
        elif choice == "ths_fund_inflow0":
            is_write_stock('ths_fund_inflow0.blk', fund_inflow_list, "write")
        return str(len(fund_inflow_list))
    return ""


# 需要改cookie
def ths_lgt(date):
    headers = {
        "Cookie": "PHPSESSID=4ea026e44bb06c30e0a0dafdf5e0dfd1; other_uid=Ths_iwencai_Xuangu_9j2forruvxyo7a62uswxpyk2j2z2su4r; cid=3334c28eb211f54e6885d80b6b2f051a1609357041; user_status=0; Hm_lvt_78c58f01938e4d85eaf619eae71b4ed1=1615319471; cid=3334c28eb211f54e6885d80b6b2f051a1609357041; ComputerID=3334c28eb211f54e6885d80b6b2f051a1609357041; WafStatus=0; Hm_lpvt_78c58f01938e4d85eaf619eae71b4ed1=1617115747; v=A1YU4TH-Kl4n8x76z21Nvq4kpwdL95hI7DjOlcC9Q-vUm_izKIfqQbzLHrCT",
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36"
    }
    ths_data = {
        'question': date + '陆股通净买入额大于600万',
        'perpage': '500',
        'page': '1',
        'log_info': '{"input_type":"typewrite"}',
        'source': 'Ths_iwencai_Xuangu',
        'version': '2.0',
        'add_info': '{"urp":{"scene": 1, "company": 1, "business": 1}, "contentType":"json"}',
    }
    lgt = requests.post("http://x.10jqka.com.cn/unifiedwap/unified-wap/v2/result/get-robot-data", headers=headers, data=ths_data)
    print(lgt)
    if lgt.status_code == 200:
        lgt_data = lgt.json().get('data', '').get('answer', '')[0].get('txt', '')[0].get('content', '') \
            .get('components', '')[0].get('data', '').get('datas', '')
        # print(lgt_data)
        lgt_list = []
        for item in lgt_data:
            # lgt_list.append('1' + item.get('code', '')+'\n')
            code = code_add(item.get('code', ''))
            lgt_list.append(code + '\n')
        print(lgt_list)
        is_write_stock('lgt.blk', lgt_list, "write")
        return len(lgt_list)


# 同花顺公告利好
def ths_notice_good(date):
        cookie = get_cookie()
        headers = {
            "Cookie": "chat_bot_session_id=7bcebb62fb52a2de11bf41ec05073886; "
                      "other_uid=Ths_iwencai_Xuangu_dz8du2owtuaf0fcmkdy2h8ewto46jpsh; "
                      "cid=b39c75faa8d86f5eee3a5502e26c76f81617695835; "
                      "v=" + cookie,
            "Connection": "keep-alive",
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36"
        }
        ths_data = {
            'question': date + '公告利好个股且' + date + '涨幅大于4',
            'perpage': '100',
            'page': '1',
            'log_info': '{"input_type":"typewrite"}',
            'source': 'Ths_iwencai_Xuangu',
            'version': '2.0',
            'add_info': '{"urp":{"scene": 1, "company": 1, "business": 1}, "contentType":"json"}',
        }
        lgt = requests.post("http://x.10jqka.com.cn/unifiedwap/unified-wap/v2/result/get-robot-data", headers=headers,
                            data=ths_data)
        print(lgt)
        if lgt.status_code == 200:
            lgt_data = lgt.json().get('data', '').get('answer', '')[0].get('txt', '')[0].get('content', '') \
                .get('components', '')[0].get('data', '').get('datas', '')
            # print(lgt_data)
            lgt_list = []
            for item in lgt_data:
                code = code_add(item.get('code', ''))
                lgt_list.append(code + '\n')
            lgt_list = sorted(set(lgt_list), key=lgt_list.index)
            # print(lgt_list)
            # print(len(lgt_list))
            is_write_stock('ths_notice_good.blk', lgt_list, "write")
            combine_stock = read_file(r"D:\myzq\axzq\T0002\blocknew\combine.blk")
            tmp = list(set(lgt_list).intersection(set(combine_stock)))
            print(len(tmp), "交集后")
            is_write_stock('ths_good_contraction.blk', tmp, "write")
            return str(len(lgt_list)) + ";" + str(len(tmp))
        return ""


# 同花顺涨停或大于5
def ths_rise(date, up_rise):
        cookie = get_cookie()
        headers = {
            "Cookie": "chat_bot_session_id=7bcebb62fb52a2de11bf41ec05073886; "
                      "other_uid=Ths_iwencai_Xuangu_dz8du2owtuaf0fcmkdy2h8ewto46jpsh; "
                      "cid=b39c75faa8d86f5eee3a5502e26c76f81617695835; "
                      "v=" + cookie,
            "Connection": "keep-alive",
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36"
        }
        if up_rise == "5":
            strong = date + '涨幅大于5个股且' + date + '非涨停'
        else:
            strong = date + '涨停个股'
        print(strong)
        ths_data = {
            'question': strong,
            'perpage': '400',
            'page': '1',
            'log_info': '{"input_type":"typewrite"}',
            'source': 'Ths_iwencai_Xuangu',
            'version': '2.0',
            'add_info': '{"urp":{"scene": 1, "company": 1, "business": 1}, "contentType":"json"}',
        }
        lgt = requests.post("http://x.10jqka.com.cn/unifiedwap/unified-wap/v2/result/get-robot-data", headers=headers,
                            data=ths_data)
        print(lgt)
        if lgt.status_code == 200:
            lgt_data = lgt.json().get('data', '').get('answer', '')[0].get('txt', '')[0].get('content', '') \
                .get('components', '')[0].get('data', '').get('datas', '')
            # print(lgt_data)
            lgt_list = []
            for item in lgt_data:
                code = code_add(item.get('code', ''))
                lgt_list.append(code + '\n')
            # lgt_list = sorted(set(lgt_list), key=lgt_list.index)
            # print(lgt_list)
            # print(len(lgt_list))
            if up_rise == "5":
                is_write_stock('ths_rise5.blk', lgt_list, "write")
            else:
                is_write_stock('ths_rise10.blk', lgt_list, "write")
            return len(lgt_list)
        return ""


# 同花顺选股 1为写入通达信
def ths_choice(ths_in, t="1", f="ths_choice.blk"):
    cookie = get_cookie()
    headers = {
        "Cookie": "chat_bot_session_id=7bcebb62fb52a2de11bf41ec05073886; "
                  "other_uid=Ths_iwencai_Xuangu_dz8du2owtuaf0fcmkdy2h8ewto46jpsh; "
                  "cid=b39c75faa8d86f5eee3a5502e26c76f81617695835; "
                  "v=" + cookie,
        "Connection": "keep-alive",
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36"
    }
    ths_data = {
        'question': ths_in,
        'perpage': '1500',
        'page': '1',
        'log_info': '{"input_type":"typewrite"}',
        'source': 'Ths_iwencai_Xuangu',
        'version': '2.0',
        'add_info': '{"urp":{"scene": 1, "company": 1, "business": 1}, "contentType":"json"}',
    }
    lgt = requests.post("http://x.10jqka.com.cn/unifiedwap/unified-wap/v2/result/get-robot-data", headers=headers,
                        data=ths_data)
    if lgt.status_code == 200:
        lgt_data = lgt.json().get('data', '').get('answer', '')[0].get('txt', '')[0].get('content', '') \
            .get('components', '')[0].get('data', '').get('datas', '')
        # print(lgt_data)
        if lgt_data:
            lgt_list = []
            for item in lgt_data:
                if t == "1":
                    lgt_list.append(code_add(item.get('code', '')) + '\n')
                else:
                    lgt_list.append(item.get('code', ''))
            if t == "1":
                is_write_stock(f, lgt_list, "write")
            return lgt_list
    return ""


# 显示choice板块个股,加雪球自选
def read_choice(file, xue_qiu="xue_qiu"):
    stock_list = is_write_stock(file, "", "read")
    # print(stock_list)
    if stock_list:
        if stock_list[0] == "\n" or stock_list[0] == "":
            del stock_list[0]
        # 登陆系统
        lg = bs.login()
        # 显示登陆返回信息
        # print('login respond error_code:' + lg.error_code)
        # print('login respond  error_msg:' + lg.error_msg)
        for i, value in enumerate(stock_list):
            stock_code = value.strip()[1:]
            if value.startswith('0'):
                code = "sz." + stock_code
            elif value.startswith('1'):
                code = "sh." + stock_code
            else:
                pass
            # print(code)
            rs = bs.query_stock_basic(code=code)
            # print('query_stock_basic respond error_code:' + rs.error_code)
            # print('query_stock_basic respond  error_msg:' + rs.error_msg)
            result = rs.get_row_data()
            # print("tyr", result)
            if result:
                stock_list[i] = stock_code + result[1]
            else:
                print("没有数据新股？", code)
                stock_list[i] = stock_code + "没有数据新股？"
            # add雪球自选
            if xue_qiu == "xue_qiu":
                add_xue_qiu(stock_code)
        # print(stock_list)
        # 登出系统
        bs.logout()
        return stock_list
        # return ""


# 读choice写自选
def write_self_hai_tong():
    choice = read_file("choice.blk")
    choice += read_file("choice_sell.blk")
    zxg = read_file("ZXG.blk")
    # 备份安信自选
    if zxg:
        p = is_not_path("ZXG_COPY.blk")
        with open(p, "w") as f:  # 自动关闭
            f.writelines(zxg)
    # ht_zxg = read_file("ZXG.blk")
    # 备份海通自选
    # if ht_zxg:
    #     with open(r"D:\myzq\haitong\T0002\blocknew\ZXG_COPY.blk", "w") as f:  # 自动关闭
    #         f.writelines(ht_zxg)
    # print(choice)
    # print(ht_zxg)
        if choice:
            choice = add_linefeed(choice)
            print(choice[::-1])
            for each in choice[::-1]:
                if each not in zxg:
                    if len(zxg) > 11:
                        zxg.insert(11, each)
                    else:
                        zxg.append(each)
            # for each in choice:
            #     if each not in ht_zxg:
            #         ht_zxg.append(each)
            zxg = add_linefeed(zxg)
            # ht_zxg = add_linefeed(ht_zxg)
            # print(zxg)
            #  写安信自选
            pp = is_not_path("ZXG.blk")
            with open(pp, "w") as f:  # 自动关闭
                f.writelines(zxg)
            # 写海通自选
            # ht_zxg = add_linefeed(ht_zxg)
            # # print(ht_zxg)
            # with open(r"D:\myzq\haitong\T0002\blocknew\ZXG.blk", "w") as f:  # 自动关闭
            #     f.writelines(ht_zxg)


# 登录海通
def log_on_ht():
    pid = judge_process("xiadan.exe")
    path_folder = r"D:\myzq\thwt\xiadan.exe"
    # print(pid)
    if pid:
        app = pywinauto.Application("uia").connect(process=pid)
        ap = app.window(best_match=u"用户登录")

        if ap.exists():
            ap.close()
            sleep(5)
            new_app = pywinauto.Application(backend="uia").start(path_folder)
            new_app.wait("ready", timeout=15, retry_interval=2)
            # sleep(5)
            new_ap = new_app.window(best_match=u"用户登录")
            new_ap.wait("ready", timeout=15, retry_interval=2)
            # sleep(10)
            if new_ap.exists():
                new_ap.children()[1].type_keys("282766")
                sleep(1)
                new_ap.children()[2].type_keys("282766")
                sleep(1)
                new_ap.children()[13].click().wait("ready")
                sleep(2)
                trade = new_app.window(best_match=u"网上股票交易系统")
                trade.wait("ready", timeout=15, retry_interval=2)
                # sleep(4)
                # exists = trade.exists()
                # print(exists)
                if trade.exists():
                    return trade
                else:
                    new_ap.close()
                    print("无法登录，请检查网络，交易时间或者交易界面设置是否有误")
                    return False
            else:
                print("无法打开程序，请检查程序路径是否正确")
        else:
            app.top_window().set_focus()
            ap = app.window(best_match=u"网上股票交易系统")
            ap.wait("ready", timeout=15, retry_interval=2)
            if ap:
                return ap
            else:
                print("无法打开程序，请检查程序是否正确")
                return False
    else:
        app = pywinauto.Application(backend="uia").start(path_folder)
        sleep(3)
        ap = app.window(best_match=u"用户登录")
        ap.wait("ready", timeout=15, retry_interval=2)
        sleep(4)
        # is_no = ap.exists()
        # print(is_no)
        if ap.exists():
            # ap.children()[1].draw_outline(colour='red', thickness=5)
            # pass
            sleep(1)
            ap.children()[1].type_keys("282766")
            sleep(2)
            ap.children()[2].type_keys("282766")
            sleep(2)
            ap.children()[13].click()
            sleep(5)
            pid = judge_process("xiadan.exe")
            if pid:
                connect = pywinauto.Application("uia").connect(process=pid)
                trade = connect.window(best_match=u"网上股票交易系统")
                trade.wait("ready", timeout=10, retry_interval=2)
                # sleep(3)
                # exists = trade.exists()
                # print(exists)
                if trade.exists():
                    # trade.draw_outline(colour='red', thickness=5)
                    # print(trade)
                    # pass
                    return trade
                else:
                    ap.close()
                    print("无法登录，请检查网络，交易时间或者交易界面设置是否有误")
                    return False
            else:
                ap.close()
                print("无法登录，请检查网络，交易时间或者交易界面设置是否有误")
                return False
        else:
            print("无法打开程序，请检查程序路径是否正确")
            return False


# 预埋单 t="interface"从交易界面输入，backstage从后台数据库输入
def pre_paid(stock_dict, dialog="", t="interface"):
    # print(dialog)
    if t == "interface":
        dialog.window(best_match="查询[F4]", auto_id="", class_name="", control_type="TreeItem").click_input(button='left', double=True)
        sleep(1)
        dialog.window(best_match="预设单", auto_id="", class_name="", control_type="TreeItem").set_focus()
        for key, value in stock_dict.items():
            # print(key + ':' + value)
            dialog.window(best_match="", auto_id="1032", class_name="Edit", control_type="Edit").set_text(key)
            sleep(0.5)
            dialog.window(auto_id="1033", class_name="Edit", control_type="Edit").set_text(value)
            sleep(0.5)
            dialog.window(auto_id="1034", class_name="Edit", control_type="Edit").set_text("100")
            sleep(0.5)
            dialog.window(best_match="添加[A]", auto_id="1006", class_name="Button", control_type="Button").click()
            sleep(0.5)
            dialog.window(best_match="重填[R]", auto_id="1007", class_name="Button", control_type="Button").click()
            sleep(0.5)
    if t == "backstage":
        with sqlite3.connect(is_not_path("data/ymddata.db", path_list=mysetting.JY_URL, flag="3")) as conn:
            conn.text_factory = lambda x: str(x, 'gbk', 'ignore')
            cu = conn.cursor()
            cu.execute("delete FROM ymd_1280194006")
            for t in stock_dict:
                cu.execute(
                    "INSERT INTO ymd_1280194006 (xd_2102,xd_2103,xd_2106,xd_2109,xd_2127,xd_2126,xd_2108,xd_2105,xd_3630) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        t["xd_2102"],
                        t["xd_2103"].encode(encoding='gbk'),
                        t["xd_2106"],
                        t["xd_2109"].encode(encoding='gbk'),
                        t["xd_2127"],
                        t["xd_2126"],
                        t["xd_2108"].encode(encoding='gbk'),
                        t["xd_2105"],
                        t["xd_3630"],
                    )
                    )
            # data = cu.execute("SELECT * FROM ymd_1280194006").fetchall()
            # # print(len(data))
            # if data:
            #     cookie_xq = ""
            #     for result in data:
            #         print(result)
            #         if result:
            #             pass


# 查询当天收盘价并构建数据给交易软件。5.30后  f == "2"查询收盘价和名字
def inquiry_close(stock_list, date, buy="买入", f="1"):
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    # print('login respond error_code:' + lg.error_code)
    if f == "1":
        stock_dict = {}
        if lg.error_code == "0":
            for item in stock_list:
                # print(item)
                rs = bs.query_history_k_data_plus(item, "close,", start_date=date, end_date=date,
                                                  frequency="d", adjustflag="3")
                # print('query_history_k_data_plus respond error_code:' + rs.error_code)
                while (rs.error_code == '0') & rs.next():
                    # 获取一条记录，将记录合并在一起
                    result = rs.get_row_data()
                    # print(result)
                    if len(result):
                        if result[0] != "":
                            stock_dict[item[3:]] = result[0]
                            # print(result[0])
                        else:
                            print("有误，空", item)
                    else:
                        print("有误", item)
        # 登出系统
        bs.logout()
        print(stock_dict)
        return stock_dict
    if f == "2":
        stock = []
        if lg.error_code == "0":
            for item in stock_list:
                if item.startswith("sz"):
                    xd_2106 = mysetting.HOLDER_CODE[0][0]
                    xd_2108 = mysetting.HOLDER_CODE[0][1]
                elif item.startswith("sh"):
                    xd_2106 = mysetting.HOLDER_CODE[1][0]
                    xd_2108 = mysetting.HOLDER_CODE[1][1]
                else:
                    print(item, "error")
                    xd_2106 = ""
                st = {}
                # print(item)
                rs = bs.query_history_k_data_plus(item, "close,", start_date=date, end_date=date,
                                                  frequency="d", adjustflag="3")
                if not rs.next():
                    print("无数据，检查输入是否为未上市新股或日期（每天收盘5点后）或代码格式（sz,000001,sh.600000）")
                # print('query_history_k_data_plus respond error_code:' + rs.error_code)
                while (rs.error_code == '0') & rs.next():
                    # 获取一条记录，将记录合并在一起
                    result = rs.get_row_data()
                    # print(result)
                    if len(result):
                        # print(result[0])
                        if result[0] != "":
                            st["xd_2102"] = item[3:]
                            st["xd_2127"] = result[0]
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
                            if xd_2106 != "":
                                st["xd_2103"] = result[1]
                                st["xd_2106"] = xd_2106
                                st["xd_2108"] = xd_2108
                                st["xd_2109"] = buy
                                st["xd_2126"] = "100"
                                st["xd_2105"] = ""
                                st["xd_3630"] = "0.000000"
                        else:
                            print("有误，空", item)
                    else:
                        print("有误", item)

                if len(st) == 9:
                    stock.append(st)
        # 登出系统
        bs.logout()
        # print(stock)
        return stock

driver = ""


# 启动360浏览器
def selenium_open():
    global driver
    chrome_options = webdriver.ChromeOptions()
    user_data_dir = r'--user-data-dir=D:\mysoft\360chrome\chrome\User Data\Default'
    chrome_options.add_argument(user_data_dir)
    chrome_options.binary_location = r"D:\mySoft\360Chrome\Chrome\Application\360chrome.exe"  # 这里是360安全浏览器的路径
    # chrome_options.add_argument(r'--lang=zh-CN')  # 这里添加一些启动的参数
    # 设置好应用扩展
    extension_path = r'D:\mySoft\360Chrome\Chrome\User Data\Default\Extensions\eimadpbcbfnmbkopoojfekhnkhdbieeh\4.9.26_0.crx'
    chrome_options.add_extension(extension_path)
    driver = webdriver.Chrome(chrome_options=chrome_options)
    return driver
    # driver.quit()


# 点击雪球小标题
def open_xue_qiu_able(dri, xue_code, num):
    dri.execute_script('window.open("https://xueqiu.com/S/%s");' % xue_code)
    dri.switch_to.window(driver.window_handles[-1])  # 切换到最新页面
    try:
        WebDriverWait(dri, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "stock-timeline-tabs")))
        sleep(0.5)
        dri.find_element_by_css_selector("div.stock-timeline-tabs>a:nth-child({})".format(num)).click()
        sleep(0.5)
    except RuntimeError:
        print("out")


#  判断程序是否已经运行
def judge_process(process):
    pl = psutil.pids()
    for pid in pl:
        if psutil.Process(pid).name() == process:
            print(pid)
            return pid
    else:
        print("not found")
        return ""


# 给list stock加换行符
def add_linefeed(data):
    for i, value in enumerate(data):
        if value[-1] != "\n":
            data[i] = value + "\n"
    return data


# 读取choice板块，获取code
def read_choice_code(file):
    stock_list = read_file(file)
    # print(stock_list)
    if len(stock_list):
        for i, value in enumerate(stock_list):
            v = value.strip()[1:]
            if v.startswith("6"):
                v = "sh." + v
            else:
                v = "sz." + v
            stock_list[i] = v
    # print(stock_list)
    return stock_list


# 简化版读文件
def read_file(file, f="0"):
    if f == "0":
        file = is_not_path(file)
        print(file)
    with open(file) as f:  # 自动关闭
        stock_list = f.readlines()
        # print(stock_list)
    if stock_list:
        for i, value in enumerate(stock_list):
            if value == "\n":
                stock_list.remove("\n")
            elif value == "":
                stock_list.remove("")
        return stock_list
    return []


# 读龙虎榜自定义个股
def read_dragon(file):
    stock_list = is_write_stock(file, "", "read")
    return update_list(stock_list)


# 修改列表元素
def update_list(stock_list):
    if stock_list:
        if stock_list[0] == "\n" or stock_list[0] == "":
            del stock_list[0]
        for i, value in enumerate(stock_list):
            stock_list[i] = value.strip()[1:]
    # print(stock_list)
    return stock_list


# add雪球自选
def add_xue_qiu(stock_code):
    base_url = "https://xueqiu.com/stock/search.json"
    params = {
        'code': stock_code,
        'size': '10'
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.67 Safari/537.36',
        'Cookie': get_cookie(wet="xueqiu.com")
    }
    # print("ghj", stock_code)
    res = requests.get(base_url, params=params, headers=headers)
    res.encoding = 'utf-8'
    # print('wqe', res.json())
    stocks = res.json().get('stocks', '')
    # print(stocks)
    if stocks:
        has_exist = stocks[0].get('hasexist', 'no')
        # print(has_exist, 'uiyoiw')
        if has_exist == "false":
            post_code = stocks[0].get('code', '')
            # print(post_code)
            post_url = " https://stock.xueqiu.com/v5/stock/portfolio/stock/add.json"
            data = {
                'symbols': post_code,
                'category': '1'
            }
            response = requests.post(post_url, params=data, headers=headers)
            # print("Status code:", response.status_code)
    else:
        print("cookie可能有误")


# code加(0 or 1) or (1 or 0) or (1.  0.)
def code_add(code, param=""):
    if param == "":
        if code.startswith("0") or code.startswith("3") or code.startswith("2"):
            code = "0" + code
        elif code.startswith("5") or code.startswith("6") or code.startswith("9"):
            code = "1" + code
        else:
            code = "0" + code
    elif param == "param":
        if code.startswith("0") or code.startswith("3") or code.startswith("2"):
            code = "1" + code
        elif code.startswith("5") or code.startswith("6") or code.startswith("9"):
            code = "0" + code
        else:
            code = "0" + code
    elif param == "1.":
        if code.startswith("0") or code.startswith("3") or code.startswith("2"):
            code = "0." + code
        elif code.startswith("5") or code.startswith("6") or code.startswith("9"):
            code = "1." + code
        else:
            code = "0" + code
    return code


# code加(sh or sz) or (SZ or SH)
def add_sh(code, big=""):
    if big == "":
        if code.startswith("0") or code.startswith("3") or code.startswith("2"):
            code = "sz" + code
        elif code.startswith("5") or code.startswith("6") or code.startswith("9"):
            code = "sh" + code
        else:
            code = "sz" + code
    else:
        if code.startswith("0") or code.startswith("3") or code.startswith("2"):
            code = "SZ" + code
        elif code.startswith("5") or code.startswith("6") or code.startswith("9"):
            code = "SH" + code
        else:
            code = "SZ" + code
    return code


# 获取chrome同花顺和雪球cookie
def get_cookie(wet='.10jqka.com.cn'):
    username = os.environ.get('USERNAME')
    cookie_file = "C:\{name}\{UserName}\AppData\Local\Google\Chrome\{use}\Default\Cookies"\
        .format(name='Users', UserName=username, use='User Data')
    # print(cookie_file)
    sql = "SELECT * FROM cookies WHERE host_key like '%{}%';".format(wet)
    # print(sql)
    with sqlite3.connect(cookie_file) as conn:
        cu = conn.cursor()
        data = cu.execute(sql).fetchall()
        # print(len(data))
        if data:
            cookie_xq = ""
            for result in data:
                # print(result)
                if result:
                    if wet == '.10jqka.com.cn':
                        try:
                            if result[2] == 'v':
                                # print(result[12])
                                password = win32crypt.CryptUnprotectData(result[12], None, None, None, 0)
                                # print("rer", password)
                                print(password[1].decode())
                                return password[1].decode()
                        except Exception as e:
                            print('[-] %s' % e)
                            pass
                    else:
                        if result[12]:
                            p = win32crypt.CryptUnprotectData(result[12], None)[1].decode()
                        else:
                            p = result[12]
                        if result[2]:
                            cookie_xq += result[2] + "=" + p + ";"
            # pprint(cookie_xq)
            return cookie_xq
    print("获取cookie失败")
    return ""


# 打开google chrome 同花顺网站和雪球
def open_chrome(process_name="chrome.exe", url="http://x.10jqka.com.cn/unifiedwap/result?w=20210428%E5%85%AC%E5%91%8A%E5%88%A9%E5%A5%BD%E4%B8%9420210428%E6%B6%A8%E5%B9%85%E5%A4%A7%E4%BA%8E5&querytype=&issugs"):
    if judge_process(process_name):
        print("ready open")
    else:
        import webbrowser
        path = is_not_path(r"D:\mySoft\chrome\Google\Chrome\Application\chrome.exe", flag="2")
        # print(path)
        webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(path))
        webbrowser.get('chrome').open(url)
        sleep(65)


def is_write_stock(file, data, is_write):
    path = is_not_path(file)
    if is_write == "write":
        # print("djgla")
        with open(path, "w") as f:  # 自动关闭
            f.writelines(data)
    elif is_write == "read":
        # print("read...." + file)
        with open(path) as f:  # 自动关闭
            # stock_list = f.readlines()
            # splitlines() 按照行('\r', '\r\n', \n')分隔，返回一个包含各行作为元素的列表，如果参数 keepends 为默认 False，不包含换行符，如果为 True，则保留换行符
            stock_list = f.read().splitlines()
            # print(stock_list)
            if stock_list:
                for i, value in enumerate(stock_list):
                    if value == "\n":
                        stock_list.remove("\n")
                    elif value == "":
                        stock_list.remove("")
            # print(stock_list)
            return stock_list


# 判断路径是否存在
def is_not_path(file, path_list="", flag="1"):
    if flag == "1":
        for u in mysetting.ZQ_URL:
            if os.access(u + file, os.F_OK):
                # print(u + file)
                return u + file
    elif flag == "2":
        if os.access(file, os.F_OK):
            path = file
        else:
            path = r"D:\mysoft\chrome\Application\chrome.exe"
            # path = r"C:\Users\Administrator\AppData\Local\Google\Chrome\Application\chrome.exe"
        return path
    elif flag == "3":
        for u in path_list:
            if os.access(u + file, os.F_OK):
                # print(u + file)
                return u + file


def index_django(request):
    book_list = my_custom_sql(request.GET.dict())
    return JsonResponse(book_list)


def search_integration(request):
    search = request.GET.get("search", "no")
    sort = request.GET.get("sort", "no")
    print(sort)
    if search == "no" or search == '':
        return render(request, 'datatables/search_integration.html')
    else:
        option = webdriver.ChromeOptions()
        # 关闭“chrome正受到自动测试软件的控制”
        # V75以及以下版本 貌似都能用
        option.add_argument('disable-infobars')
        # V76以及以上版本
        # option.add_experimental_option('useAutomationExtension', False)
        # option.add_experimental_option('excludeSwitches', ['enable-automation'])
        # 不自动关闭浏览器
        option.add_experimental_option("detach", True)
        # 打开chrome浏览器
        driver = webdriver.Chrome(chrome_options=option)
        driver.get('http://www.baidu.com')
        wait = WebDriverWait(driver, 10)
        ele = wait.until(expected_conditions.element_to_be_clickable((By.ID, 'kw')))
        # ele = driver.find_element_by_id('kw')  # 查找元素
        ele.clear()
        ele.send_keys(search)
        ele.send_keys(Keys.RETURN)
        return HttpResponse()


def shgtDf2021():
    data = ShgtDf2021.objects.all()[:200].values()
    total_length = ShgtDf2021.objects.all()[:200].count()
    print(total_length)

    # re = json.dumps(list(aa), ensure_ascii=False)
    re = json.dumps({"data": list(data), "iTotalDisplayRecords": total_length}, ensure_ascii=False)
    # print(re)
    # print(getType(re))
    return mark_safe(re)


def queryset_json(query, parameter, para):  # 默认为return list and not mark_safe
    data = query.values()
    # total_length = ShgtDf2021.objects.all()[:200].count()
    # print(data)
    if parameter == 'dict_json':
        re = json.dumps({"data": list(data)}, ensure_ascii=False)
    elif parameter == 'list_json':  # parameter=2不加data
        re = json.dumps(list(data), ensure_ascii=False)
    elif parameter == 'dict':  # parameter=3返回字典，不json化
        re = {"data": list(data)}
    else:
        re = list(data)
    if para == 'mark_safe':
        return mark_safe(re)
    else:
        return re
TOTAL = ''


def my_custom_sql(request):
    field_no_list = ["code", "name", "trade"]
    offset = request.get('offset', '0')
    # print(offset)
    limit = request.get('limit')
    # print(limit)
    sort = request.get('sort', 'code')
    # print(sort not in field_no_list)
    if sort not in field_no_list:  # 如果sort没有在list里的要变为数字
        sort_sql = "order by cast(%s as '9999')" % sort
    else:
        sort_sql = "order by %s " % sort
    # print(sort_sql)
    order = request.get('order', 'asc')
    # print(order)
    search = request.get('search', '')
    # search = request.get('search', '收盘价10-30；涨幅>1.2；')
    print(search)
    where_sql = ""
    # print(search != '')
    if search != '':
        field_list = {
            "编号": "code",
            "名称": "name",
            "收盘价": "todayClosePrice",
            "涨幅": "todayUp",
            "持股量": "todayQuantity",
            "持股市值": "todayValue",
            "占流通比": "circulateRate",
            "占总比": "totalRate",
            "增加股数": "addNumber",
            "增加市值": "addValue",
            "增加百分比": "addValueRate",
            "增加占流通比": "addValueRateCirculate",
            "增加占总比": "addValueRateTotal",
            "行业": "trade",
            "日期": "date",
        }
        if ";" in search:  # 半角分号换成全角
            search.replace(";", "；")
        if not search.endswith('；'):  # 结尾没有分号则加上
            search += "；"
            print(search)
        # print(search)
        i = 0
        for key in field_list.keys():
            # print(key)
            result = key in search
            # print(result)
            if result:
                i += 1
                # i = i + 1
                # print(i)
                field_value = field_list.get(key, "no")  # 获取code
                # print(field_value not in field_no_list)
                take_string = re.findall(r"%s(.+?)；" % key, search)[0]  # 提取“长江”和其后的“人”之间的字符，返回一个列表
                if field_value not in field_no_list:  # 如果没有在list里的要变为数字
                    field_value = "cast(%s as '9999')" % field_value
                    # print(field_value)
                    if "-" in take_string:  # 判断是否查询范围
                        if "=" in take_string:
                            take_string = take_string.replace("=", "")  # 查询范围的等号要去除
                        take_string = " BETWEEN %s" % (take_string.replace("-", " and "))
                else:
                    if "=" in take_string:
                        take_string = " = '%s'" % take_string.replace("=", "")
                # print(take_string)
                if i == 1:
                    where_sql += "WHERE %s %s " % (field_value, take_string)
                else:
                    where_sql += " and %s %s " % (field_value, take_string)
    print(where_sql)

    with connection.cursor() as c:  # 等价于try和自动close链接
        if offset == '0' and sort == 'code':
            c.execute("select count(*) from shgt_df_2021 %s" % where_sql)
            global TOTAL
            TOTAL = c.fetchone()[0]
            print(TOTAL)

        sql = "select * from shgt_df_2021 %s %s %s LIMIT %s, %s" \
              % (where_sql, sort_sql, order, offset, limit)
        print(sql)
        c.execute(sql)
        row = c.fetchall()
        # row = cursor.fetchone()
        # cursor.rowcount  # 属性指出上次查询或更新所发生行数。-1表示还没开始查询或没有查询到数据。
        # 构建字典
        col_names = [desc[0] for desc in c.description]
        # print(col_names)
        res = []
        for row_list in row:
            row_dict = dict(zip(col_names, row_list))
            res.append(row_dict)
        # print(res)
    return {'rows': res, 'total': TOTAL}


def dict_json(query, parameter=1):  # parameter=1加data
    if parameter == 1:
        re = json.dumps({"data": query}, ensure_ascii=False)
    elif parameter == 2:  # parameter=2不加data
        re = json.dumps(query, ensure_ascii=False)
    else:
        print('错误的参数')
    # return re
    return mark_safe(re)


# 判断变量类型的函数
def type_of(variate):
    type = None
    if isinstance(variate, int):
        type = "int"
    elif isinstance(variate, str):
        type = "str"
    elif isinstance(variate, float):
        type = "float"
    elif isinstance(variate, list):
        type = "list"
    elif isinstance(variate, tuple):
        type = "tuple"
    elif isinstance(variate, dict):
        type = "dict"
    elif isinstance(variate, set):
        type = "set"
    return type


# 返回变量类型
def get_type(variate):
    arr = {"int": "整数", "float": "浮点", "str": "字符串", "list": "列表", "tuple": "元组", "dict": "字典", "set": "集合"}
    var_type = type_of(variate)
    print(var_type)
    if not (var_type in arr):
        return "未知类型"
    return arr[var_type]


# 将class转dict,以_开头的属性不要
def props(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        if not name.startswith('__') and not callable(value) and not name.startswith('_'):
            pr[name] = value
    return pr


# 将class转dict,以_开头的也要
def props_with_(obj):
    pr = {}
    for name in dir(obj):
        value = getattr(obj, name)
        pr[name] = value
    return pr


# dict转obj，先初始化一个obj
def dict2obj(obj, dict):
    obj.__dict__.update(dict)
    return obj


# 判断是否可以float
def isfloat(value):
    try:
        return float(value)
    except ValueError:
        print("转换float失败" + value)
        return 0


# 判断时间范围
def judge_time_scope(now, start_time, end_time):
    # 范围时间
    start_time = datetime.strptime(str(now.date()) + start_time, '%Y-%m-%d%H:%M')
    end_time = datetime.strptime(str(now.date()) + end_time, '%Y-%m-%d%H:%M')
    # 判断当前时间是否在范围时间内
    if (now > start_time) and now < end_time:
        return True
    else:
        return False


# 判断是否交易时间范围
def judge_trade_time(i_date):
    print(i_date)
    # 判断是否为节假日,周六日
    if (i_date.isoweekday() == (6 or 7)) or is_holiday(d_date(i_date.year, i_date.month, i_date.day)):
        return "节假日"
    else:
        if judge_time_scope(i_date, "9:25", "15:00"):
            return "交易时段"
        else:
            return "交易日但非交易时段"


# 复制文件
def copy_file(source, target):
    try:
        shutil.copyfile(source, target)
        return 1
    except IOError as e:
        print("Unable to copy file. %s" % e)
        sys.exit(1)
    except:
        print("Unexpected error:", sys.exc_info())
        sys.exit(1)
    print("复制文件失败")