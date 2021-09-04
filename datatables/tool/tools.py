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


# 压缩和备份
def create_zip(file_path, save_path, note=""):
    import zipfile
    '''
    将文件夹下的文件保存到zip文件中。
    :param filePath: 待备份文件
    :param savePath: 备份路径
    :param note: 备份文件说明
    :return:
    '''
    file_list = []
    if note:
        target = save_path + os.sep + note + '.zip'
    else:
        target = save_path + os.sep + time.strftime('%Y%m%d') + "_" + time.strftime('%H%M%S') + '.zip'
    # print(target)
    new_zip = zipfile.ZipFile(target, 'w')
    for dir_path, dir_names, file_names in os.walk(file_path):
        for file_name in file_names:
            file_list.append(os.path.join(dir_path, file_name))
    # pprint(file_list)
    for tar in file_list:
        new_zip.write(tar, tar[len(file_path):])  # tar为写入的文件，tar[len(filePath)]为保存的文件名
    new_zip.close()

