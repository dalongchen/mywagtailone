import numpy as np
import pandas as pd
import sqlite3
import global_variable as gl_v


# 获取可转债正股股价
class KzzStockPrice(object):
    def __init__(self):
        pass

    # baostock 输入代码和时间段获取股票k线数据. adjustflag(1：后复权， 2：前复权，3：不复权）end_date=''为最新日期
    def history_k_data(self, code="sh.000001", start_date='', end_date='', frequency="d", adjustflag="2", col="all", y=1):
        import baostock as bs
        lg = bs.login()

        if start_date == '' or end_date == '':
            import datetime
            now_time = datetime.datetime.now()  # 获取当前时间
            if end_date == '':
                end_date = now_time.strftime('%Y-%m-%d')
                # print("now time: ", )
            if start_date == '':
                # 获取前一天时间
                end_time = now_time + datetime.timedelta(days=-365*y)
                # 前一天时间只保留 年-月-日
                start_date = end_time.strftime('%Y-%m-%d')  # 格式化输出
                # print("end date: ", end_date)
        if (not code.startswith("sh.")) and (not code.startswith("sz.")):
            # print("ghjppp", code)
            if code.startswith("SH.") or code.startswith("SZ."):
                code = code.lower()
            else:
                # print("ghjppkkkkkkp", code)
                code = self.add_sh(code, big="baostock")
        # 显示登陆返回信息
        # print('login respond error_code:' + lg.error_code)
        # print('login respond  error_msg:' + lg.error_msg)
        # “分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
        # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
        # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
        if col == "all":
            str_col = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
        if col == "only_up_rate":
            str_col = "date,close,pctChg"
        # print("ghj", code)
        rs = bs.query_history_k_data_plus(code, str_col, start_date, end_date, frequency, adjustflag)
        # print('query_history_k_data_plus respond error_code:' + rs.error_code)
        # print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        bs.logout()
        if col == "all":
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
            # result.to_csv("D:\\history_A_stock_k_data.csv", index=False)
        if col == "only_up_rate":
            import sqlite3
            db_path = r"D:\myzq\axzq\T0002\stock_load\thesis\kzz\monte.db"  # 数据库路径
            result = pd.DataFrame(data_list, columns=rs.fields, dtype='float').round(3)
            # 对数收益率 = log(收盘价/前一个收盘价)
            result['log_'] = (np.log(result['close'] / result['close'].shift(periods=1, axis=0))*100).round(3)
            # print(result[1:])
            with sqlite3.connect(db_path) as conn:
                result[1:].to_sql(code, con=conn, if_exists='replace', index=False)
            # print(data_list[-5:])
            # return data_list

    @staticmethod
    def add_sh(code, big=""):  # big="baostock"加(sh. or sz.)code加(sh or sz) or (SZ or SH)
        if big == "":
            if code.startswith("0") or code.startswith("3") or code.startswith("2"):
                code = "sz" + code
            elif code.startswith("5") or code.startswith("6") or code.startswith("9"):
                code = "sh" + code
            else:
                print("err1", code)
        elif big == "baostock":
            if code.startswith("0") or code.startswith("3") or code.startswith("2"):
                code = "sz." + code
            elif code.startswith("5") or code.startswith("6") or code.startswith("9"):
                code = "sh." + code
            else:
                print("err2", code)
        else:
            if code.startswith("0") or code.startswith("3") or code.startswith("2"):
                code = "SZ" + code
            elif code.startswith("5") or code.startswith("6") or code.startswith("9"):
                code = "SH" + code
            else:
                print("err3", code)
        return code

    @staticmethod  # 东财，输入代码获取股票k线数据 前复权, save == "y":  # 是否保存,fq=1前，=2后复权
    def east_history_k_data(code, fq, save=''):
        """http://quote.eastmoney.com/concept/sh603233.html#fschart-k"""
        import requests
        import demjson
        net = r"""http://push2his.eastmoney.com/api/qt/stock/kline/get
                ?fields1=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&beg=0&end=20500101&ut=fa5fd1943c7b386f172d6893dbfba10b&rtntype=6&secid=1.{}&klt=101&fqt={}&cb="""
        dragon_t = requests.get(net.format(code, fq)).text
        dragon_t = demjson.decode(dragon_t)
        # print(dragon_t)
        dragon_t = dragon_t.get('data', '')
        if dragon_t:
            dragon_t = dragon_t.get('klines', '')
            dragon_t = [i.split(",") for i in dragon_t]
            """['2022-06-21', '27.53', '28.02', '28.05', '27.08', '59788', '166037892.00', '3.55', '2.60', '0.71', '0.63']
                date          open     close    high low volume money amplitude振幅 up_change num_change涨跌(元） turnover
            """
            arr1 = ['date', 'open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'up_change', 'num_change',
                    'turnover']
            dragon_t = pd.DataFrame(dragon_t, columns=arr1)
            # dragon_t[['date']] = dragon_t[['date']].apply(pd.to_datetime)
            dragon_t[['open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude', 'up_change', 'num_change',
                      'turnover']] = dragon_t[['open', 'close', 'high', 'low', 'volume', 'amount', 'amplitude',
                                               'up_change', 'num_change', 'turnover']].apply(pd.to_numeric)
            print(dragon_t.head())
            print(dragon_t.dtypes)
            if save == "y":  # 是否保存
                # 对数收益率 = log(收盘价/前一个收盘价)
                dragon_t['log_'] = (np.log(dragon_t['close']/dragon_t['close'].shift(periods=1, axis=0)) * 100).round(3)
                with sqlite3.connect(gl_v.db_path) as conn:
                    dragon_t.to_sql(code+'_'+str(fq), con=conn, if_exists='replace', index=False)

stock = KzzStockPrice()
stock.east_history_k_data('603233', fq=2, save='y')  # fq=1前，=2后复权 大参
# stock.east_history_k_data('603368', fq=2, save='y')  # fq=1前，=2后复权  柳药
