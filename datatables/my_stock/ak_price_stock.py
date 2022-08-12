import numpy as np
import pandas as pd
import sqlite3
import global_variable as gl_v
import akshare as ak


# 获取行情价
class StockPrice(object):

    def __init__(self):
        pass

    @staticmethod  # 新浪获取k线数据.
    def dc_stock_price(code, save='', adjust=''):
        # stock_df = ak.stock_zh_a_daily(symbol=code, adjust=adjust)
        # stock_df = ak.stock_zh_a_hist_163(symbol=code)
        stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20170301", end_date='20210907', adjust="")
        print(stock_zh_a_hist_df)
        # print(stock_df)
        # stock_df.index = stock_df.index.strftime('%Y-%m-%d')
        # if save == "y":
        #     with sqlite3.connect(gl_v.db_path) as conn:
        #         stock_df.to_sql(code+adjust, con=conn, if_exists='replace', index=True)

    @staticmethod  # 财务指标
    def ak_stock_finance(code, save=''):
        stock_financial = ak.stock_financial_analysis_indicator(stock=code)
        print(stock_financial)
        stock_financial.index = stock_financial.index.strftime('%Y-%m-%d')
        if save == "y":
            with sqlite3.connect(gl_v.db_path) as conn:
                stock_financial.to_sql(code, con=conn, if_exists='replace', index=True)

    @staticmethod  # 历史波动率
    def history_volatility(s_code):
        with sqlite3.connect(gl_v.db_path) as conn:
            cur = conn.cursor()
            se_sql = r"select date,log_ from '{}'".format(s_code)
            # se_sql = r"select date,up_change,log_ from '{}'".format(s_code)
            data = pd.read_sql(se_sql, conn)
            # data = pd.read_sql(se_sql, conn, parse_dates=['date'])
            # data = data.astype('float', errors='ignore')
            # data = data.dropna()
            # dd = data.dropna().head(1189)
            # print(dd.values)

            """滚动计算年华波动率"""
            # data['y_v_change'] = (np.sqrt(244) * data['up_change'].rolling(122).std()).round(3)
            data['y_v_log'] = (np.sqrt(244) * (data['log_']/100).rolling(122).std()).round(6)
            data = data.loc[:, ['y_v_log', 'date']]
            # data = data.loc[:, ['y_v_change', 'y_v_log', 'date']]
            # print(data.values)
            """加列"""
            # sql2 = r"""alter table '{}' add y_v_change number(6)""".format(code)
            # sql2 = r"""alter table '{}' add y_v_log number(6)""".format(s_code)
            # cur.execute(sql2)
            """更新"""
            sql3 = "UPDATE '{}' SET y_v_log=(?) WHERE date=(?)".format(s_code)
            # sql3 = "UPDATE '{}' SET y_v_change=(?),y_v_log=(?) WHERE date=(?)".format(code)
            # print(sql3)
            # cur.executemany(sql3, data.values)
            cur.close()

stock_price = StockPrice()
stock_price.dc_stock_price("sh603368", save='y', adjust='qfq')  # 新浪获取k线数据 qfq
# stock_price.ak_stock_finance("sh601398", save='y')  # 财务指标

