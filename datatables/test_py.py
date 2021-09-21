from . import views
from .tool import tools
import pytest


# 新浪
# @pytest.mark.tested
def test_sina_real_time():
    d = views.sina_real_time("688711")
    print(d)
    assert len(d) == 6


# 网易收盘价
# @pytest.mark.tested
def test_w163_history():
    d = views.w163_history("301040")
    print(d)
    assert 6 == 6


# 是否为未上市新股或日期（每天收盘5点后）或代码格式（sz,000001,sh.600000）
def test_inquiry_close():
    d = views.inquiry_close(["sz.301040", ], "2021-08-03", f="2")
    print(d)
    # [{'xd_2102': '301040', 'xd_2127': '45.9600', 'xd_2103': '中环海陆', 'xd_2106': '0280719034', 'xd_2108': '深圳Ａ股', 'xd_2109': '买入', 'xd_2126': '100', 'xd_2105': '', 'xd_3630': '0.000000'}]
    assert len(d) == 1


# 查指数加sh.
def test_history_k_data():
    tools.history_k_data(code="sh.000001", start_date='2018-07-31', end_date='2021-07-31', frequency="d", adjustflag="1")


# 判断返回那一天
@pytest.mark.tested
def test_get_date():
    da = tools.get_date()
    print(da)
    assert (1, 2, 3) == (1, 2, 3)


# 传入路径，读取csv数据,
# @pytest.mark.tested
def test_read_data():
    dd = tools.read_data()



