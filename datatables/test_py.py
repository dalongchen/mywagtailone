from . import views


def test_sina_real_time():
    d = views.sina_real_time("301040")
    print(d)
    assert len(d) == 6


# 是否为未上市新股或日期（每天收盘5点后）或代码格式（sz,000001,sh.600000）
def test_inquiry_close():
    d = views.inquiry_close(["sz.301040", ], "2021-08-03", f="2")
    print(d)
    # [{'xd_2102': '301040', 'xd_2127': '45.9600', 'xd_2103': '中环海陆', 'xd_2106': '0280719034', 'xd_2108': '深圳Ａ股', 'xd_2109': '买入', 'xd_2126': '100', 'xd_2105': '', 'xd_3630': '0.000000'}]
    assert len(d) == 1


