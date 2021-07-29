from baiduspider import BaiduSpider
from pprint import pprint
from time import sleep
import time
import datetime
import re
spider = BaiduSpider()


def test_self():
    arr = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

    # 方法2
    ar2 = list(map(list, zip(*arr)))
    print(ar2)


# 百度个股负面
def bai_du(kw):
    import html
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
            if ii.get("origin", "") != "股吧":
                l = list(ii.values())
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



