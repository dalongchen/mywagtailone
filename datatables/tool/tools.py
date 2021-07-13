from baiduspider import BaiduSpider
from pprint import pprint
from time import sleep
spider = BaiduSpider()


def test_self():
    arr = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

    # 方法2
    ar2 = list(map(list, zip(*arr)))
    print(ar2)


# 百度个股负面
def bai_du(kw):
    bai = spider.search_web(query=kw, pn=1, exclude=['tieba', 'video'])
    # print(type(bai))
    b = bai.get("results", "")
    to = bai.get("total", "")
    # print(kw)
    # pprint(b[2:])
    # bb.get("time", "")
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
            sleep(0.6)
            # pprint(i)
            sp = spider.search_web(query=kw, pn=i, exclude=['tieba', 'video'])
            sp = sp.get("results", "")
            if sp:
                bai_d.extend(sp[2:])
        # print(bai_d)
        return bai_d
    return




