import time


if __name__ == "__main__":
    from . import views
    print("今日的日期：" + time.strftime("%Y-%m-%d"))
    s = views.sina_real_time("000001")
    print(s)


