db_path = r"D:\myzq\axzq\T0002\stock_load\my_stock\my_stock.db"  # 数据库路径


def time_show(func):
    from time import time
    def new_func(*arg, **kw):
        t1 = time()
        res = func(*arg, **kw)
        t2 = time()
        print(f"{func.__name__: >10} : {t2-t1:.6f} sec")
        return res
    return new_func

