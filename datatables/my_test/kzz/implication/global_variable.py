db_path = r"D:\myzq\axzq\T0002\stock_load\thesis\kzz\monte.db"  # 数据库路径
da_sen_s = '603233'  # 大参
liu_yao_s = '603368'  # 柳药
da_sen_b = 'sh113605'  # 大参
liu_yao_b = 'sh113563'


def time_show(func):
    from time import time
    def new_func(*arg, **kw):
        t1 = time()
        res = func(*arg, **kw)
        t2 = time()
        print(f"{func.__name__: >10} : {t2-t1:.6f} sec")
        return res
    return new_func

# if __name__ == '__main__':
#     pass

