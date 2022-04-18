from mywagtailone.datatables.my_test.kzz import kzz_test
if __name__ == "__main__":
    flag = "kzz_stock_option_value"
    if flag == "bond_price":
        kzz_test.bond_price()  # 1 get可转债85-120 into table kzz80-120
    elif flag == "update_bond_day_year_yield":
        kzz_test.update_bond_day_year_yield()  # 2 更新债券剩余天数,年数,债券年收益率
    elif flag == "my_interest_value":
        kzz_test.my_interest_value()  # 3 my可转债纯债值,默认只更新my_bond_value
        # kzz_test.my_interest_value(f="east")  # my可转债纯债值,包括更新东财纯债价值
    elif flag == "my_bond_value_revise":
        kzz_test.my_bond_value_revise()  # 添加未付利息贴现，修正my可转债纯债值   更新my_bond_revise
    elif flag == "get_bond_stock_up":
        kzz_test.get_bond_stock_up()  # 获取可转债对应股票涨幅
    elif flag == "kzz_stock_standard_deviation":
        kzz_test.kzz_stock_standard_deviation()  # 计算可转债对应stock of 标准差
    elif flag == "kzz_stock_option_value":
        kzz_test.kzz_stock_option_value()  # 计算可转债对应stock期权价值
    elif flag == "plot_probability_distribution_6":
        kzz_test.plot_probability_distribution()  # 画数据的概率分布为kzz_stock_option_value的子方法
    elif flag == "east_kzz_only_bond_value_7":
        kzz_test.east_kzz_only_bond_value()  # 读东财可转债的纯债价值为my_interest_value的子方法
    elif flag == "east_kzz_redeem_8":
        kzz_test.east_kzz_redeem()  # 读东财可转债的到期赎回价为bond_price的子方法
    else:
        print("err", flag)

