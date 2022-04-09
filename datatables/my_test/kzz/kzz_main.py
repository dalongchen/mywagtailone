from mywagtailone.datatables.my_test.kzz import kzz_test
if __name__ == "__main__":
    flag = "update_bond_day_year_3"
    if flag == "bond_price_1":
        kzz_test.bond_price()  # get可转债85-120 into table kzz80-120
    elif flag == "my_interest_value_2":
        kzz_test.my_interest_value()  # my可转债纯债值,默认只更新my_bond_value,仅2步以完成
        # kzz_test.my_interest_value(f="east")  # my可转债纯债值
    elif flag == "update_bond_day_year_3":
        kzz_test.update_bond_day_year()  # 更新债券剩余天数和年数
    elif flag == "get_bond_stock_up_3":
        kzz_test.get_bond_stock_up()  # 获取可转债对应股票涨幅
    elif flag == "kzz_stock_standard_deviation_4":
        kzz_test.kzz_stock_standard_deviation()  # 计算可转债对应stock of 标准差
    elif flag == "kzz_stock_option_value_5":
        kzz_test.kzz_stock_option_value()  # 计算可转债对应stock期权价值
    elif flag == "plot_probability_distribution_6":
        kzz_test.plot_probability_distribution()  # 画数据的概率分布
    elif flag == "east_kzz_only_bond_value_7":
        kzz_test.east_kzz_only_bond_value()  # 读东财可转债的纯债价值
    elif flag == "east_kzz_redeem_8":
        kzz_test.east_kzz_redeem()  # 读东财可转债的到期赎回价

