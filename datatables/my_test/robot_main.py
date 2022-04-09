if __name__ == "__main__":
    robot = "fuzzy_decision"
    if robot == "test":
        from mywagtailone.datatables.my_test import robot_test
        # group, labels = robot_test.createDataSet()
        # print(group)
        # gr = robot_test.classify0([5, 10, 70], group, labels, 3)
        # print(gr)
        # robot_test.datingClassTest()
        # robot_test.get_mnist_picture_data()
        # robot_test.base_learning()
        # robot_test.film_review_imdb()
        # robot_test.learn_reuters()
        # robot_test.predict_house_price()
        # robot_test.predict_c()
        ww = ""
        if ww == "1":
            robot_test.predict_stock(r"D:\ana\envs\py36\mywagtailone\datatables\datatable.db")
        # robot_test.robot_my_test()
        # robot_test.robot_relevance()  # 相关性分析
    elif robot == "unsupervised_data":
        from mywagtailone.datatables.my_test import unsupervised_data as un_da
        rob = "06_activity13"
        if rob == "01_exercise06":
            un_da.unsupervised_lesson01_exercise06()
        elif rob == "06_exercise12":
            un_da.unsupervised_lesson06_exercise12()
        elif rob == "06_activity13":
            un_da.unsupervised_lesson06_activity13()
    elif robot == "machine_learning_in_action":
        from mywagtailone.datatables.my_test import machine_learning_in_action as un_da
        rob = "ch02_2"
        if rob == "ch02_2":
            un_da.machine_learning_in_action_ch02_2()
        elif rob == "06_exercise12":
            un_da.unsupervised_lesson06_exercise12()
        elif rob == "06_activity13":
            un_da.unsupervised_lesson06_activity13()
    elif robot == "fuzzy_decision":
        from mywagtailone.datatables.my_test import fuzzy_decision as fuzzy
        rob = "fuzzy_dragon_tiger2"
        if rob == "fuzzy_decision":
            fuzzy.fuzzy_decision()
        elif rob == "multiple_fuzzy_decision":
            fuzzy.multiple_fuzzy_decision()
        elif rob == "fuzzy_dragon_tiger":
            fuzzy.fuzzy_dragon_tiger()
