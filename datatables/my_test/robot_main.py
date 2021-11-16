from mywagtailone.datatables.my_test import robot_test
if __name__ == "__main__":
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
    robot_test.predict_stock(r"D:\ana\envs\py36\mywagtailone\datatables\datatable.db")
