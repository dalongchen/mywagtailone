import pandas as pd
import sqlite3
your_db_path = r"D:\myzq\axzq\T0002\stock_load\thesis\statistics\my_statistics.db"  # 数据库路径


# 加载元数据 dimensionless, f如果有输出总体
def load_data(db_path, f=""):
    # 导入train_test_split用于拆分训练集和测试集
    from sklearn.model_selection import train_test_split
    with sqlite3.connect(db_path) as conn:
        select_table = """select * from dimensionless"""
        df = pd.read_sql(select_table, conn)
        # print(df)
        x_ = df.iloc[:, 2:]  # 访问第2-69列
        y_ = df["label"]
        # print(x_train)
        # print(y_train)
        if not f:
            """*arrays ：需要分割的数据，可以是list、numpy array等类型
            test_size：测试集所占的比例，取值范围在0到1之间
            train_size：训练集所占的比例，默认是等于1减去test_size
            shuffle：是否在分割之前打乱数据集，默认是True
            random_state：是随机数的种子。随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，
            保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。
            不填的话默认值为False，即每次切分的比例虽然相同，但是切分的结果不同。随机数的产生取决于种子，
            随机数和种子之间的关系遵从以下两个规则：种子不同，产生不同的随机数；种子相同，
            即使实例不同也产生相同的随机数。"""
            # 划分训练集和测试集
            x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.2, random_state=7)
            return x_train, x_test, y_train, y_test
        if f:
            return x_, y_

# load_data(your_db_path)


# pandas自动建表并读入数据.db_path：数据库路径, path：文件路径, table_name表名
def pandas_create_table(db_path, path="", table_name="", df="", drop=""):
    if path:
        df = pd.read_excel(path, engine="openpyxl")
        # data = df.values
        # print("获取到所有的值:\n{}".format(tuple(df.keys())))
        # print("获取到所有的值:\n{}".format(data))
    # data_frame = pd.DataFrame(df)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        if drop:  # 有则删除表
            print("删除表", table_name)
            cur.execute("DROP TABLE IF EXISTS {}".format(table_name))
        """如果不存在就创建表"""
        # create_sql = """CREATE TABLE IF NOT EXISTS {}{}""".format(table_name, col)
        # cur.execute(create_sql)
        df.to_sql(table_name, con=conn, if_exists='replace', index=False)
        cur.close()


def pca_analysis(db_path):
    # 加载元数据 dimensionless
    x_train, y_train = load_data(db_path, f="hh")
    # x_train, x_test, y_train, y_test = load_data(db_path)
    from sklearn.decomposition import PCA
    """
    class sklearn.decomposition.PCA(
        n_components=None,     # 指定保留的特征数量（降维以后的维数）
        copy=True,                    # 训练过程中是否影响原始样本数据，也影响到最后降维数据的返回方式
        whiten=False,                # 是否对降维后的数据提供白化处理。（白化就是每一维的特征做一个标准差归一化处理，除以一个标准偏差）
        svd_solver=’auto’,          #  SVD分解的计算方法
        tol=0.0,                         # 计算奇异值的公差
        iterated_power=’auto’,    # 采用svd_solver采用randomized方式的迭代次数。
        random_state=None)      # 同来创建随机数的实例，比如随机种

    n_components：代表返回的主成分的个数,也就是你想把数据降到几维
    n_components=2 代表返回前2个主成分
    0 < n_components < 1代表满足最低的主成分方差累计贡献率
    n_components=0.98，指返回满足主成分方差累计贡献率达到98%的主成分
    n_components=None，返回所有主成分
    n_components=‘mle’，将自动选取主成分个数n，使得满足所要求的方差百分比
    whiten: 判断是否进行白化。所谓白化，就是对降维后的数据的每个特征进行归一化，让方差都为1.对于PCA降维本身来说，
    一般不需要白化。如果你PCA降维后有后续的数据处理动作，可以考虑白化。默认值是False，即不进行白化。
    svd_solver：str类型，str {‘auto’, ‘full’, ‘arpack’, ‘randomized’}
    意义：定奇异值分解 SVD 的方法。
    svd_solver=auto：PCA 类自动选择下述三种算法权衡。
    svd_solver=‘full’:传统意义上的 SVD，使用了 scipy 库对应的实现。
    svd_solver=‘arpack’:直接使用 scipy 库的 sparse SVD 实现，和 randomized 的适用场景类似。
    svd_solver=‘randomized’:适用于数据量大，数据维度多同时主成分数目比例又较低的 PCA 降维。

    copy : bool (default True)，False：返回降维数据使用：fit_transform(X) True：返回降维数据使用：fit(X).transform(X)
    whiten : bool, optional (default False) 降维后的数据进行白化处理（标准差归一化处理）。
    tol : float >= 0, optional (default .0) 公差：在svd_solver == ‘arpack’计算奇异值方法中需要使用的公差。
    iterated_power : int >= 0, or ‘auto’, (default ‘auto’)使用svd_solver == ‘randomized’计算SVD需要用到的幂法迭代次数。
    random_state : int, RandomState instance or None, optional (default None)
     如果为int，则随机数生成器使用的种子；如果为random state实例，则随机数生成器为randomstate；如果为none，
     则随机数生成器为np.random使用的randomstate实例。当svd_solver=='arpack'或'randomized'时使用。
    提示：尽管上面参数显得有点复杂，并涉及数学计算的概念，但大部分情况下，除了第一个参数，其他参数我们采用默认参数即可。
    """
    model = PCA(
        n_components=0.99,  # 指定保留的特征数量（降维以后的维数）
        copy=True,  # 训练过程中是否影响原始样本数据，也影响到最后降维数据的返回方式
        whiten=True,  # 是否对降维后的数据提供白化处理。（白化就是每一维的特征做一个标准差归一化处理，除以一个标准偏差）
        svd_solver='auto',  # SVD分解的计算方法
        tol=0.0,  # 计算奇异值的公差
        iterated_power='auto',  # 采用svd_solver采用randomized方式的迭代次数。
        random_state=None
    )
    model.fit(x_train)  # fit(self, X，Y=None) #模型训练，由于PCA是无监督学习，所以Y=None，没有标签。
    X_new = model.fit_transform(x_train)
    # print('降维后的数据:', X_new)
    # ratio = model.explained_variance_ratio_
    # print('保留主成分的方差贡献率:', ratio)
    df = pd.DataFrame(X_new)
    # 在0位置加一列
    col_name = df.columns.tolist()
    col_name.insert(0, 'label')
    df = df.reindex(columns=col_name)
    df["label"] = y_train
    # print(df)
    # pandas自动建表并读入数据.db_path：数据库路径, path：文件路径, table_name表名
    pandas_create_table(db_path, path="", table_name="my_statistics_pca", df=df, drop="have")
    """（某个事件x发生概率）P(x|θ) 是参数 θ(概率) 下 x（样本） 出现的可能性，同时也是 x 出现时参数为 θ 的似然。
    P(x|θ) 越大，就说明参数如果为 θ，你的观测数据（或事件） x 就越可能出现。那你既然已经观测到数据 x 了，
    那么越可能让这个 x 出现的参数 θ，就越是一个靠谱的模型参数。这个参数的靠谱程度，就是似然。
    概率用于在已知一些参数的情况下，预测接下来的观测所得到的结果p；
    而似然性则是用于在已知某些观测所得到的结果时，对有关事物的性质的参数进行估计。
    """
    # Maxcomponent = model.components_
    # print('返回具有最大方差的成分:', Maxcomponent)
    # score = model.score(x_train)
    # print('所有样本的log似然平均值:', score)
    # print('奇异值:', model.singular_values_)  # 事件特征值中那些可以表征事务特征的值，把平庸不太能表征的元素去掉
    # print('噪声协方差:', model.noise_variance_)
    """inverse_transform(self, X)#将降维后的数据转换成原始数据，但可能不会完全一样
    2. explained_variance_：它代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分。
    3. explained_variance_ratio_：它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。（主成分方差贡献率）
    4. singular_values_：返回所被选主成分的奇异值。
    """
# pca_analysis(your_db_path)


# 子算法1
def my_bag_sun1(x_test, x_train, y_test, y_train):
    from sklearn.ensemble import BaggingClassifier  # 导入用于构建分类模型的 BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    """这里的bagging算法采用K近邻算法分类器KNeighborsClassifier，其原理是使用欧式距离的方法进行数据的测量以及分类。
    max_samples取0.1，也就是每次从训练集中抽取10%的数据用于训练每个基本估计器，再本算法中，
    max_samples高于0.1将会降低模型预测准确率。max_features取0.7，从训练集中提取的用于训练每个基本估计器的特征数。
    我们的数据表列数据为69列，扣除id列、和标签列后，有效特征数为67个。也就是每次喂给估计器的特征数为67的70%。如果低于70%，
    预测准确率有一定的下降。random_state用于控制原始数据集的随机重采样（采样和特征）。 如果基估计器接受随机状态属性，
    则为集合中的每个实例生成不同的种子。再本例中random_state设置为1，也就是采用随机采样的方式。
    如果设置为0，准确率略有下降。"""
    """base_estimator：object, default=None
                    适合于数据集的随机子集的基估计量。
                    如果 None，则基估计量是决策树。
    n_estimators：int, default=10
                  训练基学习器（基本估计器）的个数。

    max_samples：int or float, default=1.0
                 从X抽取的样本数，用于训练每个基本估计器
                 如果为int，则抽取 max_samples 个样本。
                 如果为float，则抽取 max_samples * X.shape[0] 个样本。

    max_features：int or float, default=1.0
                  从X中提取的用于训练每个基本估计器的特征数
                  如果为int，则绘制 max_features 个特征。
                  如果是float，则绘制特征 max_features * X.shape[1] 个特征。

    bootstrap：bool, default=True
               是否为放回取样。如果为False，则执行不替换的样本。

    bootstrap_features：bool, default=False
                        是否针对特征随机取样。

    oob_score：bool, default=False
               是否使用袋外样本估计泛化误差。（是否使用现成的样本来估计泛化误差。）

    warm_start：bool, default=False
                当设置为True时，重用前面调用的解决方案来拟合并向集成添加更多的估计
                量，否则，只拟合一个全新的集成。

    n_jobs：int, default=None
            使用几个处理起来运行；
            -1表示使用所有处理器。

    random_state：int or RandomState, default=None
                  控制原始数据集的随机重采样（采样和特征）。
                  如果基估计器接受随机状态属性，则为集合中的每个实例生成不同的种子。
                  为跨多个函数调用的可复制输出传递int。

    verbose：int, default=0
             控制拟合和预测时的详细程度。

    Attributes:
    base_estimator_：估计量
                     集合增长的基估计量。

    n_features_：int
                 执行拟合（fit）时的特征数。

    estimators_：list of estimators
                 拟合基估计量的集合。

    estimators_samples_: list of arrays
                         每个基本估计量的抽取样本的子集。

    estimators_features_：list of arrays
                          每个基本估算器的绘制要素的子集。

    classes_：ndarray of shape (n_classes,)
              类标签。

    n_classes_：int or list
                The number of classes.
                类数。

    oob_score_：float
                使用没有选到的样本数据估计获得的训练数据集的分数。
                该属性仅在oob_score为True 时存在。

    oob_decision_function_：ndarray of shape (n_samples, n_classes)
                            使用训练集上的实际估计值计算的决策函数。
                            如果n_estimators、小，则可能在引导过程中从未遗漏任何数据点。在这种情况下， oob_decision_function_可能包含NaN。
                            该属性仅在oob_score为True 时存在。
    """
    bagging = BaggingClassifier(KNeighborsClassifier(),
                                max_samples=0.1,
                                max_features=0.7,
                                random_state=1)
    bagging.fit(x_train, y_train)
    y_pre = bagging.predict(x_test)
    accuracy_ = accuracy_score(y_test, y_pre)
    print("子算法1准确率：", accuracy_)


# 子算法2
def my_bag_sun2(x_test, x_train, y_test, y_train):
    from sklearn.ensemble import BaggingClassifier  # 导入用于构建分类模型的 BaggingClassifier
    # 导入用于构建分类模型的DecisionTreeClassifier
    from sklearn.tree import DecisionTreeClassifier
    # 导入评估模型预测性能所需的函数
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import recall_score
    # 这里我们采用决策时模型作为基分类器，并采用熵作为指标对属性进行划分
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
    """bootstrap：bool, default=True 是否为放回取样。如果为False，则执行不替换的样本。通过装袋集成方法生成500个决策树.
    bootstrap_features：bool, default=False 是否针对特征随机取样。
    random_state：int or RandomState, default=None, 控制原始数据集的随机重采样（采样和特征）。
    如果基估计器接受随机状态属性，则为集合中的每个实例生成不同的种子。为跨多个函数调用的可复制输出传递int。
    'max_features': 0.85, 'max_samples': 0.9, 'n_estimators': 400"""
    bag = BaggingClassifier(base_estimator=tree, n_estimators=400,
                            max_features=0.85, max_samples=0.9, bootstrap=True,
                            bootstrap_features=False, n_jobs=-1, random_state=7)

    # 1、评估单个决策树构建的模型性能

    # 通过训练集训练单个决策树
    tree = tree.fit(x_train, y_train)
    # 用单个决策树模型对训练集的类别进行预测
    y_train_pred = tree.predict(x_train)
    # 用单个决策树模型对测试集的类别进行预测
    y_test_pred = tree.predict(x_test)
    # 查看单个决策树模型在训练集上的准确率
    tree_train_accuracy = accuracy_score(y_train, y_train_pred)
    # 查看单个决策时模型在测试集上的准确率
    tree_test_accuracy = accuracy_score(y_test, y_test_pred)

    # 打印出单个模型在训练集和测试集上的准确率

    # 查看单个决策树模型在训练集上的灵敏度
    tree_train_sen = recall_score(y_train, y_train_pred, pos_label=1)
    # 查看单个决策时模型在测试集上的灵敏度
    tree_test_sen = recall_score(y_test, y_test_pred, pos_label=1)

    # 查看单个决策树模型在训练集上的特异性
    tree_train_spe = recall_score(y_train, y_train_pred, pos_label=0)
    # 查看单个决策时模型在测试集上的特异性
    tree_test_spe = recall_score(y_test, y_test_pred, pos_label=0)

    # 打印出单个模型在训练集和测试集上的准确率、灵敏度、特异性
    print('Decision tree train/test accuracies(准确率) %.3f/%.3f' % (tree_train_accuracy, tree_test_accuracy))
    print('Decision tree train/test sen(灵敏度) %.3f/%.3f' % (tree_train_sen, tree_test_sen))
    print('Decision tree train/test spe(特异性) %.3f/%.3f' % (tree_train_spe, tree_test_spe))

    # 2、评估通过bagging集成的分类器性能

    # 通过训练集训练多个决策树集成的模型
    bag = bag.fit(x_train, y_train)
    # 运用bagging集成的模型预测训练集的类别
    y_train_pred = bag.predict(x_train)
    # 运用bagging集成的模型预测测试集的类别
    y_test_pred = bag.predict(x_test)

    # 评估集成模型的性能
    # 查看集成模型在训练集上的准确率
    bag_train_accuracy = accuracy_score(y_train, y_train_pred)
    # 查看集成模型在测试集上的准确率
    bag_test_accuracy = accuracy_score(y_test, y_test_pred)

    # 查看集成模型在训练集上的灵敏度
    bag_train_sen = recall_score(y_train, y_train_pred, pos_label=1)
    # 查看集成模型在测试集上的灵敏度
    bag_test_sen = recall_score(y_test, y_test_pred, pos_label=1)

    # 查看集成模型在训练集上的特异性
    bag_train_spe = recall_score(y_train, y_train_pred, pos_label=0)
    # 查看集成模型在测试集上的特异性
    bag_test_spe = recall_score(y_test, y_test_pred, pos_label=0)

    # 打印出集成模型在训练集和测试集上的准确率、灵敏度、特异性
    print('Bagging train/test accuracies(准确率) %.3f/%.3f' % (bag_train_accuracy, bag_test_accuracy))
    print('Bagging train/test sen(灵敏度) %.3f/%.3f' % (bag_train_sen, bag_test_sen))
    print('Bagging train/test spe(特异性) %.3f/%.3f' % (bag_train_spe, bag_test_spe))


# 交叉验证法1
def my_cross_value_score(db_path):
    # from sklearn import svm
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier  # 导入用于构建分类模型的DecisionTreeClassifier
    x_, y_ = load_data(db_path, f="my_cross_value_score")  # 加载元数据 dimensionless
    # print(x_train)
    # print(x_test)
    """clf是我们使用的算法，cv是我们使用的交叉验证的生成器或者迭代器，它决定了交叉验证的数据是如何划分的，
    当cv的取值为整数的时候，使用(Stratified)KFold方法。 scoring，决定了其中的分数计算方法"""
    # clf = svm.SVC(kernel='linear', C=1)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
    # 通过装袋集成方法生成500个决策树
    clf = BaggingClassifier(base_estimator=tree, n_estimators=400,
                            max_features=0.85, max_samples=0.9, bootstrap=True,
                            bootstrap_features=False, n_jobs=-1, random_state=7)
    scores = cross_val_score(clf, x_, y_, cv=5, scoring='f1_macro')
    print(scores)
    print("交叉验证法加权平均分：", scores.mean())


# 随机森林
def random_forest_classifier(db_path):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier  # 决策树

    Xtrain, Xtest, Ytrain, Ytest = load_data(db_path)    # 加载元数据 dimensionless
    # sklearn建模的基本流程
    clf = DecisionTreeClassifier(random_state=0)
    """'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'max_features': 9, 'min_samples_leaf': 5, 'min_samples_split': 8}
 'bootstrap': False, 'criterion': 'entropy', 'max_depth': None, 'max_features': 8, 'min_samples_leaf': 1, 'min_samples_split': 2}
 """
    rfc = RandomForestClassifier(n_estimators=20, criterion='gini', max_depth=None,
                                 min_samples_split=8, min_samples_leaf=5, min_weight_fraction_leaf=0.0,
                                 max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                                 min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=-1,
                                 random_state=None, verbose=0, warm_start=False, class_weight=None)

    clf = clf.fit(Xtrain, Ytrain)
    rfc = rfc.fit(Xtrain, Ytrain)
    score_c = clf.score(Xtest, Ytest)  # 是精确度
    score_r = rfc.score(Xtest, Ytest)
    print('Single Tree:{}'.format(score_c), 'Random Forest:{}'.format(score_r))  # format是将分数转换放在{}中


# 交叉验证法, 网格搜索优化参数
def my_grid_search_cv(db_path):
    from sklearn.ensemble import BaggingClassifier  # 导入用于构建分类模型的 BaggingClassifier
    from sklearn.model_selection import StratifiedKFold  # 交叉验证
    from sklearn.tree import DecisionTreeClassifier  # 导入用于构建分类模型的DecisionTreeClassifier
    from sklearn.model_selection import GridSearchCV  # 网格搜索

    # 加载元数据 dimensionless
    x_train, x_test, y_train, y_test = load_data(db_path)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
    # 通过装袋集成方法生成500个决策树
    model = BaggingClassifier(base_estimator=tree, bootstrap=True,
                              bootstrap_features=False, n_jobs=-1, random_state=1)
    # model = BaggingClassifier(base_estimator=tree, n_estimators=500,
    #                         max_samples=1.0, max_features=0.6, bootstrap=True,
    #                         bootstrap_features=False, n_jobs=1, random_state=1)
    # print(model)
    n_estimators = [500, 400, 300]  #
    max_features = [0.9, 0.85, 0.8]
    max_samples = [1, 0.95, 0.9]
    param_grid = dict(max_features=max_features, max_samples=max_samples, n_estimators=n_estimators)  # 转化为字典格式，网络搜索要求
    kflod = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)  # 将训练/测试数据集划分10个互斥子集，
    """
    class sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None, fit_params=None, n_jobs=1,
    iid=True, refit=True, cv=None, verbose=0, pre_dispatch=‘2*n_jobs’, error_score=’raise’, return_train_score=’warn’)
    （1）estimator,选择使用的分类器，并且传入除需要确定最佳的参数之外的其他参数。每一个分类器都需要一个scoring参数，或者score方法：
    estimator=RandomForestClassifier(min_samples_split=100,min_samples_leaf=20,max_depth=8,max_features='sqrt',random_state=10),
    （2）param_grid,需要最优化的参数的取值，值为字典或者列表，例如：param_grid =param_test1，
    param_test1 = {'n_estimators':range(10,71,10)}。
    （3）scoring=None,模型评价标准，默认None,这时需要使用score函数；或者如scoring='roc_auc'，根据所选模型不同，评价准则不同。
    字符串（函数名），或是可调用对象，需要其函数签名形如：scorer(estimator, X, y)；如果是None，则使用estimator的误差估计函数。
    具体值的选取看本篇第三节内容。
    （4）fit_params=None（5）n_jobs=1,n_jobs: 并行数，int：个数,-1：跟CPU核数一致, 1:默认值
    （6）iid=True,iid:默认True,为True时，默认为各个样本fold概率分布一致，误差估计为所有样本之和，而非各个fold的平均。
    （7）refit=True,默认为True,程序将会以交叉验证训练集得到的最佳参数，重新对所有可用的训练集与开发集进行，
    作为最终用于性能评估的最佳模型参数。即在搜索参数结束后，用最佳参数结果再次fit一遍全部数据集。
    （8）cv=None,交叉验证参数，默认None，使用三折交叉验证。指定fold数量，默认为3，也可以是yield训练/测试数据的生成器。
    （9）verbose=0, scoring=None,verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：偶尔输出，>1：对每个子模型都输出。
    （10）pre_dispatch=‘2*n_jobs’,指定总共分发的并行任务数。当n_jobs大于1时，数据将在每个运行点进行复制，
    这可能导致OOM，而设置pre_dispatch参数，则可以预先划分总共的job数量，使数据最多被复制pre_dispatch次
    （11） error_score=’raise’（12）   return_train_score=’warn’,如果“False”，cv_results_属性将不包括训练分数
    回到sklearn里面的GridSearchCV，GridSearchCV用于系统地遍历多种参数组合，通过交叉验证确定最佳效果参数。

    4.属性
    （1）cv_results_ : dict of numpy (masked) ndarrays,具有键作为列标题和值作为列的dict，可以导入到DataFrame中。
    注意，“params”键用于存储所有参数候选项的参数设置列表。
    （2）best_estimator_ : estimator,通过搜索选择的估计器，即在左侧数据上给出最高分数（或指定的最小损失）的估计器。
    如果refit = False，则不可用。
    （3）best_score_ : float best_estimator的分数.（4）best_params_ : dict   在保存数据上给出最佳结果的参数设置
    （5）best_index_ : int      对应于最佳候选参数设置的索引（cv_results_数组）。
    search.cv_results _ ['params'] [search.best_index_]中的dict给出了最佳模型的参数设置，
    给出了最高的平均分数（search.best_score_）
    （6）scorer_ : function,,Scorer function used on the held out data to choose the best parameters for the model.
    （7）n_splits_ : int,,The number of cross-validation splits (folds/iterations).
    （8）grid_scores_：给出不同参数情况下的评价结果
    """
    # scoring指定损失函数类型，n_jobs指定全部cpu跑，cv指定交叉验证
    grid_search = GridSearchCV(model, param_grid, scoring='neg_log_loss', n_jobs=-1, cv=kflod)
    """grid_scores_：给出不同参数情况下的评价结果。best_params_：描述了已取得最佳结果的参数的组合best_score_：
    成员提供优化过程期间观察到的最好的评分具有键作为列标题和值作为列的dict，可以导入到DataFrame中。
    注意，“params”键用于存储所有参数候选项的参数设置列表。"""
    grid_result = grid_search.fit(x_train, y_train)  # 运行网格搜索
    """max_features': 0.85, 'max_samples': 0.9, 'n_estimators': 350"""
    print("Best: %f using %s" % (grid_result.best_score_, grid_search.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    params = grid_result.cv_results_['params']
    for mean, param in zip(means, params):
        print("%f  with:   %r" % (mean, param))

# my_bag(your_db_path)


# 用于报告超参数搜索的最好结果的函数
def report(results, n_top=0):  # 从每次交叉验证中的历史信息中找到最好的三个
    import numpy as np
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == 1)  # flatnozero返回不为零的索引，==1指的是值最大的那个
        print("candidates:", candidates)
        for candidate in candidates:
            print("结果中的排名 : {0}".format(i))  # 结果中的排名
            print("平均validation score:{0:.3f} (标准差: {1:.3f})".format(
                results['mean_test_score'][candidate],  # 平均
                results['std_test_score'][candidate]  # 标准差
            ))
            print("参数最终优化的值: {0}".format(results['params'][candidate]))  # 参数最终的优化的值


# 随机采样参数优化
def randomized_search_cv(db_path, tab_name=""):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV  # 导入
    from scipy.stats import randint as sp_randint  # 生成a-b随机数
    from time import time
    x, y = load_data(db_path, tab_name=tab_name, f="my_cross_value_score")  # 加载元数据 dimensionless
    """
    class sklearn.ensemble.RandomForestClassifier
    (n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
    min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0,
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0,
    warm_start=False, class_weight=None)
    criterion : 衡量分裂质量的性能（函数）。 受支持的标准是基尼不纯度的"gini",和信息增益的"entropy"（熵）。注意：这个参数是特定树的
    Gini impurity衡量的是从一个集合中随机选择一个元素，基于该集合中标签的概率分布为元素分配标签的错误率。
    对于任何一个标签下的元素，其被分类正确的条件概率可以理解为在选择元素时选中该标签的概率与在分类时选中该标签的概率。
    基于上述描述，Gini impurity的计算就非常简单了，即1减去所有分类正确的概率，得到的就是分类不正确的概率。
    若元素数量非常多，切所有元素单独属于一个分类时，Gini不纯度达到极小值0.
    max_features : 默认值为"auto".寻找最佳分割时需要考虑的特征数目：&如果是int，就要考虑每一次分割处的max_feature特征
    &如果是float，那么max_features就是一个百分比，那么（max_feature*n_features）特征整数值是在每个分割处考虑的。
    &如果是auto，那么max_features=sqrt(n_features)，即n_features的平方根值。
    &如果是log2，那么max_features=log2(n_features),&如果是None,那么max_features=n_features
    注意：寻找分割点不会停止，直到找到最少一个有效的节点划分区，即使它需要有效检查超过max_features的特征
    max_depth :（决策）树的最大深度。如果值为None，那么会扩展节点，直到所有的叶子是纯净的，
    或者直到所有叶子包含少于min_sample_split的样本。

    min_samples_split : 默认值为2）分割内部节点所需要的最小样本数量：如果为int，那么考虑min_samples_split作为最小的数字。
    ~如果为float，那么min_samples_split是一个百分比，并且把ceil(min_samples_split*n_samples)是每一个分割最小的样本数量。
    在版本0.18中更改：为百分比添加浮点值。

    min_samples_leaf : 默认值为1）需要在叶子结点上的最小样本数量： ~如果为int，那么考虑min_samples_leaf作为最小的数字。
    ~如果为float，那么min_samples_leaf为一个百分比，并且ceil(min_samples_leaf*n_samples)是每一个节点的最小样本数量。
    在版本0.18中更改：为百分比添加浮点值。

    bootstrap : 默认值为True）建立决策树时，是否使用有放回抽样
    注意点：
    参数的默认值控制决策树的大小（例如，max_depth，，min_samples_leaf等等），
    导致完全的生长和在某些数据集上可能非常大的未修剪的树。为了降低内容消耗，
    决策树的复杂度和大小应该通过设置这些参数值来控制。
    这些特征总是在每个分割中随机排列。 因此，即使使用相同的训练数据，max_features = n_features和bootstrap = False，
    如果在搜索最佳分割期间所列举的若干分割的准则的改进是相同的，那么找到的最佳分割点可能会不同。
    为了在拟合过程中获得一个确定的行为，random_state将不得不被修正

    """
    clf = RandomForestClassifier(n_estimators=20, n_jobs=-1)  # 随机森林分类
    # 需要优化的参数字典
    param_dist = {"max_depth": [3, None],
                  "criterion": ["gini", "entropy"],
                  "min_samples_split": sp_randint(2, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "max_features": sp_randint(1, 11),
                  "bootstrap": [True, False]}

    """随机搜索采用的方法与网格稍有不同。它不是详尽地尝试超参数的每一个单独组合，这在计算上可能是昂贵和耗时的，
    它随机抽样超参数，并试图接近最好的集合。
    你永远不要根据RandomSearchCV的结果来选择你的超参数。只使用它来缩小每个超参数的值范围，
    以便您可以为GridSearchCV提供更好的参数网格.
    你会问，为什么不从一开始就使用GridSearchCV呢?看看初始参数网格:
    n_iterations = 1for value in param_grid.values():n_iterations *= len(value)>>> n_iterations13680
    有13680个可能的超参数组合和3倍CV, GridSearchCV将必须适合随机森林41040次。使用RandomizedGridSearchCV，我们得到了相当好的分数，并且只需要100 * 3 = 300 次训练。
    现在，是时候在之前的基础上创建一个新的参数网格，并将其提供给GridSearchCV:
    new_params = {"n_estimators": [650, 700, 750, 800, 850, 900, 950, 1000], "max_features": ['sqrt'], "max_depth": [10, 15, 20, 25, 30], "min_samples_split": [2, 4, 6], "min_samples_leaf": [1, 2], "bootstrap": [False],}
    这次我们有
    n_iterations = 1for value in new_params.values():n_iterations *= len(value)>>> n_iterations240
    240种组合，这还是很多,但是比起之前的计算已经少很多了。让我们导入GridSearchCV并实例化它:
    from sklearn.model_selection import GridSearchCVforest = RandomForestRegressor()grid_cv = GridSearchCV(forest, new_params, n_jobs=-1)
    我不需要指定评分和CV，因为我们使用的是默认设置，所以不需要指定。让我们适应并等待:
    %%time_ = grid_cv.fit(X, y)print('Best params:\n')print(grid_cv.best_params_, '\n')Best params:{'bootstrap': False, 'max_depth': 15, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 950} Wall time: 35min 18s
    35分钟后，我们得到了以上的分数，这一次——确实是最优的分数。让我们看看他们与RandomizedSearchCV有多少不同:
    >>> grid_cv.best_score_0.8696576413066612
    你感到惊讶吗?我也是。结果的差别很小。然而，这可能只是给定数据集的一个特定情况。
    当您在实践中使用需要大量计算的模型时，最好得到随机搜索的结果，并在更小的范围内在网格搜索中验证它们。
    其搜索策略如下：
    （a）对于搜索范围是distribution的超参数，根据给定的distribution随机采样；
    （b）对于搜索范围是list的超参数，在给定的list中等概率采样；
    （c）对a、b两步中得到的n_iter组采样结果，进行遍历。
    （补充）如果给定的搜索范围均为list，则不放回抽样n_iter次
    """
    n_iter_search = 20  # 迭代次数
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=-1)  # 随机采样搜索
    start = time()
    random_search.fit(x, y)  # 拟合
    print("random took %.2f seconds for %d candidates parameter settings" % (time()-start, len(random_search.cv_results_['params'])))  # 花费时间和多少个候选参数
    cv_results = random_search.cv_results_
    print("cv_results:", cv_results)
    report(cv_results)
    # best parameters
    print("最好优化值", random_search.best_params_)
    """ """


# randomized_search_cv(your_db_path, tab_name="my_statistics_pca")  # 随机采样参数优化
