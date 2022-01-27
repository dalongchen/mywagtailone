"""
文件数据路径已经改变，运行本页代码请改路径D:\ana\envs\py36\mywagtailone\datatables\my_test\unsupervited_data
file path has been changed.if you are runing this script to need to change path
"""


def unsupervised_lesson01_exercise06():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import cdist
    p = r"D:\ana\envs\py36\mywagtailone\datatables\unsupervited_data\Applied-Unsupervised-Learning-with-Python-master\Lesson01\Exercise06\iris_data.csv"
    iris = pd.read_csv(p, header=None)
    iris.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'species']
    X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    # print(X)

    def k_means(X, K):
        # Keep track of history so you can see k-means in action
        centroids_history = []
        labels_history = []
        """np.random.choice(5, 3)#在[0, 5)内输出五个数字并组成一维数组（ndarray）
        相当于np.random.randint(0, 5, 3)"""
        rand_index = np.random.choice(X.shape[0], K)
        centroids = X[rand_index]
        # print(centroids)
        centroids_history.append(centroids)
        # print(centroids_history)
        while True:
            """语法：scipy.spatial.distance.cdist(XA, XB, metric='euclidean', p=None, V=None, VI=None, w=None)，
            该函数用于计算两个输入集合的距离，通过metric参数指定计算距离的不同方式得到不同的距离度量值
             braycurtis
             canberra
             chebyshev：切比雪夫距离
             cityblock
             correlation：相关系数
             cosine：余弦夹角
             dice
             euclidean：欧式距离
             hamming：汉明距离
             jaccard：杰卡德相似系数
             kulsinski
             mahalanobis：马氏距离
             matching
             minkowski：闵可夫斯基距离
             rogerstanimoto
             russellrao
             seuclidean：标准化欧式距离
             sokalmichener
             sokalsneath
             sqeuclidean
             wminkowski
             yule
            常见的欧氏距离计算：
            from scipy.spatial.distance import cdist
            import numpy as np
            x1 =np.array([(1,3),(2,4),(5,6)])
            x2 =[(3,7),(4,8),(6,9)]
            cdist(x1,x2,metric='euclidean')
            array([[ 4.47213595,  5.83095189,  7.81024968],
                   [ 3.16227766,  4.47213595,  6.40312424],
                   [ 2.23606798,  2.23606798,  3.16227766]])
            解析上述计算过程：结果数组中的第一行数据表示的是x1数组中第一个元素点与x2数组中各个元素点的距离，
            计算两点之间的距离,以点(1,3)与(3,7)点的距离为例：
            np.power((1-3)**2 +(3-7)**2,1/2)
            =4.4721359549995796"""
            cd = cdist(X, centroids)
            """axis=0 表示列方向上的最小值索引，axis=1表示行方向的最小值索引
            a = [[2, 0, 5], [3, 4, 1]]
            b = np.argmin(a, axis=0)
            结果：[0, 0, 1] #在列方向上2<3, 0<4, 1<5
            a = [[2, 0, 5], [3, 4, 1]]
            b = np.argmin(a, axis=1)
            结果：[1, 2] # 在行方向上，第一行0最小，在1号位置，第二行1最小，在2号位置.
            默认axis=none，为list打平后在取，如2维变一维。"""
            labels = np.argmin(cd, axis=1)
            # print("gg", labels)
            labels_history.append(labels)
            # ccc = []
            # for i in range(K):
            #     a = X[labels == i]
            #     b = a.mean(axis=0)
            #     ccc.append(b)
            #     # print(i)
            #     # print(a)
            # print("ccc", ccc)
            # Take mean of points within clusters to find new centroids:
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(K)])
            print("new_centroids", new_centroids)
            centroids_history.append(new_centroids)

            # If old centroids and new centroids no longer change, k-means is complete and end. Otherwise continue
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids

        return centroids, labels, centroids_history, labels_history

    X_mat = X.values
    centroids, labels, centroids_history, labels_history = k_means(X_mat, 3)
    print(labels)
    print(X[['PetalLengthCm', 'PetalWidthCm']])
    c = silhouette_score(X[['PetalLengthCm', 'PetalWidthCm']], labels)
    print("dd", c)


def unsupervised_lesson06_exercise12():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    def pca_principal():
        """协方差实现原理
        a=[1,2,3]
        b=[5,5,6]
        z=np.stack((a,b))
        np.cov(z)
        #[cov(a,a),cov(a,b)]
        #[cov(b,a),cov(b,b)]
        #output:
        #      array([[1.        , 0.5       ],
        #             [0.5       , 0.33333333]])
        代码：
        def cov(x,y):# x为向量，即观测值(样本)向量
            n=len(x)
            x_bar=np.mean(x)
            y_bar=np.mean(y)
            var=np.sum((x-x_bar)*(y-y_bar))/(n-1)
            return var
        print(cov(a,a),cov(a,b),cov(b,b))
        output:  1.0 0.5 0.33333333333333337
        """
        pp = r"D:\ana\envs\py36\mywagtailone\datatables\unsupervited_data\Applied-Unsupervised-Learning-with-Python-master\Lesson04\Exercise11\iris-data.csv"
        df = pd.read_csv(pp)
        # df = pd.read_csv(p, header=None)
        # print(df.head())
        df = df[['Sepal Length', 'Sepal Width']]
        print(df.head())
        print(df.head().values)
        print(df.head().values.T)
        # print(df.cov())
        # print(df.mean())
        # plt.figure(figsize=(10, 7))
        # plt.scatter(df['Sepal Length'], df['Sepal Width'])
        # plt.xlabel('Sepal Length (mm)')
        # plt.ylabel('Sepal Width (mm)')
        # plt.title('Sepal Length versus Width')
        # plt.show()
        data = np.cov(df.values.T)
        print("hh", data)
        eigenvectors, eigenvalues, _ = np.linalg.svd(data, full_matrices=False)
        # eigenvectors, eigenvalues, _ = np.linalg.svd(df.values, full_matrices=False)
        # print(_)
        print(eigenvalues)
        eigenvalues = np.cumsum(eigenvalues)
        eigenvalues /= eigenvalues.max()
        print("/max", eigenvalues)
        print(eigenvectors)
        P = eigenvectors[0]
        print("p", P)
        x_t_p = P.dot(df.values.T)
        print("xtp", x_t_p)
        print("xtp", x_t_p.shape)
        plt.figure(figsize=(10, 7))
        plt.plot(x_t_p)
        plt.title('Principal Component of Selected Iris Dataset');
        plt.xlabel('Sample')
        plt.ylabel('Component Value')
        plt.show()
    pca_principal()
    # pca_principal2()
    p = r"D:\ana\envs\py36\mywagtailone\datatables\unsupervited_data\Applied-Unsupervised-Learning-with-Python-master\Lesson06\Activity12\wine.data"
    df = pd.read_csv(p, header=None)
    labels = df[0]
    # print(df.head())
    del df[0]
    print(df.head())
    """principal component analysis,主成份分析
    1.n_components：这个参数可以帮我们指定希望PCA降维后的特征维度数目。最常用的做法是直接指定降维到的维度数目，此时n_components是一个大于等于1的整数。当然，我们也可以指定主成分的方差和所占的最小比例阈值，让PCA类自己去根据样本特征方差来决定降维到的维度数，此时n_components是一个（0，1]之间的数。当然，我们还可以将参数设置为"mle", 此时PCA类会用MLE算法根据特征的方差分布情况自己去选择一定数量的主成分特征来降维。我们也可以用默认值，即不输入n_components，此时n_components=min(样本数，特征数)

2.copy：类型：bool，True或者False，缺省时默认为True。意义：表示是否在运行算法时，将原始训练数据复制一份。若为True，则运行PCA算法后，原始训练数据的值不会有任何改变，因为是在原始数据的副本上进行运算；若为False，则运行PCA算法后，原始训练数据的值会改，因为是在原始数据上进行降维计算

3.whiten ：判断是否进行白化。所谓白化，就是对降维后的数据的每个特征进行归一化，让方差都为1.对于PCA降维本身来说，一般不需要白化。如果你PCA降维后有后续的数据处理动作，可以考虑白化。默认值是False，即不进行白化

4.svd_solver：即指定奇异值分解SVD的方法，由于特征分解是奇异值分解SVD的一个特例，一般的PCA库都是基于SVD实现的。有4个可以选择的值：{‘auto’, ‘full’, ‘arpack’, ‘randomized’}。randomized一般适用于数据量大，数据维度多同时主成分数目比例又较低的PCA降维，它使用了一些加快SVD的随机算法。 full则是传统意义上的SVD，使用了scipy库对应的实现。arpack和randomized的适用场景类似，区别是randomized使用的是scikit-learn自己的SVD实现，而arpack直接使用了scipy库的sparse SVD实现。默认是auto，即PCA类会自己去在前面讲到的三种算法里面去权衡，选择一个合适的SVD算法来降维。一般来说，使用默认值就够了

5.tol：svd_solver =='arpack'计算的奇异值的公差，float> = 0，可选（默认.0）

6.iterated_power： int> = 0或'auto'，（默认为'auto'）,svd_solver =='随机化'计算出的幂方法的迭代次数

属性：

1.components_ ：特征空间中的主轴，表示数据中最大方差的方向。组件按排序 explained_variance_

2.explained_variance_：它代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分

3.explained_variance_ratio_：它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分

4.singular_values_：每个特征的奇异值，奇异值等于n_components 低维空间中变量的2范数

5.mean_：每个特征的均值

6.n_components_：即是上面输入的参数值

7.n_features_：训练数据中的特征数量

8.n_samples_：训练数据中的样本数

9.noise_variance_：等于X协方差矩阵的（min（n_features，n_samples）-n_components）个最小特征值的平均值

方法：

fit（X [，y]）用X拟合模型。
fit_transform（X [，y]）使用X拟合模型，并在X上应用降维。
get_covariance（）用生成模型计算数据协方差。
get_params（[深]）获取此估计量的参数。
get_precision（）用生成模型计算数据精度矩阵。
inverse_transform（X）将数据转换回其原始空间。
score（X [，y]）返回所有样本的平均对数似然率。
score_samples（X）返回每个样本的对数似然。
set_params（**参数）设置此估算器的参数。
transform（X）对X应用降维。"""
    model_pca = PCA(n_components=6)
    wine_pca = model_pca.fit_transform(df)
    nn = np.sum(model_pca.explained_variance_ratio_)
    print(nn)


def unsupervised_lesson06_activity13():
    import pandas as pd
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    def pca_principal():
        pp = r"D:\ana\envs\py36\mywagtailone\datatables\unsupervited_data\Applied-Unsupervised-Learning-with-Python-master\Lesson06\Activity13\wine.data"
        df = pd.read_csv(pp, header=None)
        labels = df[0]
        del df[0]
        # df = pd.read_csv(pp)
        print(df.head())
        model_pca = PCA(n_components=6)
        wine_pca = model_pca.fit_transform(df)
        wine_pca = wine_pca.reshape((len(wine_pca), -1))

        MARKER = ['o', 'v', '^', ]
        for perp in [1, 5, 20, 30, 80, 160, 320]:
            tsne_model = TSNE(random_state=0, verbose=1, perplexity=perp)
            wine_tsne = tsne_model.fit_transform(wine_pca)
            plt.figure(figsize=(10, 7))
            plt.title(f'Low Dimensional Representation of Wine. Perplexity {perp}')
            for i in range(1, 4):
                selections = wine_tsne[labels == i]
                plt.scatter(selections[:, 0], selections[:, 1], marker=MARKER[i - 1], label=f'Wine {i}', s=30)
                plt.legend()
            plt.show()

    pca_principal()