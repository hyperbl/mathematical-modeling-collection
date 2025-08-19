import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    from scipy.stats import zscore
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import cdist, squareform
    return cdist, dendrogram, linkage, mo, np, pl, plt, squareform, zscore


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 聚类分析

    聚类分析又称为群分析，是对多个样本（或指标）进行定量分析的一种多元统计分析方法。其中，

    - 对样本进行分类称为 Q 型聚类分析
    - 对指标进行分类称为 R 型聚类分析
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Q 型聚类分析

    ### 样本的相似性度量

    一般用**距离**来度量样本点间的相似程度。

    距离的定义

    记 \(\mathit \Omega \) 是样本点集，其中每个样本点由 \(p\) 个变量描述， 距离 \(d(\cdot,\cdot)\) 可以定义为 \(\mathit \Omega \times \mathit \Omega \to \mathbb R^+ \) 的一个函数，满足以下条件：

    1. \(d(x, y)\geqslant 0, x, y \in \mathit \Omega \)
    2. \(d(x, y) = 0\) 当且仅当 \(x = y\)
    3. \(d(x, y) = d(y, x), x, y, \in \mathit \Omega \)
    4. \(d(x, y) \leqslant d(x, z) + d(z, y), x, y, z \in \mathit \Omega \)

    下面列举了聚类分析中常用的距离：

    **闵氏距离**（Minkowski Distance）

    \begin{equation*}
        d_q(x, y) = \bigl[\sum_{k=1}^{p} |x_k - y_k|^q \bigr]^\frac{1}{q}, q > 0 
    \end{equation*}

    当 \(q = 1, 2\) 或 \(q\to+\infty\) 时，分别得到：

    1. 曼哈顿距离（Manhattan Distance）或城市街区距离（City Block Distance）：

    \begin{equation*}
       d_1(x, y) = \sum_{k=1}^{p} |x_k - y_k|
    \end{equation*}

    2. 欧氏距离（Euclidean Distance）：

    \begin{equation*}
        d_2(x, y) = \sqrt{\sum_{k=1}^{p} |x_k - y_k|^2}
    \end{equation*}

    3. 切比雪夫距离（Chebyshev Distance）：

    \begin{equation*}
        d_\infty(x, y) = \max_{k=1,\ldots,p} |x_k - y_k|
    \end{equation*}

    在采用 Minkowski 距离时注意：

    - 一定要采用**相同量纲**的变量
    - 尽可能避免变量的**多重相关性**（Multicollinearity）

    **马氏距离**（Mahalanobis Distance）

    \begin{equation*}
        d_M(x, y) = \sqrt{(x - y)^T S^{-1} (x - y)}
    \end{equation*}

    式中：\(x\)，\(y\) 来自 \(p\) 维总体 \(Z\) 的样本观测值；\(S\) 是 \(Z\) 的样本协方差矩阵，实际中往往利用样本协方差矩阵的估计值 \(S\) 来代替。

    马氏距离对于样本点的量纲不敏感。

    ### 类与类间的相似性度量

    如果有两个样本类 \(G_1\) 和 \(G_2\)，可以用下面的方法度量它们之间的距离：

    1. 最短距离（Single Linkage）

    \begin{equation*}
        d(G_1, G_2) = \min_{x \in G_1, y \in G_2} d(x, y)
    \end{equation*}

    它的直观意义是两个样本类中最近的两个样本点之间的距离。

    2. 最长距离（Complete Linkage）

    \begin{equation*}
        d(G_1, G_2) = \max_{x \in G_1, y \in G_2} d(x, y)
    \end{equation*}

    它的直观意义是两个样本类中最远的两个样本点之间的距离。

    3. 重心法（Centroid Linkage）

    \begin{equation*}
        d(G_1, G_2) = d(\bar x, \bar y)
    \end{equation*}

    式中，\(\bar x\) 和 \(\bar y\) 分别是样本类 \(G_1\) 和 \(G_2\) 的重心（Centroid）。

    4. 类平均法（Average Linkage）

    \begin{equation*}
        d(G_1, G_2) = \frac{1}{|G_1||G_2|} \sum_{x \in G_1} \sum_{y \in G_2} d(x, y)
    \end{equation*}    

    它等于两个样本类中所有样本点之间的平均距离。

    5. Ward 法（Ward's Method）

    Ward 法是基于最小化样本类间的平方和（Sum of Squares）来度量样本类间的距离。

    \begin{equation*}
        d(G_1, G_2) = D_{12} - D_1 - D_2
    \end{equation*}

    其中，\(D_{12}\) 是样本类 \(G_1\) 和 \(G_2\) 合并后的平方和，\(D_1\) 和 \(D_2\) 分别是样本类 \(G_1\) 和 \(G_2\) 的平方和，即

    \begin{align*}
        D_{12} &= \sum_{x \in G_1 \cup G_2} (x - \bar x_{12})^T (x - \bar x_{12}) \\
        D_1 &= \sum_{x \in G_1} (x - \bar x_1)^T (x - \bar x_1) \\
        D_2 &= \sum_{x \in G_2} (x - \bar x_2)^T (x - \bar x_2)
    \end{align*}

    式中

    \begin{align*}
        \bar x_{12} &= \frac{1}{|G_1| + |G_2|} \sum_{x \in G_1 \cup G_2} x \\
        \bar x_1 &= \frac{1}{|G_1|} \sum_{x \in G_1} x \\
        \bar x_2 &= \frac{1}{|G_2|} \sum_{x \in G_2} x
    \end{align*}

    ### 聚类图

    设 \(\mathit \Omega = \{w_1, w_2, \cdots, w_n\}\)，聚类图的生成步骤如下：

    1. 计算 \(n\) 个样本点两两之间的距离 \(d_{ij} = d(w_i, w_j)\)，得到距离矩阵 \(D=(d_{ij})_{n\times n}\)。
    2. 首先构造 \(n\) 个类，每一个类包含一个样本点，每一类的平台高度均为 \(0\)。
    3. 合并距离最近的两个类为新类，并且以这两类间的距离值作为聚类图中的平台高度。
    4. 计算新类与当前各类的距离，若类的个数已经等于 \(1\)，则转入步骤 5，否则回到步骤 3。
    5. 画聚类图。
    6. 决定类的个数和类。

    显然，采用不同的距离，得到的聚类结果也会不同。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## R 型聚类分析

    ### 变量相似性度量

    1. **相关系数**。记变量 \(x_j\) 的取值 \((x_{1j}, x_{2j}, \cdots, x_{nj})^T \in \mathbb R^n (j=1,2,\cdots, m)\)。则可以用两变量 \(x_j\) 和 \(x_k\) 的样本相关系数作为它们的相似性度量，即

    \begin{equation*}
        r_{jk} = \frac{\sum_{i=1}^{n} (x_{ij} - \bar x_j)(x_{ik} - \bar x_k)}{\sqrt{\sum_{i=1}^{n} (x_{ij} - \bar x_j)^2 \sum_{i=1}^{n} (x_{ik} - \bar x_k)^2}}
    \end{equation*}

    在对变量进行聚类分析时，利用相关系数矩阵是最多的。

    2. **夹角余弦**。也可以直接利用两变量 \(x_j\) 和 \(x_k\) 的夹角余弦 \(r_{jk}\) 来定义它们的相似性度量，有

    \begin{equation*}
        r_{jk} = \frac{\sum_{i=1}^{n} x_{ij} x_{ik}}{\sqrt{\sum_{i=1}^{n} x_{ij}^2 \sum_{i=1}^{n} x_{ik}^2}}
    \end{equation*}

    不管怎样，各种定义的相似性度量均应具有以下两个性质：

    1. \(|r_{jk}| \leqslant 1\)，对于一切 \(j, k\)。
    2. \(r_{jk} = r_{kj}\)，对于一切 \(j, k\)。

    ### 变量聚类法

    在变量聚类问题中，常用的有最长距离法、最短距离法等。

    1. 最长距离法。在最长距离法中，两个变量类 \(G_1\) 和 \(G_2\) 之间的距离定义为

    \begin{equation*}
        d(G_1, G_2) = \max_{x_j \in G_1, x_k \in G_2} \{ d_{jk}\}
    \end{equation*}

    式中；\(d_{jk} = 1-|r_{jk}|\) 或 \(d_{jk}^2 = 1 - r_{jk}^2\)，这时，\(d(G_1, G_2)\) 与两类中的变量相关性最小的两个变量之间的相似性度量值有关。

    2. 最短距离法。在最短距离法中，两个变量类 \(G_1\) 和 \(G_2\) 之间的距离定义为

    \begin{equation*}
        d(G_1, G_2) = \min_{x_j \in G_1, x_k \in G_2} \{ d_{jk}\}
    \end{equation*}

    式中： \(d_{jk} = 1-|r_{jk}|\) 或 \(d_{jk}^2 = 1 - r_{jk}^2\)，这时，\(d(G_1, G_2)\) 与两类中的变量相关性最大的两个变量之间的相似性度量值有关。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 聚类分析案例——我国各地区普通高等教育发展状况分析

    ### 符号说明

    | 符号 | 含义 |
    | -- | -- |
    | \(x_1\) | 每百万人口高等院校数 |
    | \(x_2\) | 每 \(10\) 万人口高等院校毕业生数 |
    | \(x_3\) | 每 \(10\) 万人口高等院校招生数 |
    | \(x_4\) | 每 \(10\) 万人口高等院校在校生数 |
    | \(x_5\) | 每 \(10\) 万人口高等院校教职工数 |
    | \(x_6\) | 每 \(10\) 万人口高等院校专职教师数 |
    | \(x_7\) | 高级职称占专职教师的比例 |
    | \(x_8\) | 平均每所高等院校的在校生数 |
    | \(x_9\) | 国家财政预算内普通高教经费占国内生产总值的比例 |
    | \(x_10\) | 生均教育经费 |

    ### 数据准备

    从 [example_data 文件夹](./example_data) 中读取数据文件 `development_of_higher_education.xlsx`。其前几行如下所示：
    """
    )
    return


@app.cell(hide_code=True)
def _(pl):
    development_of_higher_education_df = pl.read_excel("MultivariateAnalysis/example_data/development_of_higher_education.xlsx")

    development_of_higher_education_matrix = development_of_higher_education_df \
        .drop("地区") \
        .to_numpy()

    development_of_higher_education_df.head()
    return (
        development_of_higher_education_df,
        development_of_higher_education_matrix,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 对指标进行 R 型聚类

    首先对每个变量（指标）进行标准化处理，使得每个变量的均值为 \(0\)，方差为 \(1\)。变量间相似性度量采用相关系数，类间相似性度量的计算采用类平均法。

    得到的相关系数矩阵如下所示：
    """
    )
    return


@app.cell(hide_code=True)
def _(
    development_of_higher_education_df,
    development_of_higher_education_matrix,
    np,
    pl,
    zscore,
):

    # 对指标进行标准化处理
    norm_dohe_matrix = zscore(development_of_higher_education_matrix, axis=0)

    # 计算相关系数矩阵，精确到小数点后四位
    corrcoef_dohe_matrix =  np.corrcoef(norm_dohe_matrix, rowvar=False).round(4)

    corrcoef_dohe_df = pl.DataFrame(corrcoef_dohe_matrix, 
                                     schema=development_of_higher_education_df.columns[1:],
                                     orient="row")

    corrcoef_dohe_df
    return (corrcoef_dohe_matrix,)


@app.cell(hide_code=True)
def _(dohe_R_ax, mo):
    mo.md(
        rf"""
    得到的聚类图如下所示：

    {mo.as_html(dohe_R_ax)}
    """
    )
    return


@app.cell(hide_code=True)
def _(
    corrcoef_dohe_matrix,
    dendrogram,
    development_of_higher_education_df,
    linkage,
    plt,
):


    # 按照类平均法聚类
    dohe_R_Z = linkage(corrcoef_dohe_matrix, method='average')

    dendrogram(dohe_R_Z, labels=development_of_higher_education_df.columns[1:], 
            leaf_font_size=10, color_threshold=0.2)

    dohe_R_ax = plt.gca()
    return (dohe_R_ax,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    由此可以从 \(10\) 个指标中选出 \(6\) 个指标用以后续的分析：

    - \(x_1\)：每百万人口高等院校数
    - \(x_2\)： 每 \(10\) 万人口高等院校毕业生数
    - \(x_7\)： 高级职称占专职教师的比例
    - \(x_8\)： 平均每所高等院校的在校生数
    - \(x_9\)： 国家财政预算内普通高教经费占国内生产总值的比例
    - \(x_{10}\)： 生均教育经费
    """
    )
    return


@app.cell(hide_code=True)
def _(dohe_Q_ax, mo):
    mo.md(
        rf"""
    ### 对样本进行 Q 型聚类分析

    首先对每个变量的数据分别进行标准化处理，样本间相似性采用欧式距离度量，类间距离的计算采用类平均法。

    得到的聚类图如下所示：

    {mo.as_html(dohe_Q_ax)}
    """
    )
    return


@app.cell(hide_code=True)
def _(
    cdist,
    dendrogram,
    development_of_higher_education_df,
    linkage,
    plt,
    squareform,
    zscore,
):

    # 选出指标
    selected_dohe_matrix = development_of_higher_education_df.select([
        f"x_{i}" for i in [1, 2, 7, 8, 9, 10]
    ]).to_numpy()

    # 对样本进行标准化处理
    norm_dohe_Q_matrix = zscore(selected_dohe_matrix, axis=0)

    # 样本间相似性
    dohe_Q_Y = cdist(norm_dohe_Q_matrix, norm_dohe_Q_matrix, metric='euclidean')

    # 转为压缩距离向量
    dohe_Q_Y_condensed = squareform(dohe_Q_Y)

    # 按照类平均法聚类
    dohe_Q_Z = linkage(dohe_Q_Y_condensed, method='average')

    dendrogram(dohe_Q_Z, leaf_font_size=10, color_threshold=0)

    dohe_Q_ax = plt.gca()
    return (dohe_Q_ax,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## K 均值聚类法

    上文的聚类方法是层次聚类方法，在样本的数量很大时计算量较大，下面介绍 K 均值聚类法。

    算法的基本思想是假定样本集中的全部样本可分为 \(C\) 类，并选定 \(C\) 个初始聚类中心，然后根据最小距离原则将每个样本分配到某一类中，之后不断迭代计算各类的聚类中心，并根据新的聚类中心调整聚类情况，直到迭代收敛或聚类中心不再改变。

    K 均值聚类算法最后将总样本 \(G\) 划分为 \(C\) 个子类：\(G_1, G_2, \cdots, G_C\)，它们满足下面条件：

    1. \(G_1\cup G_2\cup \cdots\cup G_C = G\)；
    2. \(G_i\cap G_j = \mathbb\empty\ (1 \leqslant i < j \leqslant C)\)；
    3. \(G_i \ne \mathbb\empty,\ G_i \ne G\ (1 \leqslant i \leqslant C\)。

    设 \(m_i\ (i=1,2,\cdots C)\) 为 \(C\) 个聚类中心，记

    \begin{equation*}
        J_e = \sum_{i=1}^{C}\sum_{\omega \in G_i} || \omega - m_i||^2
    \end{equation*}

    使 \(J_e\) 最小的解是误差平方和准则下的最优结果。

    ### 最佳簇族 \(k\) 值的确定

    1. 簇内离差平方和拐点法。在不同的 \(k\) 值下计算簇内离差平方和，然后通过可视化的方法找到“拐点”所对应的 \(k\) 值：当斜率由大突然变小时，并且之后的斜率变化缓慢，则认为突然变换的点就是寻找的目标点。
    2. 轮廓系数法。核心思想：如果数据集被分割为理想的 \(k\) 个簇，那么对应的的簇内样本会很密集，而簇间样本会很分散。有关轮廓系数的计算，可以通过 `sklearn` 提供的函数实现。

    使用 K 均值聚类的注意点：

    1. 聚类前必须指定具体的簇族 \(k\) 值；
    2. 对原始数据集做必要的标准化处理；

    以下以鸢尾花（Iris）数据集为例，演示 K 均值聚类的过程。

    """
    )
    return


@app.cell(hide_code=True)
def _(iris_ax, mo):
    mo.md(
        rf"""
    最终结果如下所示：

    {mo.as_html(iris_ax)}
    """
    )
    return


@app.cell(hide_code=True)
def _(plt):
    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans

    # 获取数据集
    iris_datasets = load_iris()

    # 获取特征数据和目标数据
    iris_data = iris_datasets.data
    iris_target = iris_datasets.target

    # 获取特征名称和目标名称
    iris_feature_names = iris_datasets.feature_names
    iris_target_names = iris_datasets.target_names

    # 声明 KMeans 聚类模型
    iris_model = KMeans(n_clusters=3, random_state=42)

    iris_model.fit(iris_data)

    # 获取预测的聚类标签
    iris_label = iris_model.labels_

    # 可视化展示,
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(iris_data[:, 2], iris_data[:, 3], c=iris_label, cmap='viridis', marker='o', edgecolor='k')
    plt.title('K-Means Clustering of Iris Dataset')
    plt.xlabel(iris_feature_names[2])
    plt.ylabel(iris_feature_names[3])
    plt.colorbar(label='Cluster Label')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.scatter(iris_data[:, 2], iris_data[:, 3], c=iris_target, cmap='viridis', marker='o', edgecolor='k')
    plt.title('Iris Dataset with True Labels')
    plt.xlabel(iris_feature_names[2])
    plt.ylabel(iris_feature_names[3])
    plt.colorbar(ticks=[0, 1, 2], label='True Label')
    plt.grid()

    iris_ax = plt.gca()
    return (iris_ax,)


if __name__ == "__main__":
    app.run()
