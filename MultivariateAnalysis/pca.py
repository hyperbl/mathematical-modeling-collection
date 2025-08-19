import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_iris
    import plotly.graph_objs as go
    return PCA, go, load_iris, mo, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 主成分分析

    主成分分析（Principal Component Analysis, PCA）是一种常用的降维技术，用于将高维数据转换为低维数据，同时保留尽可能多的原始数据的方差信息。

    关于原理的讲解，除了教材外还可以参考[B站视频](https://www.bilibili.com/video/BV1E5411E71z/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=06c3cf7df2df48362c07d081443e3363)。

    ## 主成分分析的基本步骤

    假设有 \(n\) 个研究对象，\(m\) 个指标变量 \(X_1, X_2, \ldots, X_m\)，第 \(i\) 个对象第 \(j\) 个指标取值 \(a_{ij}\)，则构成数据矩阵 \(A = (a_{ij})_{n\times m}\).

    主成分分析的基本步骤如下：

    1. 对原来的 \(m\) 个指标进行**标准化**，得到标准化的指标变量

    \begin{equation*}
        y_j = \frac{x_j - \mu_j}{s_j},\quad j=1,2,\cdots, m,
    \end{equation*}

    其中，

    \begin{align*}
        \mu_j &= \frac{1}{n}\sum_{i=1}^n a_{ij}, \\
        s_j &= \sqrt{\frac{1}{n-1}\sum_{i=1}^n (a_{ij} - \mu_j)^2}.
    \end{align*}

    对应地，得到标准化的数据矩阵 \(B = (b_{ij})_{n\times m}\)，其中

    \begin{equation*}
        b_{ij} = \frac{a_{ij} - \mu_j}{s_j},\quad i=1,2,\cdots,n; j=1,2,\cdots,m.
    \end{equation*}

    2. 根据标准化的数据矩阵 \(B\) 计算样本相关系数（协方差）矩阵 \(R = (r_{ij})_{m\times m}\)，其中

    \begin{equation*}
        r_{ij} = \frac{1}{n-1}\sum_{k=1}^n b_{ki}b_{kj},\quad i,j=1,2,\cdots,m.
    \end{equation*}

    3. 求相关系数矩阵 \(R\) 的特征值 \(\lambda_1\geqslant \lambda_2\geqslant \ldots\geqslant \lambda_m\) 及其对应的标准正交化特征向量 \(u_1, u_2, \cdots, u_m\)，其中 \(u_j = (u_{1j}, u_{2j}, \cdots, u_{mj})^T\)，\(j=1,2,\cdots,m\)，由特征向量组成 \(m\) 个新的指标变量：

    \begin{equation*}
        \left\{\begin{aligned}
            F_1 &= u_{11}y_1 + u_{21}y_2 + \cdots + u_{m1}y_m, \\
            F_2 &= u_{12}y_1 + u_{22}y_2 + \cdots + u_{m2}y_m, \\
            &\ \vdots \\
            F_m &= u_{1m}y_1 + u_{2m}y_2 + \cdots + u_{mm}y_m.
        \end{aligned}\right.
    \end{equation*}

    式中，\(F_j\) 称为第 \(j\) 个主成分，\(u_{ij}\) 称为第 \(j\) 个主成分在第 \(i\) 个指标上的**载荷**。

    4. 计算主成分贡献率和累积贡献率，主成分 \(F_j\) 的贡献率为

    \begin{equation*}
        w_j = \frac{\lambda_j}{\sum_{i=1}^m \lambda_i},\quad j=1,2,\cdots,m.
    \end{equation*}

    前 \(i\) 个主成分的累积贡献率为

    \begin{equation*}
        W_i = \sum_{j=1}^i w_j = \frac{\sum_{j=1}^i \lambda_j}{\sum_{j=1}^m \lambda_j},\quad i=1,2,\cdots,m.
    \end{equation*}

    一般取累积贡献率达到 \(85\%\) 以上的前 \(k\) 个主成分作为综合指标。

    5. 最后利用得到的主成分 \(F_1, F_2, \cdots, F_k\) 分析问题，或者继续评价、回归、聚类等其他建模.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## PCA 在 Iris 数据集上的应用

    代码源自 `sklearn` 官方示例。其任务是将鸢尾花数据集降维到三维，并绘制散点图。

    ### 获取数据
    """
    )
    return


@app.cell
def _(load_iris):
    iris = load_iris(as_frame=True)
    iris.frame["target"] = iris.target_names[iris.target]
    return (iris,)


@app.cell(hide_code=True)
def _(iris_pairplot_ax, mo):
    mo.md(
        rf"""
    ### 原始数据可视化

    {mo.as_html(iris_pairplot_ax)}
    """
    )
    return


@app.cell(hide_code=True)
def _(iris, sns):

    iris_pairplot_ax = sns.pairplot(iris.frame, hue="target")
    return (iris_pairplot_ax,)


@app.cell(hide_code=True)
def _(iris_ax, mo):
    mo.md(
        rf"""
    ### PCA 处理及结果散点图

    {mo.as_html(iris_ax)}

    下面是交互式图表
    """
    )
    return


@app.cell(hide_code=True)
def _(go, iris, iris_X_reduced, mo):

    iris_fig_go = go.Figure()
    for i, name in enumerate(iris.target_names):
        mask = iris.target == i
        iris_fig_go.add_trace(go.Scatter3d(
            x=iris_X_reduced[mask, 0],
            y=iris_X_reduced[mask, 1],
            z=iris_X_reduced[mask, 2],
            mode='markers',
            name=name,
            marker=dict(size=2)
        ))
    iris_fig_go.update_layout(
        title="First three PCA dimensions of Iris dataset",
        scene=dict(
            xaxis_title="1st eigenvector",
            yaxis_title="2nd eigenvector",
            zaxis_title="3rd eigenvector"
        )
    )

    mo.ui.plotly(iris_fig_go)
    return


@app.cell
def _(PCA, iris, plt):
    iris_fig = plt.figure(1, figsize=(8, 6))
    iris_ax = iris_fig.add_subplot(111, projection="3d", elev=-150, azim=110)

    iris_X_reduced = PCA(n_components=3).fit_transform(iris.data)
    iris_scatter = iris_ax.scatter(
        iris_X_reduced[:, 0],
        iris_X_reduced[:, 1],
        iris_X_reduced[:, 2],
        c=iris.target,
        s=40,
    )

    iris_ax.set(
        title="First three PCA dimensions of Iris dataset",
        xlabel="1st eigenvector",
        ylabel="2nd eigenvector",
        zlabel="3rd eigenvector",
    )

    iris_ax.xaxis.set_ticklabels([])
    iris_ax.yaxis.set_ticklabels([])
    iris_ax.zaxis.set_ticklabels([])

    iris_legend1 = iris_ax.legend(
        iris_scatter.legend_elements()[0],
        iris.target_names.tolist(),
        loc="upper right",
        title="Classes",
    )
    _ = iris_ax.add_artist(iris_legend1)
    return iris_X_reduced, iris_ax


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### PCA 在综合评价中的应用

    一般步骤如下：

    1. 若各指标的属性不同（极大型、极小型），则将原始数据矩阵 \(A = (a_{ij})_{n\times m}\) 统一趋势化，得到属性一致的的指标数据矩阵 \(B\).
    2. 计算 \(B\) 的协方差矩阵 \(\Sigma\) 或相关系数矩阵 \(R\)（当 \(B\) 的量纲不同或协方差矩阵 \(\Sigma\) 主对角线元素相差悬殊时，宜采用相关系数矩阵 \(R\)）.
    3. 计算 \(R = (r_{ij})_{m\times m}\) 的特征值 \(\lambda_1\geqslant \lambda_2\geqslant \cdots\geqslant \lambda_m\) 与相应的特征向量 \(u_1, u_2, \cdots, u_m\).
    4. 根据特征值计算累积贡献率，确定主成分的个数，而特征向量 \(u_i\) 就是第 \(i\) 个主成分的载荷向量.
    5. 计算主成分的得分矩阵，若选定前 \(k\) 个主成分，则得分矩阵为

    \begin{equation*}
        F = B\cdot [u_1, u_2, \cdots, u_k]
    \end{equation*}

    6. 计算综合评价值 \(Z = FW\)，其中 \(W = [w_1, w_2,\cdots, w_k]^T\)，这里 \(w_i=\frac{\lambda_i}{\sum_{j=1}^k \lambda_j}, \; i=1,2,\cdot,k\) 是第 \(i\) 主成分的贡献率。由此即可进行排名。
    """
    )
    return


if __name__ == "__main__":
    app.run()
