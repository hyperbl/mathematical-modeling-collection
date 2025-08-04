import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    return mo, np, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 层次分析法（AHP）案例：最佳旅游地选择

    ## 层次结构图

    ![最佳旅游地选择](public/best_resort_selection.png)

    上图是一个递阶层次结构，从上到下分为**目标层**、**准则层**和**方案层**三个层次。

    ## 构造判断矩阵

    从层次结构的准则层内，对于从属于（或影响）上一层的某个因素的子因素，构造判断矩阵，具体的构造原则如下：

    1. 不把所有因素放在一起比较，而是两两相互比较；
    2. 对此时采用相对尺度，以尽可能减少性质不同的各个因素相互比较的困难，以提高准确度；

    Saaty 的 \(1\sim 9\) 标度法：

    | 标度 | 含义 | | 
    | -- | -- | -- |
    | \(1\) | 表示两个因素相比，具有同等重要性 | | 
    | \(3\) | 表示一个因素比另一个因素稍微重要 | |
    | \(5\) | 表示一个因素比另一个因素明显重要 | |
    | \(7\) | 表示一个因素比另一个因素非常重要 | |
    | \(9\) | 表示一个因素比另一个因素极端重要 | |
    | \(2, 4, 6, 8\) | 表示介于上述两者之间的值 | |
    | \(1/n\) | 如果因素 \(i\) 与 \(j\) 比较的判断为 \(a_{ij}\)，则 \(a_{ji} = 1/a_{ij}\) | |

    在本例中，确定判断矩阵如下：
    """
    )
    return


@app.cell(hide_code=True)
def _(pl):
    decision_df = pl.DataFrame(
        data={
            "准则": ["景色", "费用", "饮食", "居住", "旅途"],
            "景色": [1, 2, 1/5, 1/5, 1/3],
            "费用": [1/2, 1, 1/7, 1/7, 1/5],
            "饮食": [5, 7, 1, 2, 3],
            "居住": [5, 7, 1/2, 1, 2],
            "旅途": [3, 5, 1/3, 1/2, 1]

        },
        strict=False
    )

    decision_matrix = decision_df.drop("准则").to_numpy()

    decision_df
    return (decision_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 一致性检验

    ### 一致性的定义

    如果一个正互反（对称）矩阵 \(A = (a_{ij})_{n\times n}\) 满足

    \begin{equation}
        a_{ij}a_{jk} = a_{ik},\quad i, j, k=1,2,\dots n
    \end{equation}

    则称 \(A\) 为一致性判断矩阵，简称一致阵。

    定理：

    1. \(n\) 阶一致阵的唯一非零特征根为 \(n\)；
    2. \(n\) 阶正互反（对称）矩阵 \(A\) 的最大特征根 \(\lambda \geqslant n\)，当前仅当 \(\lambda= n\) 时 \(A\) 为一致阵；

    ### 一致性指标 \(\mathrm{CI}\)

    当判断矩阵不具有一致性，则其最大特征值 \(\lambda_{max} \ne n\)，衡量不一致程度的数量指标称为一致性指标，定义为

    \begin{equation}
        \mathrm{CI} = \frac{\lambda - n}{n - 1} 
    \end{equation}

    由于矩阵 \(A\) 的所有特征值和为 \(n\)，实际上 \(\mathrm{CI}\) 是去除最大特征值后矩阵 \(A\) 的 \(n - 1\) 个特征值的平均值的相反数。

    ### 平均随机一致性指标 \(\mathrm{RI}\)

    一般平均随机一致性指标都已给出，如下表

    | \(n\) | \(1\) | \(2\) | \(3\) | \(4\) | \(5\) | \(6\) | \(7\) | \(8\) | \(9\) |
    | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
    | \(\mathrm{RI}\) | \(0\) | \(0\) | \(0.58\) | \(0.90\) | \(1.12\) | \(1.24\) | \(1.32\) | \(1.41\) | \(1.45\) |

    ### 一致性比率 \(\mathrm{CR}\)

    \begin{equation}
        \mathrm{CR} = \frac{\mathrm{CI}}{\mathrm{RI}}
    \end{equation}

    一般当 \(\mathrm{CR} < 0.1\) 时，认为判断矩阵具有满意的一致性，否则就需要调整判断矩阵，使之具有满意的一致性。

    当判断矩阵具有满意的一致性时，其最大特征值对应的归一化特征向量即可作为该层的权向量。
    """
    )
    return


@app.cell(hide_code=True)
def _(decision_matrix, np):
    RI = np.array([0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45])

    eigen_values, eigen_vectors = np.linalg.eig(decision_matrix)

    # CI
    decision_matrix_CI = (eigen_values.max() - decision_matrix.shape[0]) / (decision_matrix.shape[0] - 1)
    decision_matrix_CI = decision_matrix_CI.real

    # RI
    decision_matrix_RI = RI[decision_matrix.shape[0] - 1]

    # CR
    decision_matrix_CR = decision_matrix_CI / decision_matrix_RI

    if decision_matrix_CR < 0.1:
        print("经计算，本例中给出的判断矩阵具有满意的一致性。")
        weight = eigen_vectors[:, eigen_values.argmax()] / eigen_vectors[:, eigen_values.argmax()].sum()
        weight = weight.real
        print(f"所得权向量为：{weight}")
    else:
        print("经计算，本例中给出的判断矩阵不具有满意的一致性，请调整判断矩阵。")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 多层的情况

    此处省略，主要是 \(\mathrm{CR}\) 的新定义，参考[层次分析法原理及其计算过程详解](https://zhuanlan.zhihu.com/p/266405027)
    """
    )
    return


if __name__ == "__main__":
    app.run()
