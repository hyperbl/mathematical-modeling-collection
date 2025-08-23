import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import os
    import sys
    return mo, np, os, plt, sys


@app.cell
def _(os, sys):
    root_path = os.getcwd()

    assert root_path[-3:] == "MCM", "Please run this code in the root directory."

    example_data_path = os.path.join(root_path, "optimization", "IntelligentOptimization", "example", "data")

    sys.path.append(root_path)

    from optimization.IntelligentOptimization.SAA import SimulatedAnnealingBase 
    return SimulatedAnnealingBase, example_data_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 模拟退火算法示例

    ## TSP 问题

    ### 问题描述

    已知 \(100\) 个目标的经纬度如 `data/Pdata17_1.txt` 所示。我方有一个基地，经度和纬度为 \((70, 40)\)。假设我方飞机的速度为 \(1000\,\mathrm{km/h}\)。我方派一架飞机从基地出发，侦察完所有目标，再返回原来的基地。在每一目标的侦察时间不计，求该架飞机所花费的最少时间（假设我方飞机巡航时间可以充分长）。

    已知地球上用经纬度描述位置的两点 \((x_1, x_2)\) 和 \((x_3, x_4)\) 间的距离为

    \begin{equation*}
        d = R\arccos\bigl[\cos(x_1-x_2)\cos y_1 \cos y_2 + \sin y_1 \sin y_2\bigr]
    \end{equation*}

    ### 问题分析

    这是一个旅行商问题。给我方基地编号为 \(0\)，目标依次编号为 \(1, 2, \cdots, 100 \)，最后我方基地编号为 \(101\)。距离矩阵 \(D = (d_{ij})_{102\times 102} \)，其中 \(d_{ij}\) 表示 \(i, j\) 两点的距离，\(i, j = 0,1,\cdots, 102\)，这里 \(D\) 为实对称矩阵。则问题是从点 \(0\) 出发，走遍所有中间点，到达点 \(101\) 的最短路径。

    ### 求解步骤

    1. 解空间。解空间 \(S\) 可表示为 \(\{0, 1, \cdots, 101\}\) 的所有固定起点和终点的循环排列集合，即

    \begin{equation*}
        S = \bigl\{(\pi_0, \cdots, \pi_{101})\ |\ \pi_0 = 0, \pi_{101} = 101, (\pi_1, \cdots, \pi_{100}) 为 \{1, \cdots, 100\} 的循环排列 \bigr\}
    \end{equation*}

    2. 目标函数。目标函数（或称代价函数）为侦察所有目标的路径长度，即

    \begin{equation*}
        f(\pi_0, \pi_1, \cdots, \pi_{101}) = \sum_{i=0}^{100} d_{\pi_i, \pi_{i+1}} + d_{\pi_{101}, \pi_0}
    \end{equation*}

    其中 \(d_{\pi_{101}, \pi_0} = 0\)。

    3. 新解的产生。对于本问题，采用“交换”的方式：任选两个序号进行交换位置。

    4. 代价函数差。计算前后路径差，并根据接受准则决定是否接受新解。本题中采用的准则：

    \begin{equation*}
        P = \left\{\begin{aligned}
            &1, &\Delta f < 0, \\
            &\exp(-\Delta f / T), &\Delta f \geqslant 0
        \end{aligned}\right.
    \end{equation*}

    6. 降温。利用选定的温度系数进行降温，取新的温度 \(T\) 为 \(\alpha T\)，这里选定 \(\alpha = 0.999\)
    7. 终止条件。用选定的温度 `T_end`， 判断退火是否结束。
    """
    )
    return


@app.cell
def _(example_data_path, np, os):
    raw_data = np.loadtxt(os.path.join(example_data_path, "Pdata17_1.txt"))

    raw_data = np.vstack([
        raw_data[:, i:i+2]
            for i in range(0, raw_data.shape[1], 2)
    ])

    data = np.vstack([
        [70, 40],
        raw_data,
        [70, 40]
    ])

    def distance(p1, p2):
        """
        计算两点间的距离
        """
        R = 6371  # 地球半径，单位为 km
        x1, y1 = p1
        x2, y2 = p2
        d = R * np.arccos(np.clip(
            np.cos(np.radians(x1 - x2)) * np.cos(np.radians(y1)) * np.cos(np.radians(y2)) +
            np.sin(np.radians(y1)) * np.sin(np.radians(y2))
        , -1.0, 1.0))
        return d

    get_distance_matrix = lambda x : np.array([
        [distance(x[i], x[j]) for j in range(x.shape[0])]
            for i in range(x.shape[0])
    ])

    distance_matrix = get_distance_matrix(data)
    return data, distance_matrix


@app.cell
def _(SimulatedAnnealingBase, np):
    class TSPModel(SimulatedAnnealingBase):
        def __init__(self, T_begin, T_end = 1e-30, max_iter = 1000, n_iter = 100, max_stall = 20, seed = 42):
            super().__init__(T_begin, T_end, max_iter, n_iter, max_stall, seed)

        def decrease_temperature(self, T, *args, **kwargs):
            alpha = kwargs.get('alpha', 0.999)
            return alpha * T

        def neighbor(self, x, *args, **kwargs):
            x_new = x.copy()
            idx = self.rng.choice(range(1, x_new.shape[0] - 1), size=2, replace=False)
            i, j = idx
            x_new[[i, j]] = x_new[[j, i]]
            return x_new

        def accept(self, cost_old, cost_new, T, *args, **kwargs):
            delta_cost = cost_new - cost_old
            if delta_cost < 0:
                return True
            else:
                return self.rng.uniform() < np.exp(-delta_cost / T)

        def cost(self, x, *args, **kwargs):
            distance_matrix = kwargs.get('distance_matrix', None)
            if distance_matrix is None:
                breakpoint()
                raise ValueError("Distance matrix must be provided.")
            return np.array([
                distance_matrix[x[i], x[i+1]] for i in range(x.shape[0] - 1)
            ]).sum()
    return (TSPModel,)


@app.cell
def _(TSPModel, data, distance_matrix, np):
    seed = 42
    rng = np.random.default_rng(seed)

    # 不同的参数，结果也不同，可以通过调整参数来获得更好的结果，如使用optuna调整参数
    model = TSPModel(
        T_begin=1,
        T_end=1e-20,
        max_iter=20000,
        n_iter=1000,
        max_stall=20,
        seed=seed,
    )

    x_init = np.hstack([0, rng.permutation(np.arange(1, data.shape[0] - 1)), data.shape[0] - 1])

    x_best = model(x_init, distance_matrix=distance_matrix, alpha=0.999)

    x_best
    return (x_best,)


@app.cell
def _(data, plt, x_best):
    ax = plt.gca()
    ax.plot(data[x_best, 0], data[x_best, 1], marker='o', markersize=5, color='blue')
    ax.plot(data[x_best[0], 0], data[x_best[0], 1], marker='o', markersize=10, color='red', label='Start/End')
    ax.set_title("TSP Path")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
