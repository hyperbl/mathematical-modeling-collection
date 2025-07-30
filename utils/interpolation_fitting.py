import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 插值与拟合示例

    ## 插值

    ### Language 插值与牛顿插值

    缺点：只能计算单点处的值

    ### 样条插值

    主流方案，有相关 Python 库函数支持，参考 [SciPy 教程](https://docs.scipy.org/doc/scipy/tutorial/interpolate.html#tutorial-interpolate)。

    ## 一维插值示例

    ### 题目描述

    在一天 \(24\ \mathrm{h}\) 内，从零点开始每间隔 \(2\ \mathrm{h}\) 测得的环境温度（\(\degree\mathrm{C}\)） 如下：

    | 时间 | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 |
    | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
    | 温度 | 12 | 9 | 9 | 10 | 18 | 24 | 28 | 27 | 25 | 20 | 18 | 15 | 13 |

    要求分别对其进行分段线性插值和三次样条插值，并画出插值曲线。

    ### 运行结果
    """
    )
    return


@app.cell(hide_code=True)
def _(interpolate_1D, mo):
    mo.md(rf"""{mo.as_html(interpolate_1D()[0])}""")
    return


@app.cell(hide_code=True)
def _(np, plt):
    def interpolate_1D():
        from scipy.interpolate import CubicSpline
        # 准备数据
        time = np.arange(0, 25, 2)
        temperture = np.array([12, 9, 9, 10, 18, 24, 28, 27, 25, 20, 18, 15, 13])

        # 插值点
        xnew = np.linspace(0, 24, 1001)

        # 分段线性插值
        ynew_interp = np.interp(xnew, time, temperture)

        # 三次样条插值
        ynew_CubicSpline = CubicSpline(time, temperture)(xnew)

        # 可视化
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))

        axes[0].set_title("分段线性插值与样条插值曲线")

        axes[0].plot(xnew, ynew_interp, label="分段线性插值")
        axes[1].plot(xnew, ynew_CubicSpline, label="样条插值")

        for ax in axes:
            ax.plot(time, temperture, "o", label="data")
            ax.set_xlabel("time")
            ax.set_ylabel("temperture")
            ax.legend(loc="best")

        plt.tight_layout()

        return axes
    return (interpolate_1D,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 二维插值示例

    ### 数据点规则分布

    #### 题目描述

    已知平面区域 \( 0 \leqslant x \leqslant 1400,\ 0 \leqslant y \leqslant 1200 \) 的高程数据表（已提供在 example_data 文件夹下），求该区域地表面积的近似值，并用插值数据画出该区域的等高线图和三维表面图
    """
    )
    return


@app.cell(hide_code=True)
def _(interpolate_regular2D, mo):
    _res=interpolate_regular2D()

    mo.md(
        rf"""
    #### 运行结果
    地表面积的近似值为 {_res[0]:.4e},
    等高线图和三维表面图如下：
    {mo.as_html(_res[1])}
    """
    )
    return


@app.cell(hide_code=True)
def _(np, plt):
    def interpolate_regular2D():
        from scipy.interpolate import RectBivariateSpline

        # 导入高程数据
        z = np.loadtxt("utils/example_data/interp2D.txt", delimiter=",")

        # 对数据进行处理
        # 由于数据是从左下角开始的，需要进行翻转和
        z = np.flipud(z)

        # 坐标数据
        x = np.arange(0, 1401, 100)
        y = np.arange(0, 1201, 100)

        # 插值点
        xnew = np.linspace(0, 1400, 141)
        ynew = np.linspace(0, 1200, 121)

        # 使用 RectBivariateSpline 进行二维插值
        r = RectBivariateSpline(x, y, z.T)
        znew = r(xnew, ynew)

        # 计算该区域表面积的近似值
        xx, yy = np.meshgrid(xnew, ynew, indexing="ij")

        # 取出每个小格子的四个顶点
        p1 = np.stack(
            [xx[:-1, :-1], yy[:-1, :-1], znew[:-1, :-1]], axis=-1
        )  # 左上
        p2 = np.stack([xx[1:, :-1], yy[1:, :-1], znew[1:, :-1]], axis=-1)  # 右上
        p3 = np.stack([xx[:-1, 1:], yy[:-1, 1:], znew[:-1, 1:]], axis=-1)  # 左下
        p4 = np.stack([xx[1:, 1:], yy[1:, 1:], znew[1:, 1:]], axis=-1)  # 右下

        # 通过三角形计算面积
        area1 = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1), axis=-1)
        area2 = 0.5 * np.linalg.norm(np.cross(p2 - p4, p3 - p4), axis=-1)
        area = np.sum(area1 + area2)

        # 等高线图和三维表面图
        ax0 = plt.subplot(121)
        ax1 = plt.subplot(122, projection="3d")

        # 等高线图
        contour = ax0.contour(xx, yy, znew)
        ax0.clabel(contour)
        ax0.set_xlabel("$x$")
        ax0.set_ylabel("$y$", rotation=0)
        ax0.set_title("等高线图")

        # 三维表面图
        ax1.plot_surface(xx, yy, znew, cmap="viridis")
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$y$")
        ax1.set_zlabel("$z$")
        ax1.set_title("三维表面图")

        return (area, ax0)
    return (interpolate_regular2D,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 数据点不规则分布

    #### 题目描述

    在某海域测得一些点 \((x, y)\) 处的水深 \(z\) 由下表给出，试画出海底区域的地形和等高线图。

    | x | 129 | 140 | 103.5 | 88 | 185.5 | 195 | 105 | 157.5 | 107.5 | 77 | 81 | 162 | 162 | 117.5 |
    | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
    | y | 7.5 | 141.5 | 23 | 147 | 22.5 | 137.5 | 85.5 | -6.5 | -81 | 3 | 56.5 | -66.5 | 84 | -33.5 |
    | z | 4 | 8 | 6 | 8 | 6 | 8 | 8 | 9 | 9 | 8 | 8 | 9 | 4 | 9 |
    """
    )
    return


@app.cell(hide_code=True)
def _(interpolate_irregular2D, mo):
    mo.md(
        rf"""
    #### 运行结果

    {mo.as_html(interpolate_irregular2D())}
    """
    )
    return


@app.cell(hide_code=True)
def _(np, plt):
    def interpolate_irregular2D():
        from scipy.interpolate import NearestNDInterpolator

        # 数据
        x = np.array([
            129, 140, 103.5, 88, 185.5, 195, 105, 157.5, 107.5, 77, 81, 162, 162, 117.5
        ])
        y = np.array([
            7.5, 141.5, 23, 147, 22.5, 137.5, 85.5, -6.5, -81, 3, 56.5, -66.5, 84, -33.5
        ])
        z = -np.array([
            4, 8, 6, 8, 6, 8, 8, 9, 9, 8, 8, 9, 4, 9
        ])

        xy = np.vstack([x, y]).T

        # 插值点
        xx = np.linspace(x.min(), x.max(), 100)
        yy = np.linspace(y.min(), y.max(), 100)
        xnew, ynew = np.meshgrid(xx, yy)

        # 采用最邻近点插值
        interp = NearestNDInterpolator(xy, z)

        znew = interp(xnew, ynew)

        # 可视化
        plt.figure(figsize=(8, 4))

        ax0 = plt.subplot(121, projection="3d")
        ax1 = plt.subplot(122)

        # 地形图
        ax0.plot_surface(xnew, ynew, znew, cmap="viridis")
        ax0.set_xlabel("$x$")
        ax0.set_ylabel("$y$")
        ax0.set_zlabel("$z$")
        ax0.set_title("地形图")

        contour = ax1.contour(xnew, ynew, znew, 8)
        ax1.clabel(contour)
        ax1.set_xlabel("$x$")
        ax1.set_ylabel("$y$")
        ax1.set_title("等高线图")

        return ax0
    return (interpolate_irregular2D,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 拟合

    主要是最小二乘拟合。

    对于线性的情况，有正规方程法

    对于非线性的情况，Python 提供了 `numpy.polyfit` 和 `scipy.optimize.curve_fit` 等函数。

    也可以使用机器学习库中的回归模型。

    此处就省略了，例子可以从 mcm2019 项目中找到（ sklearn 实现）。
    """
    )
    return


if __name__ == "__main__":
    app.run()
