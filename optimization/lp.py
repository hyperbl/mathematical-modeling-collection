import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import cvxpy as cp
    return cp, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #  线性规划示例

    本示例只提供问题和相关代码，不涉及模型的建立和后续分析

    ## 实数线性规划

    符号说明：

    - \(s_i\): 第 \(i\) 种投资项目，\(i = 0, 1, 2, \cdots, n\)
    - \(r_i\): 平均收益率 average_yield
    - \(p_i\): 交易费率 transaction_fee
    - \(q_i\): 风险损失率 risk_loss
    - \(x_i\): 第 \(i\) 种投资项目投入的资金 captial_investment
    - \(\lambda\): 投资偏好系数 investment_preference
    - \(M\): 投资总额 total_investment

    建立数学模型如下：

    \begin{align}
        &\min\quad \lambda\left\{\max_{0\leqslant i\leqslant n}\{q_ix_i\}\right\} - (1-\lambda)\sum_{i=0}^n(r_i-p_i)x_i \\
        &s.t.\quad \left\{\begin{aligned}
            &\sum_{i=0}^n (1 + p_i)x_i = M \\
            & x_i \geqslant 0,\quad i=0,1,\cdots,n
        \end{aligned}\right.
    \end{align}

    在问题中，\(n = 4, r_i, p_i, q_i, u_i\) 均已知，\(\lambda\) 和 \(M\) 为事先给定的参数。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    slider_investment_preference = mo.ui.slider(
        label=r"投资偏好系数 $\lambda$: ",
        start=0.0,
        stop=1.0,
        value=0.5,
        step=0.01,
        show_value=True,
    )

    slider_total_investment = mo.ui.slider(
        label=r"投资总额 $M$: ",
        start=1000.0,
        stop=10000.0,
        value=5000.0,
        step=100.0,
        show_value=True
    )

    mo.md(
        rf"""
    模型求解：

    先确定 \(\lambda\) 和 \(M\)：

    {slider_investment_preference}

    {slider_total_investment}
    """
    )
    return slider_investment_preference, slider_total_investment


@app.cell(hide_code=True)
def _(mo, solve_lp_scipy, solve_lp_scipy_plot):
    mo.md(
        rf"""
    使用 Scipy 的 `linprog` 函数得到的各项目投入资金安排分别为

    {solve_lp_scipy("x")}

    目标函数的值为

    {solve_lp_scipy("obj")}

    通过交互式分析可以看到，当 \( \lambda < 0.76 \) 时，投资安排几乎不会变化；
    而 \( 0.76 < \lambda < 1\) 时，投资安排会随着 \( \lambda \) 的改变逐渐变化；
    但与投资总额 \( M \) 无关，这与经验是一致的。

    下图体现了不同 \( \lambda \) 下目标函数的变化。

    {mo.as_html(solve_lp_scipy_plot())}

    可见该模型在参数较小时几乎不变，较大时变化比较大。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, solve_lp_cvxpy, solve_lp_cvxpy_plot):
    mo.md(
        rf"""
    使用 CVXPY 得到的各项目投入资金安排分别为

    {solve_lp_cvxpy("x")}

    目标函数的值为

    {solve_lp_cvxpy("obj")}

    下图体现了不同 \( \lambda \) 下目标函数的变化。

    {mo.as_html(solve_lp_cvxpy_plot())}

    这与 Scipy 的求解结果是一致的。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""已知量如下""")
    return


@app.cell(hide_code=True)
def _(np, slider_investment_preference, slider_total_investment):
    # 参数定义

    # 投资项目数量 n
    num_projects: int = 4

    # 平均收益率 r
    average_yield = np.array([0, 0.28, 0.21, 0.23, 0.25])

    # 交易费率 p
    transaction_fee = np.array([0, 0.01, 0.02, 0.045, 0.065])

    # 风险损失率 q
    risk_loss = np.array([0, 0.025, 0.015, 0.055, 0.026])

    # 投资偏好系数
    investment_preference: float = slider_investment_preference.value

    # 投资总额
    total_investment: float = slider_total_investment.value
    return (
        average_yield,
        investment_preference,
        num_projects,
        risk_loss,
        total_investment,
        transaction_fee,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 使用 SciPy 求解

    函数定义如下
    """
    )
    return


@app.cell(hide_code=True)
def _(
    average_yield,
    investment_preference: float,
    np,
    num_projects: int,
    plt,
    risk_loss,
    total_investment: float,
    transaction_fee,
):
    def solve_lp_scipy(output: str = "", lam=investment_preference):
        from scipy.optimize import linprog

        # 平均收益率 r
        _r = np.hstack([average_yield, [-1]])

        # 交易费率 p
        _p = np.hstack([transaction_fee, [-1]])

        # 风险损失率 q
        _q = np.hstack([risk_loss, [0]])

        # 目标函数系数
        _c = np.hstack([np.repeat(0, num_projects + 1), [1]]) * lam - (1 - lam) * (
            _r - _p
        )

        # 不等式约束
        _A_ub = np.hstack(
            [
                np.diag(_q[: num_projects + 1]),
                -np.ones(shape=(num_projects + 1, 1)),
            ]
        )
        _b_ub = np.zeros(shape=(num_projects + 1, 1))

        # 等式约束
        _A_eq = (_p + 1).reshape((1, num_projects + 2))
        _b_eq = np.array([total_investment])

        # 求解
        _res = linprog(_c, _A_ub, _b_ub, _A_eq, _b_eq)

        if _res.success:
            match output:
                case "x":
                    return {
                        f"x{i}": float(_res.x[i].round(2))
                        for i in range(num_projects + 1)
                    }
                case "obj":
                    return round(_res.fun, 2)
                case _:
                    return _res
        else:
            raise ValueError("模型求解失败！")


    def solve_lp_scipy_plot():
        # 计算不同 lambda 下的风险和收益的关系
        N = 100
        lambdas = np.linspace(0, 1, N)
        revenue_values = np.empty(shape=(N,))
        risk_values = np.empty(shape=(N,))
        for i, lam in enumerate(lambdas):
            try:
                X = np.array(list(solve_lp_scipy("x", lam=lam).values()))
                revenue = ((average_yield - transaction_fee) * X).sum()
                risk = (risk_loss * X).max()
                revenue_values[i] = revenue
                risk_values[i] = risk
            except ValueError:
                revenue_values[i] = np.nan
                risk_values[i] = np.nan
        # 绘制风险和收益的关系
        fig = plt.figure(figsize=(8, 6))
        ax = fig.gca()
        ax.plot(lambdas, revenue_values, "g--")
        ax.plot(lambdas, risk_values, "r--")
        ax.legend([r"收益 $(\lambda)$", r"风险 $(\lambda)$"])
        ax.set_xlabel(r"投资偏好系数 $\lambda$")
        ax.set_ylabel("函数值")
        ax.set_title(
            rf"函数值随投资偏好系数 $\lambda$ 的变化"
            rf"| 投资总额 $M={total_investment}$"
        )
        # ax.plot(risk_values, revenue_values, "b*-")
        # ax.set_xlabel("风险")
        # ax.set_ylabel("收益")
        # ax.set_title(rf"风险收益曲线 | 投资总额 $M={total_investment}$")
        return ax
    return solve_lp_scipy, solve_lp_scipy_plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 使用 CVXPY 求解

    函数定义如下
    """
    )
    return


@app.cell(hide_code=True)
def _(
    average_yield,
    cp,
    investment_preference: float,
    np,
    num_projects: int,
    plt,
    risk_loss,
    total_investment: float,
    transaction_fee,
):
    def solve_lp_cvxpy(output: str = "", lam=investment_preference):
        # 平均收益率 r
        _r = average_yield.copy()

        # 交易费率 p
        _p = transaction_fee.copy()

        # 风险损失率 q
        _q = risk_loss.copy()

        # 定义变量
        _X = cp.Variable(num_projects + 1, nonneg=True)

        # 目标函数
        objective = cp.Minimize(lam * cp.max(cp.multiply(_q, _X)) - (1 - lam) * (_r - _p) @ _X)

        # 不等式约束
        constraints = [
            cp.sum(cp.multiply(_p + 1, _X)) == total_investment,
            _X >= 0,
        ]

        # 定义问题
        problem = cp.Problem(objective, constraints)

        # 求解问题
        problem.solve()

        if problem.status == cp.OPTIMAL:
            match output:
                case "x":
                    return {
                        f"x{i}": float(_X.value[i].round(2))
                        for i in range(num_projects + 1)
                    }
                case "obj":
                    return round(float(problem.value), 2)
                case _:
                    return problem
        else:
            raise ValueError("模型求解失败！")


    def solve_lp_cvxpy_plot():
        # 计算不同 lambda 下的风险和收益的关系
        N = 100
        lambdas = np.linspace(0, 1, N)
        revenue_values = np.empty(shape=(N,))
        risk_values = np.empty(shape=(N,))
        for i, lam in enumerate(lambdas):
            try:
                X = np.array(list(solve_lp_cvxpy("x", lam=lam).values()))
                revenue = ((average_yield - transaction_fee) * X).sum()
                risk = (risk_loss * X).max()
                revenue_values[i] = revenue
                risk_values[i] = risk
            except ValueError:
                revenue_values[i] = np.nan
                risk_values[i] = np.nan
        # 绘制风险和收益的关系
        fig = plt.figure(figsize=(8, 6))
        ax = fig.gca()
        ax.plot(lambdas, revenue_values, "g--")
        ax.plot(lambdas, risk_values, "r--")
        ax.legend([r"收益 $(\lambda)$", r"风险 $(\lambda)$"])
        ax.set_xlabel(r"投资偏好系数 $\lambda$")
        ax.set_ylabel("函数值")
        ax.set_title(
            rf"函数值随投资偏好系数 $\lambda$ 的变化"
            rf"| 投资总额 $M={total_investment}$"
        )
        # ax.plot(risk_values, revenue_values, "b*-")
        # ax.set_xlabel("风险")
        # ax.set_ylabel("收益")
        # ax.set_title(rf"风险收益曲线 | 投资总额 $M={total_investment}$")
        return ax
    return solve_lp_cvxpy, solve_lp_cvxpy_plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 整数规划

    问题建模：

    \begin{align}
        &\max\quad \sum_{j=1}^{7}w_j(x_{1j} + x_{2j}) \\
        &s.t.\quad\left\{\begin{aligned}
            &\sum_{i=1}^{2}x_{ij} \leqslant a_j,\quad j=1,2,\cdots,7 \\
            &\sum_{j=1}^{7}l_jx_{ij} \leqslant 1020,\quad i=1, 2 \\
            &\sum_{j=1}^{7}w_jx_{ij} \leqslant 40000,\quad i=1, 2 \\
            &\sum_{j=5}^{7}l_j(x_{1j} + x_{2j}) \leqslant 302.7 \\
            & x_{ij}\geqslant 0\; \text{且为整数},\quad i=1,2;\; j=1,2,\cdots 7
        \end{aligned}\right.
    \end{align}

    处于方便，下面只用 CVXPY 实现：
    """
    )
    return


@app.cell(hide_code=True)
def _(cp, mo, np):
    _a = np.array([8, 7, 9, 6, 6, 4, 8])

    _l = np.array([48.7, 52.0, 61.3, 72.0, 48.7, 52.0, 64.0])

    _w = np.array([2000, 3000, 1000, 500, 4000, 2000, 1000])

    # 声明整型变量
    _X = cp.Variable((2, 7), integer=True, nonneg=True)

    # 目标函数
    _obj = cp.Maximize(cp.sum(cp.multiply(_w, (_X[0] + _X[1]))))

    # 约束条件
    _con = [
        cp.sum(_X, axis=0, keepdims=True) <= _a.reshape((1, 7)),
        _X @ _l <= np.array([[1020]] * 2),
        _X @ _w <= np.array([[40000]] * 2),
        cp.sum(cp.multiply(_l[4:], _X[:, 4:])) <= 302.7,
    ]

    _prob = cp.Problem(_obj, _con)

    _prob.solve(solver="GLPK_MI")

    output: str = ""
    if _prob.status == cp.OPTIMAL:
        output = mo.md(
            rf'''
            ### 求解结果
            最优目标函数值为 {_prob.value:.2f}，各变量依次为：
            <br>
            {_X[0].value} <br>
            {_X[1].value}
            '''
        )
    else:
        output = mo.md("模型求解失败！请检查约束条件是否合理。")

    output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 非线性规划

    这里机器学习也可以用上，因此不再赘述了。
    """
    )
    return


if __name__ == "__main__":
    app.run()
