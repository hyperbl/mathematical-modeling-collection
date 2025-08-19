import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # 多元分析参考文献

    - 李东风.《[多元统计分析讲义](https://www.math.pku.edu.cn/teachers/lidf/course/mvr/mvrnotes/html/_mvrnotes/index.html)》. 2024.
    """
    )
    return


if __name__ == "__main__":
    app.run()
