# SA.py -- 灵敏度分析模板
from typing import Dict, Any, Protocol
import numpy as np
from SALib import ProblemSpec
from SALib.sample.sobol import sample as sobol_sample
from SALib.analyze.sobol import analyze as sobol_analyze

class Model(Protocol):
    def evaluate(self, X: np.ndarray, *args, **kwargs) -> np.ndarray:
        ...

class SA(object):
    """灵敏度分析

    Parameters
    ----------
    model : Model
        用于分析的模型，必须实现 `evaluate` 方法，其接受模型的参数并返回输出，
        且支持批量处理：输入的二维数组中每一行代表一个样本点，每一列代表一个参数。
        输出的格式应为二维数组，每一行代表一个样本点，每一列代表一个输出。
    problem : Dict[str, Any]
        灵敏度分析的问题描述，至少包括输入参数 `names` 及其范围 `bounds` ，
        多个输出时，还应包括输出变量 `outputs`，详见 SALib 文档。
    n_samples : int, optional
        生成的样本数量，默认为1024。
    """

    def __init__(self, model: Model, problem: Dict[str, Any],
                n_samples: int = 1024):
        self.model = model
        self.problem = problem
        self.n_samples = n_samples
        self.sp = None

    def run(self) -> None:
        """运行灵敏度分析
        """
        self.sp = ProblemSpec(self.problem)
        if not hasattr(self.model, 'evaluate'):
            self.sp = None
            raise AttributeError("模型必须实现 `evaluate` 方法。")
        # 采用 Sobol 方法进行采样和分析，其他方法可参考 SALib 文档
        (
            self.sp.sample(sobol_sample, N=self.n_samples)
            .evaluate(self.model.evaluate)
            .analyze(sobol_analyze)
        )

    def get_analysis(self) -> Dict:
        """获取灵敏度分析的指标，以字典形式返回，用法详见 SALib 文档。

        Returns
        -------
        Dict[str, np.ndarray]
            返回一组灵敏度指标，包括一阶、二阶和总灵敏度指数。
            返回一组灵敏度指标，包括一阶、二阶和总灵敏度指数。
        """
        if self.sp is None:
            raise ValueError("请先运行 `run` 方法以生成分析结果。")
        if not isinstance(self.sp.analysis, dict):
            raise TypeError("分析结果应为字典格式。")
        return self.sp.analysis

    def print_results(self) -> None:
        """输出分析结果
        """
        if self.sp is None:
            raise ValueError("请先运行 `run` 方法以生成分析结果。")
        print("Results of Sensitivity Analysis:")
        print("-------------------------------")
        print(self.sp)
        print("-------------------------------")

    def plot(self) -> None:
        """绘制灵敏度分析结果图
        """
        if self.sp is None:
            raise ValueError("请先运行 `run` 方法以生成分析结果。")
        self.sp.plot()

    def heatmap(self) -> None:
        """绘制灵敏度分析热力图
        """
        if self.sp is None:
            raise ValueError("请先运行 `run` 方法以生成分析结果。")
        self.sp.heatmap()

def _main() -> None:
    """主函数示例，供测试和演示使用
    """
    from SALib.test_functions import Ishigami
    import matplotlib.pyplot as plt
    import warnings
    # 忽略未来警告
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    class MyModel():
        def evaluate(self, X: np.ndarray, A: float = 7, B: float = 0.1) -> np.ndarray:
            return Ishigami.evaluate(X, A, B)
    
    problem: Dict[str, Any] = {
        "names": ["x1", "x2", "x3"],
        "bounds": [[-np.pi, np.pi]] * 3,
        "outputs": ["y1"]
    }
    model = MyModel()

    sa = SA(model, problem)

    # 运行灵敏度分析
    sa.run()

    # 打印分析结果
    sa.print_results()

    # 绘制热力图
    sa.heatmap()
    plt.show()

if __name__ == "__main__":
    _main()
