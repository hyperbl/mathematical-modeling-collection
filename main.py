from analysis.SA import SA
import numpy as np
from SALib.test_functions import Ishigami
from typing import Dict, Any

# 忽略未来警告
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class MyModel():
    def evaluate(self, X: np.ndarray, A: float = 7, B: float = 0.1) -> np.ndarray:
        return Ishigami.evaluate(X, A, B)
    
def main() -> None:
    problem: Dict[str, Any] = {
        "names": ["x1", "x2", "x3"],
        "bounds": [[-np.pi, np.pi]] * 3,
        "outputs": ["y1"]
    }
    model = MyModel()

    sa = SA(model, problem)

    sa.run()
    sa.print_results()
    sa.heatmap()
    import matplotlib.pyplot as plt
    plt.show()

if __name__ == "__main__":
    main()
