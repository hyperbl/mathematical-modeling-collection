import numpy as np
from typing import Union
from abc import ABC, abstractmethod

# 传统综合评价方法基类
class TraditionalCE(ABC):
    """
    Comprehensive Evaluation Model Base Class
    """
    def __init__(self, decision_matrix: np.ndarray, weights: Union[np.ndarray, None] = None):
        self.decision_matrix = decision_matrix
        self.weights = weights if weights is not None else np.ones(decision_matrix.shape[1]) / decision_matrix.shape[1]
        assert self.decision_matrix.shape[1] == self.weights.shape[0], "权重长度需等于指标数"

    def __call__(self, *args, **kwargs) -> "TraditionalCEResult":
        return self.evaluate(*args, **kwargs)

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> "TraditionalCEResult":
        """
        Evaluate the decision matrix using the method's specific algorithm.

        Returns
        -------
        TraditionalCEResult
            The result of the comprehensive evaluation.
        """
        ...


# 综合评价结果类
class TraditionalCEResult():
    """
    Comprehensive Evaluation Result Class
    """
    def __init__(self, scores: np.ndarray, method: str = "Unknown"):
        self.scores = scores
        self.method = method
        self.ranking = np.argsort(scores)[::-1]  # 根据得分降序自动生成排名（索引）

    def __repr__(self) -> str:
        return f"Results from Traditional Comprehensive Evaluation\n" \
            "-------------------------------------------------\n" \
            f"Method: {self.method}\n" \
            f"Scores: {self.scores}\n" \
            f"Ranking: {self.ranking}\n" \
            "-------------------------------------------------"

    def top(self, n: int = 1) -> np.ndarray:
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        elif n > len(self.ranking):
            raise ValueError(f"n must not exceed the number of scores: {len(self.ranking)}.")
        else:
            return self.ranking[:n]
    
if __name__ == "__main__":
    # Example usage
    decision_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    weights = np.array([0.5, 0.3, 0.2])
    
    result = TraditionalCEResult(scores=np.array([0.8, 0.6, 0.9]), method="ExampleMethod")
    print(result)
    print("Top score:", result.top(1))
