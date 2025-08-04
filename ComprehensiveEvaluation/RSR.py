# RSR.py -- 秩和比法
# -*- coding: utf-8 -*-
import numpy as np
from typing import Union
from .base import TraditionalCE, TraditionalCEResult
from scipy.stats import rankdata

class RSR(TraditionalCE):
    """
    Rank Sum Ratio (RSR) Method for comprehensive evaluation.
    """

    def __init__(self, decision_matrix: np.ndarray, weights: Union[np.ndarray, None] = None):
        super().__init__(decision_matrix, weights)

    def evaluate(self) -> TraditionalCEResult:
        """
        Evaluate the decision matrix using the RSR algorithm.

        Returns
        -------
        TraditionalCEResult
            The result of the comprehensive evaluation.
        """
        # Calculate the rank of each alternative
        ranks = np.vstack([
            rankdata(self.decision_matrix[:, i], method='average') 
                for i in range(self.decision_matrix.shape[1])
        ]).T

        # Calculate the weighted sum of ranks
        scores = np.dot(ranks, self.weights) / ranks.shape[0]

        return TraditionalCEResult(scores=scores, method="RSR")

if __name__ == "__main__":
    decision_matrix = np.array([
        [0.8, 0.5556, 0.9524, 0.8182, 0.7143, 1],
        [1, 1, 0.8571, 0.6923, 0.4286, 0.5],
        [0.72, 0.7407, 1, 1, 1, 0.7],
        [0.88, 0.6667, 0.9524, 0.9, 0.7143, 0.5]
    ])

    res = RSR(decision_matrix=decision_matrix)()

    print(res)