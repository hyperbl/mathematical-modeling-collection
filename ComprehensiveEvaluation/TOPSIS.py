# TOPSIS.py -- 逼近理想解法
# -*- coding: utf-8 -*-
import numpy as np
from typing import Union
from .base import TraditionalCE, TraditionalCEResult

class TOPSIS(TraditionalCE):
    """
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) 
    """
    def __init__(self, decision_matrix: np.ndarray, weights: Union[np.ndarray, None] = None):
        super().__init__(decision_matrix, weights)

    def evaluate(self) -> TraditionalCEResult:
        """
        Evaluate the decision matrix using the TOPSIS algorithm.

        Returns
        -------
        TraditionalCEResult
            The result of the comprehensive evaluation.
        """

        # Weight the normalized decision matrix
        weighted_matrix = self.decision_matrix * self.weights

        # Determine the ideal and negative-ideal solutions
        ideal_solution = np.max(weighted_matrix, axis=0)
        negative_ideal_solution = np.min(weighted_matrix, axis=0)

        # Calculate the distance to the ideal and negative-ideal solutions
        distance_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution) ** 2, axis=1))
        distance_to_negative_ideal = np.sqrt(np.sum((weighted_matrix - negative_ideal_solution) ** 2, axis=1))

        # Calculate the TOPSIS scores
        scores = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)

        return TraditionalCEResult(scores=scores, method="TOPSIS")

if __name__ == "__main__":
    decision_matrix = np.array([
        [0.8, 0.5556, 0.9524, 0.8182, 0.7143, 1],
        [1, 1, 0.8571, 0.6923, 0.4286, 0.5],
        [0.72, 0.7407, 1, 1, 1, 0.7],
        [0.88, 0.6667, 0.9524, 0.9, 0.7143, 0.5]
    ])

    res = TOPSIS(decision_matrix=decision_matrix)()

    print(res)

