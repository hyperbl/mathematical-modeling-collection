# Entropy.py -- 熵值法确定权重
# -*- coding: utf-8 -*-
import numpy as np

class Entropy:
    """
    Entropy method for determining weights in multi-criteria decision-making.
    """
    def __init__(self, decision_matrix: np.ndarray):
        """
        Parameters
        ----------
        decision_matrix : np.ndarray, shape=(n_alternatives, n_criteria)
            The decision matrix where each row represents an alternative and each column represents a criterion.
        """
        self.decision_matrix = decision_matrix

    def calculate_weights(self) -> np.ndarray:
        """
        Calculate the weights based on the entropy method.

        Returns
        -------
        np.ndarray
            The calculated weights for each criterion.
        """
        # Normalize the decision matrix
        normalized_matrix = self.decision_matrix / np.sum(self.decision_matrix, axis=0)

        # Calculate the entropy for each criterion
        entropy = -np.nansum(normalized_matrix * np.log(normalized_matrix + 1e-10), axis=0) / np.log(len(self.decision_matrix))

        # Calculate the weights
        weights = (1 - entropy) / np.sum(1 - entropy)

        return weights
    
if __name__ == "__main__":
    decision_matrix = np.array([
        [0.8, 0.5556, 0.9524, 0.8182, 0.7143, 1],
        [1, 1, 0.8571, 0.6923, 0.4286, 0.5],
        [0.72, 0.7407, 1, 1, 1, 0.7],
        [0.88, 0.6667, 0.9524, 0.9, 0.7143, 0.5]
    ])

    entropy_method = Entropy(decision_matrix)
    weights = entropy_method.calculate_weights()
    print("Calculated Weights:", weights)