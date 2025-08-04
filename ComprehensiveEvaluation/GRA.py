# GRA.py -- 灰色关联分析法
# -*- coding: utf-8 -*-
import numpy as np
from typing import Union, Callable
from .base import TraditionalCE, TraditionalCEResult

class GRA(TraditionalCE):
    """
    GRA (Grey Relational Analysis) 
    """
    def __init__(
            self,
            decision_matrix: np.ndarray, 
            weights: Union[np.ndarray, None] = None, 
            rho: Union[float, np.floating] = 0.5,
            reference: Union[np.ndarray, str, Callable[[np.ndarray], np.ndarray]] = 'max'
        ):
        """Initialize the GRA method.

        Parameters
        ----------
        decision_matrix : np.ndarray
            The decision matrix for evaluation.
        weights : Union[np.ndarray, None], optional
            The weights for each criterion, by default None
        rho : Union[float, np.floating], optional
            The rho parameter for GRA, by default 0.5
        reference : Union[np.ndarray, str, Callable[[np.ndarray], np.ndarray]], optional
            The reference point for GRA, by default 'max'

        Returns
        -------
        None
            The method has no return value.

        Raises
        ------
        TypeError
            If the reference is not a valid type.
        """
        super().__init__(decision_matrix, weights)
        self.rho = rho
        if isinstance(reference, np.ndarray):
            assert reference.shape[0] == decision_matrix.shape[1], \
                "Reference must match the number of features."
            self.reference = reference
        elif isinstance(reference, str):
            supported_methods = ['max', 'min', 'mean', 'median']
            if reference not in supported_methods:
                print(f"Warning: {reference} is not a supported method. Using 'max' instead.")
                reference = 'max'
            self.reference = np.__dict__[reference](decision_matrix, axis=0)
        elif callable(reference):
            self.reference = reference(decision_matrix)
            assert isinstance(self.reference, np.ndarray), \
                "Reference function must return a numpy array."
            assert self.reference.shape[0] == decision_matrix.shape[1], \
                "Reference must match the number of features."
        else:
            raise TypeError("Reference must be a numpy array or"
                            " a string representing a numpy function or"
                            " a callable function that returns a numpy array.")

    def evaluate(self) -> TraditionalCEResult:
        """
        Evaluate the decision matrix using the GRA algorithm.

        Returns
        -------
        TraditionalCEResult
            The result of the comprehensive evaluation.
        """

        minmin = np.min(np.abs(self.decision_matrix - self.reference))
        maxmax = np.max(np.abs(self.decision_matrix - self.reference))

        # get the grey relational coefficients: xi
        xi = (minmin + self.rho * maxmax) / \
             (np.abs(self.reference - self.decision_matrix) + self.rho * maxmax)

        return TraditionalCEResult(
            scores=np.sum(xi*self.weights, axis=1),
            method='GRA'
        )
    
if __name__ == "__main__":
    decision_matrix = np.array([
        [0.8, 0.5556, 0.9524, 0.8182, 0.7143, 1],
        [1, 1, 0.8571, 0.6923, 0.4286, 0.5],
        [0.72, 0.7407, 1, 1, 1, 0.7],
        [0.88, 0.6667, 0.9524, 0.9, 0.7143, 0.5]
    ])

    res = GRA(decision_matrix=decision_matrix)()

    print(res)