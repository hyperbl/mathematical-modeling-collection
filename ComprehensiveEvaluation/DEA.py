# DEA.py -- 数据包络分析法
# -*- coding: utf-8 -*-
import numpy as np
import cvxpy as cp
from .base import TraditionalCE, TraditionalCEResult

class DEA(TraditionalCE):
    """
    Data Envelopment Analysis (DEA) for efficiency evaluation.
    """

    SUPPORTED_MODELS = ["CCR", "BCC"]

    def __init__(self, 
                 input_matrix: np.ndarray,
                 output_matrix: np.ndarray,
                 model: str = "CCR"):
        """
        Parameters
        ----------
        input_matrix : np.ndarray, shape=(n_dmus, n_inputs)
            Input data matrix (DMUs x Inputs).
        output_matrix : np.ndarray, shape=(n_dmus, n_outputs)
            Output data matrix (DMUs x Outputs).
        model : str, optional
            DEA model type, such as "CCR" or "BCC". Default is "CCR".
        """
        super().__init__(np.hstack([input_matrix, output_matrix]))

        self.input_matrix = input_matrix
        self.output_matrix = output_matrix
        self.model = model.upper()

        if self.model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported DEA model: {self.model}. Supported models are {self.SUPPORTED_MODELS}."
            )
        
    def evaluate(self) -> TraditionalCEResult:
        """
        Evaluate the decision matrix using the DEA algorithm.

        Returns
        -------
        TraditionalCEResult
            The result of the comprehensive evaluation.
        """
        match self.model:
            case "CCR":
                scores = self._evaluate_ccr()
            case "BCC":
                scores = self._evaluate_bcc()
            case _:
                raise ValueError(
                    f"Unsupported DEA model: {self.model}. Supported models are {self.SUPPORTED_MODELS}."
                )
        
        return TraditionalCEResult(
            scores=scores,
            method=f"DEA-{self.model}"
        )

    def _evaluate_ccr(self) -> np.ndarray:
        """
        Evaluate using the CCR model.

        Returns
        -------
        np.ndarray, shape=(n_dmus, )
            The efficiency scores for each DMU.
        """

        # Define the variables
        n_dmus = self.input_matrix.shape[0]
        n_outputs = self.output_matrix.shape[1]
        n_inputs = self.input_matrix.shape[1]

        # Initialize the scores array
        scores = np.zeros(shape=(n_dmus, ))

        for i in range(n_dmus):
            input_vector = self.input_matrix[i, :]
            output_vector = self.output_matrix[i, :]

            # Declare the optimization variables
            omega = cp.Variable(shape=(n_inputs, ), nonneg=True)
            mu = cp.Variable(shape=(n_outputs, ), nonneg=True)

            # Define the objective function
            objective = cp.Maximize(
                cp.sum(cp.multiply(mu, output_vector))
            )

            # Define the constraints
            constraints: list = [
                cp.sum(cp.multiply(omega, input_vector)) == 1
            ]
            for j in range(n_dmus):
                constraints.append(
                    omega @ self.input_matrix[j, :] - mu @ self.output_matrix[j, :] >= 0
                )

            # Formulate the problem
            problem = cp.Problem(objective, constraints)
            problem.solve()

            # Store the efficiency score
            if problem.status == cp.OPTIMAL:
                assert isinstance(problem.value, (float, int))
                scores[i] = float(problem.value)
            else:
                raise ValueError(
                    f"Problem {i} could not be solved optimally. Status: {problem.status}"
                    f"\n{problem.solution}"
                )
        return scores

    def _evaluate_bcc(self) -> np.ndarray:
        """
        Evaluate using the BCC model.

        Returns
        -------
        np.ndarray, shape=(n_dmus, )
            The efficiency scores for each DMU.
        """
        
        raise NotImplementedError(
            "BCC model evaluation is not implemented yet."
        )


if __name__ == "__main__":
    data = np.array([
        [89.39, 86.25, 108.13, 106.38, 62.40, 47.19],
        [64.3, 99, 99.6, 96, 96.2, 79.9],
        [25.2, 28.2, 29.4, 26.4, 27.2, 25.2],
        [223, 287, 317, 291, 295, 222]
    ]).T

    input_matrix = data[:, :2]
    output_matrix = data[:, 2:]
    
    res = DEA(input_matrix, output_matrix, model="CCR")()

    print(res)