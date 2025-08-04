# MFCE.py -- 多层次模糊综合评价
# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from typing import Union, List, Callable

# Type alias for membership input, which can be a single ndarray or a list of ndarrays
MembershipInput = Union[np.ndarray, List["MembershipInput"]]

class FuzzyLayer:
    """
    Fuzzy Layer for Multi-Level Fuzzy Comprehensive Evaluation
    """
    def __init__(
        self,
        weights: np.ndarray,
        operator: Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = 'wedge_vee'
    ):
        """
        Initialize the Fuzzy Layer.

        Parameters
        ----------
        weights : np.ndarray, shape=(n_criteria, )
            The weights for the fuzzy layer.
        operator : Union[str, Callable[[np.ndarray], np.ndarray]]
            The aggregation operator for the fuzzy layer.
        """
        self.weights = weights
        self.operator = self._get_operator(operator)

    @staticmethod
    def _get_operator(
        op: Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        """
        Get the aggregation operator based on the input type.

        Parameters
        ----------
        op : Union[str, Callable[[np.ndarray], np.ndarray]]
            The operator to be used for aggregation.

        Returns
        -------
        Callable[[np.ndarray, np.ndarray], np.ndarray]
            The aggregation function.
        """
        if isinstance(op, str):
            match op.lower():
                case 'wedge_vee':
                    return lambda w, m: \
                        np.max(np.minimum(w[:, None], m), axis=0)
                case 'dot_vee':
                    return lambda w, m: \
                        np.max(w[:, None] * m, axis=0)
                case 'wedge_oplus':
                    return lambda w, m: \
                        np.minimum(1, np.minimum(w[:, None], m).sum(axis=0))
                case 'dot_oplus':
                    return lambda w, m: \
                        np.minimum(1, (w[:, None] * m).sum(axis=0))
                case _:
                    raise ValueError(f"Unsupported operator: {op!r}")
        elif callable(op):
            return op
        else:
            raise TypeError("Operator must be builtin or a callable function.")

    def __call__(self, membership_matrix: np.ndarray) -> np.ndarray:
        """
        Apply the fuzzy layer to the membership matrix.

        Parameters
        ----------
        membership_matrix : np.ndarray, shape=(n_criteria, n_alternatives)
            The membership degree matrix for the alternatives.

        Returns
        -------
        np.ndarray, shape=(n_alternatives, )
            The aggregated evaluation result for each alternative.
        """
        if membership_matrix.shape[0] != self.weights.shape[0]:
            raise ValueError("Membership matrix and weights must have compatible dimensions.")
        
        return self.operator(self.weights, membership_matrix)
    
    def __repr__(self) -> str:
        """
        String representation of the FuzzyLayer.

        Returns
        -------
        str
            The string representation of the FuzzyLayer.
        """
        return f"FuzzyLayer(weights={self.weights}, " \
            f"operator={self.operator.__name__})"

class MultiLevelFuzzyModel:
    """
    Multi-Level Fuzzy Comprehensive Evaluation Model
    """
    def __init__(self, 
                 layers: List[Union[FuzzyLayer, MultiLevelFuzzyModel]],
                 weights: np.ndarray,
                 operator: Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = 'wedge_vee'):
        """
        Initialize the Multi-Level Fuzzy Model.

        Parameters
        ----------
        layers : List[Union[FuzzyLayer, MultiLevelFuzzyModel]]
            The layers of the fuzzy model, which can be either FuzzyLayer or
            another MultiLevelFuzzyModel.
        weights : np.ndarray, shape=(n_layers, )
            The weights for each layer in the model.
        operator : Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]
            The aggregation operator for the model.
        """
        self.layers = layers
        self.weights = weights
        self.operator = FuzzyLayer._get_operator(operator)

    def __call__(self, 
                 membership_matrix: MembershipInput
            ) -> np.ndarray:
        """
        Apply the multi-level fuzzy model to the membership matrix.

        Parameters
        ----------
        membership_matrix : np.ndarray
            The N dimensional membership degree matrix.

        Returns
        -------
        np.ndarray
            The aggregated evaluation result.
        """
        sub_results = []
        for i, layer in enumerate(self.layers):
            sub_matrix = membership_matrix[i]
            if isinstance(layer, MultiLevelFuzzyModel):
                # recursively call the model
                sub_results.append(layer(sub_matrix))
            elif isinstance(layer, FuzzyLayer):
                if not isinstance(sub_matrix, np.ndarray):
                    raise TypeError(f"Expected np.ndarray for FuzzyLayer, got {type(sub_matrix)}")
                sub_results.append(layer(sub_matrix))
            else:
                raise TypeError(f"Layer must be FuzzyLayer or MultiLevelFuzzyModel, got {type(layer)}")
        
        if len(sub_results) != self.weights.shape[0]:
            raise ValueError("Number of sub-results must match the number of weights.")
        
        results = self.operator(self.weights, np.vstack(sub_results))
        return results
    
if __name__ == "__main__":
    # membership_matrix = np.array([
    #     [0.2, 0.5, 0.2, 0.1],
    #     [0.7, 0.2, 0.1, 0],
    #     [0, 0.4, 0.5, 0.1],
    #     [0.2, 0.3, 0.5, 0]
    # ])
    # weights = np.array([0.1, 0.2, 0.3, 0.4])
    # model = FuzzyLayer(weights)
    # print(model(membership_matrix))
    membership_matrix: MembershipInput = [
        np.array([
            [0.8, 0.15, 0.05, 0, 0],
            [0.2, 0.6, 0.1, 0.1, 0],
            [0.5, 0.4, 0.1, 0, 0],
            [0.1, 0.3, 0.5, 0.05, 0.05]
        ]),
        np.array([
            [0.3, 0.5, 0.15, 0.05, 0],
            [0.2, 0.2, 0.4, 0.1, 0.1],
            [0.4, 0.4, 0.1, 0.1, 0],
            [0.1, 0.3, 0.3, 0.2, 0.1],
            [0.3, 0.2, 0.2, 0.2, 0.1]
        ]),
        np.array([
            [0.1, 0.3, 0.5, 0.1, 0],
            [0.2, 0.3, 0.3, 0.1, 0.1],
            [0.2, 0.3, 0.35, 0.15, 0],
            [0.1, 0.3, 0.4, 0.1, 0.1],
            [0.1, 0.4, 0.3, 0.1, 0.1]
        ]),
        np.array([
            [0.3, 0.4, 0.2, 0.1, 0],
            [0.1, 0.4, 0.3, 0.1, 0.1],
            [0.2, 0.3, 0.4, 0.1, 0],
            [0.4, 0.3, 0.2, 0.1, 0]
        ])
    ]

    weights = [
        np.array([0.4, 0.3, 0.2, 0.1]),
        np.array([0.2, 0.3, 0.3, 0.2]),
        np.array([0.3, 0.2, 0.1, 0.2, 0.2]),
        np.array([0.1, 0.2, 0.3, 0.2, 0.2]),
        np.array([0.3, 0.2, 0.2, 0.3])
    ]

    model = MultiLevelFuzzyModel(
        layers=[
            FuzzyLayer(weights=weights[1], operator='dot_oplus'),
            FuzzyLayer(weights=weights[2], operator='dot_oplus'),
            FuzzyLayer(weights=weights[3], operator='dot_oplus'),
            FuzzyLayer(weights=weights[4], operator='dot_oplus'),
        ],
        weights=weights[0],
        operator='dot_oplus'
    )

    result = model(membership_matrix)
    print("Aggregated Result:", result)
