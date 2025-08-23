# SAA.py -- Simulated Annealing Algorithm
from abc import ABC, abstractmethod
from typing import Union
import numpy as np

class SimulatedAnnealingBase(ABC):
    def __init__(self, 
                 T_begin: float, T_end: float, 
                 max_iter: int = 1000, n_iter: int = 100, 
                 max_stall: int = 20, 
                 seed: int = 42
        ) -> None:
        """Simulated Annealing Algorithm Base Class

        Parameters
        ----------
        T_begin : float
            Initial temperature
        T_end : float
            Final temperature
        max_iter : int = 1000
            Maximum number of iterations
        n_iter : int = 100
            Number of iterations at each temperature
        max_stall: int = 20
            Maximum number of stalls
        seed : int = 42
            Random seed for reproducibility
        """
        self.T_begin = T_begin
        self.T_end = T_end
        self.max_iter = max_iter
        self.n_iter = n_iter
        self.max_stall = max_stall
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def decrease_temperature(self, T: Union[float, np.floating], *args, **kwargs) -> Union[float, np.floating]:
        """Decrease the temperature.
        """
        pass

    @abstractmethod
    def neighbor(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Generate a neighboring solution.

        Parameters
        ----------
        x : np.ndarray
            Current solution.

        Returns
        -------
        np.ndarray
            A neighboring solution.
        """
        pass

    @abstractmethod
    def accept(self, cost_old: Union[float, np.floating], cost_new: Union[float, np.floating], T: Union[float, np.floating], *args, **kwargs) -> bool:
        """Determine whether to accept a new solution.

        Parameters
        ----------
        cost_old : float
            Cost of the old solution.
        cost_new : float
            Cost of the new solution.
        T : float
            Current temperature.

        Returns
        -------
        bool
            Whether to accept the new solution.
        """
        pass

    @abstractmethod
    def cost(self, x: np.ndarray, *args, **kwargs) -> Union[float, np.floating]:
        """Calculate the cost of a solution.

        Parameters
        ----------
        x : np.ndarray
            The solution to evaluate.

        Returns
        -------
        float
            The cost of the solution.
        """
        pass

    def run(self, x_begin: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Run the simulated annealing algorithm.

        Parameters
        ----------
        x_begin : np.ndarray
            Initial solution.

        Returns
        -------
        np.ndarray
            Best solution found.
        """
        x: np.ndarray = x_begin.copy()
        cost: Union[float, np.floating] = self.cost(x, *args, **kwargs)
        x_best: np.ndarray = x_begin.copy()
        cost_best: Union[float, np.floating] = self.cost(x_best, *args, **kwargs)
        T_current: Union[float, np.floating] = self.T_begin
        iter_outer: int = 0
        stall_count: int = 0

        # outer loop
        while T_current > self.T_end and iter_outer < self.max_iter:

            improved: bool = False

            # inner loop
            for _ in range(self.n_iter):
                x_new: np.ndarray = self.neighbor(x, *args, **kwargs)
                cost_new: Union[float, np.floating] = self.cost(x_new, *args, **kwargs)

                if self.accept(cost, cost_new, T_current, *args, **kwargs):
                    x = x_new.copy()
                    cost = cost_new

                    if cost_best > cost:
                        x_best = x.copy()
                        cost_best = cost
                        improved = True
            
            if improved:
                stall_count = 0
            else:
                stall_count += 1

            if stall_count >= self.max_stall:
                break

            T_current = self.decrease_temperature(T_current, *args, **kwargs)
            iter_outer += 1

        return x_best

    def __call__(self, x_begin: np.ndarray, *args, **kwargs) -> np.ndarray :
        return self.run(x_begin, *args, **kwargs)

def main() -> None:
    print("Hello, World from SAA.py!")

if __name__ == "__main__":
    main()