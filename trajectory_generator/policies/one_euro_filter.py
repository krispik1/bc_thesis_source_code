import numpy as np

class OneEuroFilter:

    def __init__(
            self,
            dimension: int,
            frequency: float = 120.0,
            min_cutoff: float = 1.0,
            beta: float = 0.2,
            derivative_cutoff: float = 1.0
    ):
        """
        Class representing one euro filter described in the paper.
        """
        self.frequency = frequency
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff

        self.x_prev = None
        self.dx_prev = np.zeros(dimension)

    def alpha(
            self,
            cutoff: float
    ) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / self.frequency

        return 1.0 / (1.0 + tau / te)

    def filter(
            self,
            x: np.ndarray,
    ) -> np.ndarray:

        if self.x_prev is None:
            self.x_prev = x
            return x

        dx = (x - self.x_prev) * self.frequency

        alpha_d = self.alpha(self.derivative_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = self.alpha(cutoff)

        x_hat = alpha * x + (1 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat