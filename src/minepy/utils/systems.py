import numpy as np


class GaussianSamples:
    def __init__(
        self, rho: float, data_lenght: int = 10000, rng_seed: int | None = None
    ):
        self.rng = np.random.default_rng(rng_seed)
        self.rho = rho
        self.data_lenght = data_lenght

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        mu = np.array([0, 0])
        cov_matrix = np.array([[1, self.rho], [self.rho, 1]])
        joint_samples = self.rng.multivariate_normal(
            mean=mu, cov=cov_matrix, size=(self.data_lenght,)
        )
        x = joint_samples[:, 0]
        y = joint_samples[:, 1]

        return x, y

    def mi(self) -> float:
        return -0.5 * np.log(1 - self.rho**2)

    def h(self) -> float:
        cov_matrix = np.array([[1, self.rho], [self.rho, 1]])
        det_cov_mat = np.linalg.det(cov_matrix)
        return np.log(2 * np.pi * np.exp(1)) + 0.5 * np.log(det_cov_mat)
