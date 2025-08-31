import math

import numpy as np
import torch

rng = np.random.default_rng()


class Uniform:
    def __init__(self, x: np.ndarray):
        self.dim = x.shape[1]
        self.xmin = x.min(axis=0)
        self.xmax = x.max(axis=0)

    def sample(self, sample_size: int) -> np.ndarray:
        sample = rng.uniform(0.0, 1.0, size=(sample_size, self.dim))
        return sample * (self.xmax - self.xmin) + self.xmin

    def entropy(self) -> float:
        return np.sum(np.log(self.xmax - self.xmin))


class UniformClip:
    def __init__(self, x: np.ndarray):
        self.dim = x.shape[1]
        self.xmin = np.percentile(x, 1, axis=0)
        self.xmax = np.percentile(x, 99, axis=0)

    def sample(self, sample_size: int) -> np.ndarray:
        sample = rng.uniform(0.0, 1.0, size=(sample_size, self.dim))
        return sample * (self.xmax - self.xmin) + self.xmin

    def entropy(self) -> float:
        return np.sum(np.log(self.xmax - self.xmin))


class Gaussian:
    def __init__(self, x):
        self.dim = x.shape[1]
        self.xmean = x.mean(axis=0)
        self.xstd = x.std(axis=0, ddof=0)  # population std
        self.var = self.xstd**2 + 1e-12  # prevent zero variance
        self.cov_matrix = np.diag(self.var)  # (dim, dim) diagonal

    def sample(self, sample_size: int) -> np.ndarray:
        return rng.multivariate_normal(
            mean=self.xmean, cov=self.cov_matrix, size=sample_size
        )

    def entropy(self) -> float:
        return 0.5 * (self.dim * np.log(2 * np.pi * np.e) + np.sum(np.log(self.var)))


class RefDistribution:
    def __init__(self, x, ref_type: str = "uniform", ref_sample_mult: int = 2):
        self.ref_type = ref_type.lower()
        self.ref_sample_mult = ref_sample_mult

        if ref_type == "uniform":
            self.distribution = Uniform(x)
        elif ref_type == "uniform_clip":
            self.distribution = UniformClip(x)
        elif ref_type == "gaussian":
            self.distribution = Gaussian(x)
        else:
            raise ValueError(
                "Reference distribution not found. Distributions: uniform, uniform_clip and gaussian."
            )

    def sample(self, batch_size) -> torch.Tensor:
        sample = self.distribution.sample(self.ref_sample_mult * batch_size)
        return torch.tensor(sample, dtype=torch.float32, requires_grad=False)

    def entropy(self) -> torch.Tensor:
        h = self.distribution.entropy()
        return torch.tensor(h, dtype=torch.float32, requires_grad=False)
