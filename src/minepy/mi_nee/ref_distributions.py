import math

import numpy as np
import torch

rng = np.random.default_rng()
LOG2PI = math.log(2.0 * math.pi)


class Uniform:
    def __init__(self, x: np.ndarray, ref_type):
        self.dim = x.shape[1]
        self.xmin = x.min(axis=0)
        self.xmax = x.max(axis=0)
        self.width = np.maximum(self.xmax - self.xmin, 1e-12)  # avoid zero width
        self._log_vol = float(np.sum(np.log(self.width)))

    def sample(self, sample_size: int) -> np.ndarray:
        u = rng.uniform(0.0, 1.0, size=(sample_size, self.dim))
        return u * (self.xmax - self.xmin) + self.xmin

    def entropy(self) -> float:
        return self._log_vol  # log volume

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, dim) torch tensor
        log q(x) = -log(volume) if xmin <= x <= xmax (all dims), else -inf
        """
        device = x.device
        dtype = x.dtype
        xmin = torch.as_tensor(self.xmin, device=device, dtype=dtype)
        xmax = torch.as_tensor(self.xmax, device=device, dtype=dtype)
        inside = (x >= xmin) & (x <= xmax)
        inside_all = torch.all(inside, dim=1)
        logq = torch.full((x.shape[0],), -float("inf"), device=device, dtype=dtype)
        logq = torch.where(
            inside_all,
            torch.as_tensor(self.entropy(), device=device, dtype=dtype) * 0
            - self.entropy(),
            logq,
        )
        # The expression above sets logq = -log(volume) when inside; -inf otherwise.
        # Slightly clearer:
        logq_inside = -torch.as_tensor(self.entropy(), device=device, dtype=dtype)
        logq = torch.where(inside_all, logq_inside.expand_as(logq), logq)
        return logq


class Gaussian:
    def __init__(self, x: np.ndarray):
        self.dim = x.shape[1]
        self.xmean = x.mean(axis=0)
        xstd = x.std(axis=0, ddof=0)  # population std
        self.var = np.maximum(xstd**2, 1e-12)  # avoid zero variance
        # precompute constants for entropy
        self._log_det = float(np.sum(np.log(self.var)))
        self._entropy = 0.5 * (self.dim * (LOG2PI + 1.0) + self._log_det)

    def sample(self, sample_size: int) -> np.ndarray:
        cov_matrix = np.diag(self.var)
        return rng.multivariate_normal(
            mean=self.xmean, cov=cov_matrix, size=sample_size
        )

    def entropy(self) -> float:
        return self._entropy

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Diagonal Gaussian log-density:
        log q(x) = -0.5 * [ d*log(2π) + sum log var + sum ( (x-μ)^2 / var ) ]
        """
        device = x.device
        dtype = x.dtype
        mean = torch.as_tensor(self.xmean, device=device, dtype=dtype)
        var = torch.as_tensor(self.var, device=device, dtype=dtype)
        diff = x - mean
        quad = torch.sum(diff * diff / var, dim=1)  # (N,)
        log_det = torch.as_tensor(self._log_det, device=device, dtype=dtype)
        return -0.5 * (x.shape[1] * LOG2PI + log_det + quad)


class RefDistribution:
    def __init__(self, x, ref_type: str = "uniform", ref_sample_mult: int = 2):
        self.ref_type = ref_type.lower()
        self.ref_sample_mult = ref_sample_mult

        if self.ref_type == "uniform":
            self.distribution = Uniform(x, ref_type="uniform")
        elif self.ref_type == "gaussian":
            self.distribution = Gaussian(x)
        else:
            raise ValueError(f"Unknown reference_distribution: {ref_type}")

    def sample(self, batch_size: int) -> torch.Tensor:
        sample = self.distribution.sample(self.ref_sample_mult * batch_size)
        return torch.tensor(sample, dtype=torch.float32, requires_grad=False)

    def entropy(self) -> torch.Tensor:
        h = self.distribution.entropy()
        return torch.tensor(h, dtype=torch.float32, requires_grad=False)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(x)

    def cross_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Return -E_P[log q(X)] over the batch x."""
        lp = self.log_prob(x)  # (N,)
        return -torch.mean(lp)
