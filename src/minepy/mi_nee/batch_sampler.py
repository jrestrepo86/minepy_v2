import numpy as np
import torch


class HneeSampler:
    def __init__(
        self,
        X: np.ndarray,
        rng_seed: int | None = None,
    ):
        self.rng = np.random.default_rng(rng_seed)
        self.x = X
        self.n = self.x.shape[0]

    def sample(self, batch_size: int) -> torch.Tensor:
        # Joint samples
        idx = self.rng.permutation(self.n)[:batch_size]
        joint_samples = self.x[idx, :]

        return torch.tensor(joint_samples, dtype=torch.float32, requires_grad=False)
