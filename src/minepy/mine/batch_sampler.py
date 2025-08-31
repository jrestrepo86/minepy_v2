import numpy as np
import torch


class Sampler:
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        rng_seed: int | None = None,
    ):
        self.rng = np.random.default_rng(rng_seed)
        self.x = X
        self.y = Y
        self.n = self.x.shape[0]

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Joint Samples
        idx = self.rng.permutation(self.n)[:batch_size]
        joint_samples = np.concatenate(
            (
                self.x[idx, :],
                self.y[idx, :],
            ),
            axis=1,
        )
        # Marginal Samples
        idx_x = self.rng.permutation(self.n)[:batch_size]
        idx_y = self.rng.permutation(self.n)[:batch_size]
        marginal_samples = np.concatenate(
            (self.x[idx_x, :], self.y[idx_y, :]),
            axis=1,
        )

        return (
            torch.tensor(joint_samples, dtype=torch.float32, requires_grad=False),
            torch.tensor(marginal_samples, dtype=torch.float32, requires_grad=False),
        )
