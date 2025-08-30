from typing import Optional, Tuple

import numpy as np
import torch


class Sampler:
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        rng_seed: Optional[int] = None,
    ):
        self.rng = np.random.default_rng(rng_seed)
        self.x = X
        self.y = Y
        self.n = self.x.shape[0]

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(batch_size / 2)

        idx = self.rng.permutation(self.n)[:batch_size]
        x_batch = self.x[idx, :]
        y_batch = self.y[idx, :]

        # Joint Samples
        joint_samples = np.concatenate(
            (
                x_batch,
                y_batch,
            ),
            axis=1,
        )
        # Marginal Samples
        marginal_samples = np.concatenate(
            (x_batch, self.rng.permutation(y_batch, axis=1)),
            axis=1,
        )

        # Merge
        samples = np.concatenate((joint_samples, marginal_samples), axis=0)
        labels = np.concatenate((np.ones(batch_size), np.zeros(batch_size)), axis=0)

        # shuffle batch
        shuffle_idx = self.rng.permutation(2 * batch_size)

        return (
            torch.tensor(
                samples[shuffle_idx, :], dtype=torch.float32, requires_grad=True
            ),
            torch.tensor(labels[shuffle_idx], dtype=torch.float32, requires_grad=False),
        )
