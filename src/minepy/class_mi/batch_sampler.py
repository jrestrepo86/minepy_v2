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
        y_perm = self.rng.permutation(y_batch, axis=0)
        # This ensures no joint sample is recycled as its own fake marginal.
        # possible if batch size is small
        for i in range(len(y_perm)):
            if np.array_equal(
                y_perm[i], y_batch[i]
            ):  # y remained paired with its original x
                # swap with another random index j
                j = (i + 1) % len(y_perm)
                y_perm[i], y_perm[j] = y_perm[j].copy(), y_perm[i].copy()

        marginal_samples = np.concatenate(
            (x_batch, y_perm),
            axis=1,
        )

        # Merge
        samples = np.concatenate((joint_samples, marginal_samples), axis=0)
        labels = np.concatenate(
            (np.ones(batch_size), np.zeros(batch_size)), axis=0
        )

        # shuffle batch
        shuffle_idx = self.rng.permutation(2 * batch_size)

        return (
            torch.tensor(
                samples[shuffle_idx, :], dtype=torch.float32, requires_grad=True
            ),
            torch.tensor(
                labels[shuffle_idx], dtype=torch.float32, requires_grad=False
            ),
        )
