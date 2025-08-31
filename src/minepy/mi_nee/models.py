import math

import torch
import torch.nn as nn

from minepy.utils.utils import get_activation_fn


class HneeModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] = [128, 128, 128],
        afn: str = "elu",
    ):
        super().__init__()
        hidden_layers = [int(hl) for hl in hidden_layers]

        activation_fn = get_activation_fn(afn)
        seq = [nn.Linear(input_dim, hidden_layers[0]), activation_fn()]
        for i in range(len(hidden_layers) - 1):
            seq += [
                nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                activation_fn(),
            ]
        seq += [nn.Linear(hidden_layers[-1], 1)]
        self.network = nn.Sequential(*seq)

    def forward(self, sample, ref_sample):
        mean_f = self.network(sample).mean()
        ref_f = self.network(ref_sample)
        log_mean_ref = torch.logsumexp(ref_f, 0) - math.log(ref_f.shape[0])
        loss = -(mean_f - log_mean_ref)
        return loss
