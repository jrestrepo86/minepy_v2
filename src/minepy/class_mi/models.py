"""
Classifier based mutual information

@inproceedings{
  title={CCMI: Classifier based conditional mutual information estimation},
  author={Mukherjee, Sudipto and Asnani, Himanshu and Kannan, Sreeram},
  booktitle={Uncertainty in artificial intelligence},
  pages={1083--1093},
  year={2020},
  organization={PMLR}
}
"""

import torch
import torch.nn as nn

from minepy.utils.utils import get_activation_fn


class Classifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        afn: str,
        clip_val: float,
    ):
        super().__init__()
        hidden_layers = [int(hl) for hl in hidden_layers]
        activation_fn = get_activation_fn(afn)

        seq = [
            nn.Linear(input_dim, hidden_layers[0]),
            activation_fn(),
            nn.Dropout(p=0.2),
        ]
        for i in range(len(hidden_layers) - 1):
            seq += [
                nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                activation_fn(),
                nn.Dropout(p=0.2),
            ]
        seq += [nn.Linear(hidden_layers[-1], 1), nn.Sigmoid()]
        self.network = nn.Sequential(*seq)
        self.clip_val = clip_val

        # BCELoss expects probabilities (after sigmoid) and labels as floats
        self.bce_loss = torch.nn.BCELoss(reduction="mean")

    def forward(
        self, x, labels
    ) -> tuple[torch.Tensor, torch.Tensor]:  # x: (batch, input_dim)
        w = self.network(x).view(-1)
        w = torch.clamp(w, min=self.clip_val, max=1 - self.clip_val)
        return w, self.bce_loss(w, labels)
