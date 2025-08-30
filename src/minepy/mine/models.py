import math

import torch
import torch.nn as nn

from minepy.utils.utils import get_activation_fn

EPS = 1e-10


class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = (
            grad_output * input.exp().detach() / (running_mean + EPS) / input.shape[0]
        )
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema
    return t_log, running_mean


class Model(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] = [64, 64, 32],
        afn="elu",
        clip_val: float = 1e-6,
        loss_type: str = "mine",
        mine_alpha: float = 0.01,
        remine_reg_weight: float = 0.1,
        remine_target_val: float = 0.0,
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

        self.clip_val = clip_val
        self.running_mean = 0
        self.loss_type = loss_type.lower()
        self.mine_alpha = mine_alpha
        self.regWeight = remine_reg_weight
        self.remine_target_val = remine_target_val

    def forward(self, joint_samples, marginal_samples):
        w_joint = self.network(joint_samples).mean()
        w_joint = torch.clamp(w_joint, min=self.clip_val, max=1 - self.clip_val)

        w_marg = self.network(marginal_samples)
        w_marg = torch.clamp(w_marg, min=self.clip_val, max=1 - self.clip_val)

        if self.loss_type == "mine":
            second_term, self.running_mean = ema_loss(
                w_marg, self.running_mean, self.mine_alpha
            )
            mi = w_joint - second_term
            loss = -mi
        elif self.loss_type == "nwj":
            second_term = torch.exp(w_marg - 1).mean()
            mi = w_joint - second_term
            loss = -mi
        elif self.loss == "remine":
            second_term = torch.logsumexp(w_marg, 0) - math.log(w_marg.shape[0])
            mi = w_joint - second_term
            loss = -mi + self.remine_target_val * torch.pow(
                second_term - self.remine_target_val, 2
            )
        else:
            # mine_biased as default
            second_term = torch.logsumexp(w_marg, 0) - math.log(w_marg.shape[0])
            mi = w_joint - second_term
            loss = -mi

        return mi, loss
