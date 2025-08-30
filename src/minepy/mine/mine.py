"""
MINE: Mutual information neural estimation

"""

import copy
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import schedulefree
import torch
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split

from minepy.utils.utils import (
    EarlyStopping,
    ExpMovingAverageSmooth,
    to_col_vector,
)

from .batch_sampler import Sampler
from .models import Model


class Mine:
    def __init__(
        self,
        X,
        Y,
        hidden_layers: list[int] = [64, 64],
        afn: str = "elu",
        clip_val: float = 1e-6,
        loss_type: str = "mine",
        mine_alpha: float = 0.01,
        remine_reg_weight: float = 0.1,
        remine_target_val: float = 0.0,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.x = to_col_vector(X)
        self.y = to_col_vector(Y)

        input_dim = self.x.shape[1] + self.y.shape[1]

        self.model = Model(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            afn=afn,
            clip_val=clip_val,
            loss_type=loss_type,
            mine_alpha=mine_alpha,
            remine_reg_weight=remine_reg_weight,
            remine_target_val=remine_target_val,
        ).to(self.device)

        self.metrics = None
        self.trained = False
        self.log_metrics = False
        self.best_state = None

    def train(
        self,
        batch_size: int = 64,
        num_epochs: int = 500,
        lr: float = 1e-5,
        weight_decay: float = 5e-5,
        test_size: float = 0.3,
        stop_patience: int = 100,
        stop_min_delta: float = 1e-4,
        stop_warmup_steps: int = 1000,
        log_metrics=True,
    ):
        self.log_metrics = log_metrics

        # Optimizer
        self.optimizer = schedulefree.AdamWScheduleFree(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            warmup_steps=1000,
        )
        # self.optimizer = torch.optim.AdamW(
        #     self.model.parameters(),
        #     lr=lr,
        #     weight_decay=weight_decay,
        #     # warmup_steps=1000,
        # )

        # setup early stopping
        early_stopping = EarlyStopping(
            patience=stop_patience,
            delta=stop_min_delta,
            warmup_steps=stop_warmup_steps,
        )

        # Exponential smooth
        smooth = ExpMovingAverageSmooth()

        n = self.x.shape[0]
        train_idx, test_idx = train_test_split(list(range(n)), test_size=test_size)

        training_sampler = Sampler(
            self.x[train_idx, :],
            self.y[train_idx, :],
        )

        testing_sampler = Sampler(
            self.x[test_idx, :],
            self.y[test_idx, :],
        )

        best_state, best_loss = None, float("inf")
        metrics = []
        for epoch in range(num_epochs):
            # training
            joint_samples, marginal_samples = training_sampler.sample(batch_size)
            joint_samples = joint_samples.to(self.device)
            marginal_samples = marginal_samples.to(self.device)
            self.model.train()
            self.optimizer.train()
            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()
                _, loss = self.model(joint_samples, marginal_samples)
                loss.backward()
                self.optimizer.step()

            # test
            joint_samples, marginal_samples = testing_sampler.sample(batch_size)
            joint_samples = joint_samples.to(self.device)
            marginal_samples = marginal_samples.to(self.device)
            self.model.eval()
            self.optimizer.eval()
            with torch.no_grad():
                mi, loss = self.model(joint_samples, marginal_samples)
                # smooth
                smoothed_loss = smooth(loss.item())

                if self.log_metrics:
                    metrics.append(
                        {
                            "epoch": epoch,
                            "loss": loss.item(),
                            "smoothed_loss": smoothed_loss,
                            "mi": mi.item(),
                        }
                    )
            # After each validation step:
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
                best_state = copy.deepcopy(self.model.state_dict())

            # EarlyStopping
            early_stopping(smoothed_loss)
            if early_stopping.early_stop:
                if best_state is not None:
                    self.model.load_state_dict(best_state)
                break

        self.trained = True

        if log_metrics:
            self.metrics = pd.DataFrame(metrics)

    def get_mi(self, T=15):
        if not self.trained:
            raise ValueError("Did you call .train()?")

        sampler = Sampler(
            self.x,
            self.y,
        )
        mi = []
        for _ in range(T):
            joint_samples, marginal_samples = sampler.sample(self.x.shape[0] // T)
            joint_samples = joint_samples.to(self.device)
            marginal_samples = marginal_samples.to(self.device)
            self.model.eval()
            self.optimizer.eval()
            with torch.no_grad():
                mi_cmi, _ = self.model(joint_samples, marginal_samples)
                mi.append(mi_cmi.item())
        mi = np.mean(mi)
        return mi
