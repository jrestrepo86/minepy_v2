import copy

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
from .models import HneeModel
from .ref_distributions import RefDistribution


class HNee:
    def __init__(
        self,
        X,
        hidden_layers=[150, 150, 150],
        afn: str = "elu",
        reference_distribution: str = "uniform",
        ref_sample_mult: int = 2,
        device=None,
    ):
        super().__init__()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        torch.device(self.device)

        self.x = to_col_vector(X)

        input_dim = self.x.shape[1]

        self.model = HneeModel(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            afn=afn,
        ).to(self.device)

        # Reference distribution
        self.ref_distribution = RefDistribution(
            self.x,
            ref_type=reference_distribution,
            ref_sample_mult=ref_sample_mult,
        )

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

        training_sampler = Sampler(self.x[train_idx, :])
        testing_sampler = Sampler(self.x[test_idx, :])

        best_state, best_loss = None, float("inf")
        metrics = []

        for epoch in range(num_epochs):
            # training
            samples = training_sampler.sample(batch_size).to(self.device)
            ref_samples = self.ref_distribution.sample(batch_size).to(self.device)
            self.model.train()
            self.optimizer.train()
            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()
                loss = self.model(samples, ref_samples)
                loss.backward()
                self.optimizer.step()

            # test
            samples = testing_sampler.sample(batch_size).to(self.device)
            ref_samples = self.ref_distribution.sample(batch_size).to(self.device)
            self.model.eval()
            self.optimizer.eval()
            with torch.no_grad():
                loss = self.model(samples, ref_samples)
                entropy = loss.item() + self.ref_distribution.entropy().item()
                # smooth
                smoothed_loss = smooth(loss.item())

                if self.log_metrics:
                    metrics.append(
                        {
                            "epoch": epoch,
                            "loss": loss.item(),
                            "smoothed_loss": smoothed_loss,
                            "h": entropy,
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

    def get_h(self):
        if not self.trained:
            raise ValueError("Did you call .train()?")

        sampler = Sampler(self.x)
        samples = sampler.sample(self.x.shape[0]).to(self.device)
        ref_samples = self.ref_distribution.sample(self.x.shape[0]).to(self.device)
        self.model.eval()
        self.optimizer.eval()
        with torch.no_grad():
            loss = self.model(samples, ref_samples)
        return loss.item() + self.ref_distribution.entropy().item()

    def plot_metrics(self, text="", show=True):
        if self.metrics is None:
            raise ValueError("No metrics to plot. Did you call .train()?")
        if self.log_metrics is False:
            print("No mi metrics recorded. Set log_metrics = True and call .train()?")

        epochs = self.metrics["epoch"].values
        loss = self.metrics["loss"].values
        smoothed_loss = self.metrics["smoothed_loss"].values
        if self.log_metrics:
            h = self.metrics["h"].values
        else:
            h = np.zeros(len(epochs))

        # --- Create figure with 2 rows and 1 column ---
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Epochs vs Loss", "Epochs vs Entropy Estimate"),
        )

        # --- First row: Loss ---
        fig.add_trace(
            go.Scatter(x=epochs, y=loss, mode="lines+markers", name="Loss"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=smoothed_loss,
                mode="lines+markers",
                name="Smoothed Loss",
            ),
            row=1,
            col=1,
        )

        # --- Second row: MI metrics ---
        fig.add_trace(
            go.Scatter(x=epochs, y=h, mode="lines+markers", name="Hnee"),
            row=2,
            col=1,
        )

        # --- Update layout ---
        fig.update_layout(
            title_text=f"Hnee | Training Metrics | Ref. Distribution {self.ref_distribution.ref_type}"
            + " "
            + text,
            template="plotly_white",
        )

        # Axis labels
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="MI", row=2, col=1)

        if show:
            fig.show()
        return fig
