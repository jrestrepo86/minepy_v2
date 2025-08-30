import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import schedulefree
import torch
import torch.nn as nn
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split

from minepy.utils.utils import (
    EarlyStopping,
    ExpMovingAverageSmooth,
    get_activation_fn,
)

from .batch_sampler import Sampler
from .models import Classifier


class ClassMI:
    def __init__(
        self,
        X,
        Y,
        hidden_layers: list[int] = [64, 64],
        afn: str = "elu",
        clip_val: float = 1e-6,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.x = X
        self.y = Y

        input_dim = self.x[1] + self.y.shape[1]

        self.model = Classifier(
            input_dim=input_dim, hidden_layers=hidden_layers, afn=afn, clip_val=clip_val
        ).to(self.device)

        self.metrics = None
        self.trained = False
        self.log_metrics = False

    def _mi(self, w: torch.Tensor, labels: torch.Tensor):
        with torch.no_grad():
            # Assumes blanced classes
            log_gamma = torch.log(w) - torch.log(1 - w)

            log_gamma_joint = log_gamma[labels == 1]
            log_gamma_prod = log_gamma[labels == 0]

            cmi_dv = (
                log_gamma_joint.mean()
                - torch.logsumexp(log_gamma_prod, dim=0)
                + math.log(log_gamma_prod.shape[0])
            )
            cmi_nwj = 1 + log_gamma_joint.mean() - torch.exp(log_gamma_prod).mean()
            cmi_ldr = log_gamma_joint.mean()
        return cmi_dv, cmi_nwj, cmi_ldr

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
        log_metrics=False,
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
            patience=stop_patience, delta=stop_min_delta, warmup_steps=stop_warmup_steps
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

        metrics = []
        for epoch in range(num_epochs):
            # training
            samples, labels = training_sampler.sample(batch_size)
            samples = samples.to(self.device)
            labels = labels.to(self.device)
            self.model.train()
            self.optimizer.train()
            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()
                w, loss = self.model(samples, labels)
                loss.backward()
                self.optimizer.step()

            # test
            samples, labels = testing_sampler.sample(batch_size)
            samples = samples.to(self.device)
            labels = labels.to(self.device)
            self.model.eval()
            self.optimizer.eval()
            with torch.no_grad():
                w, loss = self.model(samples, labels)
                # smooth
                smoothed_loss = smooth(loss.item())

                if self.log_metrics:
                    dv_cmi, nwj_cmi, ldr_cmi = self._mi(w, labels)
                    metrics.append(
                        {
                            "epoch": epoch,
                            "loss": loss.item(),
                            "smoothed_loss": smoothed_loss,
                            "dv": dv_cmi,
                            "nwj": nwj_cmi,
                            "ldr": ldr_cmi,
                        }
                    )

            # EarlyStopping
            early_stopping(smoothed_loss)
            if early_stopping.early_stop:
                break

        self.trained = True

        if log_metrics:
            self.metrics = pd.DataFrame(metrics)

    def get_mi(self, T=30):
        if not self.trained:
            raise ValueError("Did you call .train()?")

        sampler = Sampler(
            self.x,
            self.y,
        )
        est_dv, est_nwj, est_ldr = [], [], []
        for _ in range(T):
            samples, labels = sampler.sample(self.x.shape[0] // T)
            samples = samples.to(self.device)
            labels = labels.to(self.device)
            self.model.eval()
            # self.optimizer.eval()
            with torch.no_grad():
                w, _ = self.model(samples, labels)
                dv_cmi, nwj_cmi, ldr_cmi = self._mi(w, labels)
                est_dv.append(dv_cmi.item())
                est_nwj.append(nwj_cmi.item())
                est_ldr.append(ldr_cmi.item())
        est_dv = np.mean(est_dv)
        est_nwj = np.mean(est_nwj)
        est_ldr = np.mean(est_ldr)
        return est_dv, est_nwj, est_ldr

    def plot_metrics(self, text="", show=True):
        if self.metrics is None:
            raise ValueError("No metrics to plot. Did you call .train()?")
        if self.log_metrics is False:
            print("No mi metrics recorded. Set log_metrics = True and call .train()?")

        epochs = self.metrics["epoch"].values
        loss = self.metrics["loss"].values
        smoothed_loss = self.metrics["smoothed_loss"].values
        if self.log_metrics:
            dv = self.metrics["dv"].values
            nwj = self.metrics["nwj"].values
            ldr = self.metrics["ldr"].values
        else:
            dv = np.zeros(len(epochs))
            nwj = np.zeros(len(epochs))
            ldr = np.zeros(len(epochs))

        # --- Create figure with 2 rows and 1 column ---
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Epochs vs Loss", "Epochs vs MI Estimates"),
        )

        # --- First row: Loss ---
        fig.add_trace(
            go.Scatter(x=epochs, y=loss, mode="lines+markers", name="Loss"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=epochs, y=smoothed_loss, mode="lines+markers", name="Smoothed Loss"
            ),
            row=1,
            col=1,
        )

        # --- Second row: CMI metrics ---
        fig.add_trace(
            go.Scatter(x=epochs, y=dv, mode="lines+markers", name="DV"), row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=nwj, mode="lines+markers", name="NWJ"), row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=ldr, mode="lines+markers", name="LDR"), row=2, col=1
        )

        # --- Update layout ---
        fig.update_layout(
            title_text="Training Metrics" + " " + text,
            template="plotly_white",
        )

        # Axis labels
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="MI", row=2, col=1)

        if show:
            fig.show()
        return fig
