import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ray

from minepy.mi_nee.h_nee import HNee
from minepy.utils.systems import GaussianSamples
from minepy.utils.utils import to_col_vector

RNG_SEED = 1
NREA = 60  # number of realizations
DATA_LENGHT = int(1e4)
RHO = 0.9

# Net parameters
model_paramters = {
    "hidden_layers": [128, 128, 128],
    "afn": "elu",
    "reference_distribution": "uniform_clip",
    "ref_sample_mult": 2,
}
# Training
training_parameters = {
    "batch_size": 256,
    "num_epochs": 30000,
    "lr": 5e-3,
    "weight_decay": 5e-4,
    "test_size": 0.3,
    "stop_patience": 100,
    "stop_min_delta": 1e-5,
}

rng = np.random.default_rng(RNG_SEED)


@ray.remote(num_gpus=1 / 15)
def parallel_calc(sim):
    return get_h(sim)


def get_h(sim):
    x = np.concatenate((to_col_vector(sim["x"]), to_col_vector(sim["y"])), axis=1)

    # create model
    hnee = HNee(X=x, **model_paramters)
    # training
    hnee.train(**training_parameters)

    # plot
    # hnee.plot_metrics(show=True)

    # output
    out = hnee.get_h()

    return {
        "h": out,
    }


def test_gaussian_model(parallel=True):
    sims = []
    data = GaussianSamples(rho=RHO, data_lenght=DATA_LENGHT, rng_seed=RNG_SEED)
    for _ in range(NREA):
        x, y = data.sample()
        sims.append(
            {
                "x": x,
                "y": y,
                "model_paramters": model_paramters,
                "training_parameters": training_parameters,
            }
        )
    res = []

    if parallel:
        for sim in sims:
            res.append(parallel_calc.remote(sim))
        results = ray.get(res)
    else:
        for sim in sims:
            res.append(get_h(sim))
        results = res

    # compile results
    results = pd.DataFrame(results)

    # mi true value
    true_mi = data.h()
    return results, true_mi


def plot(results, true_value):
    fig = go.Figure()

    # Add a box for each metric
    fig.add_trace(go.Box(y=results["h"], name="H"))

    fig.update_layout(
        title="Entropy Estimates Comparison",
        yaxis_title="H",
        boxmode="group",  # group boxes side by side
        template="plotly_white",
    )
    # Add a horizontal line for the true Entropy
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=2.5,  # span across the 3 boxplots
        y0=true_value,
        y1=true_value,
        line=dict(color="red", dash="dash"),
        xref="x",
        yref="y",
    )

    # Optional annotation for the true value
    fig.add_annotation(
        x=1,  # roughly center
        y=true_value,
        text=f"True H = {true_value:.4f}",
        showarrow=False,
        yshift=10,
        font=dict(color="red"),
    )

    fig.show()


if __name__ == "__main__":
    results, tv = test_gaussian_model(parallel=True)
    plot(results, tv)
