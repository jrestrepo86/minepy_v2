import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ray

from minepy.class_mi.class_mi import ClassMI
from minepy.utils.systems import GaussianSamples

RNG_SEED = 1
NREA = 60  # number of realizations
DATA_LENGHT = int(1e4)
RHO = 0.9

# Net parameters
model_paramters = {
    "hidden_layers": [128, 128, 128],
    "afn": "relu",
    "clip_val": 1e-6,
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
    return get_mi(sim)


def get_mi(sim):
    x, y = sim["x"], sim["y"]

    # create model
    classifier_mi = ClassMI(X=x, Y=y, **model_paramters)
    # training
    classifier_mi.train(**training_parameters)

    # plot
    # classifier_mi.plot_metrics(show=True)

    # output
    out = classifier_mi.get_mi()

    return {
        "mi_dv": out[0],
        "mi_nwj": out[1],
        "mi_ldr": out[2],
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
            res.append(get_mi(sim))
        results = res

    # compile results
    results = pd.DataFrame(results)

    # mi true value
    true_mi = data.mi()
    return results, true_mi


def plot(results, true_value):
    fig = go.Figure()

    # Add a box for each metric
    fig.add_trace(go.Box(y=results["mi_dv"], name="CMI_DV"))
    fig.add_trace(go.Box(y=results["mi_nwj"], name="CMI_NWJ"))
    fig.add_trace(go.Box(y=results["mi_ldr"], name="CMI_LDR"))

    fig.update_layout(
        title="MI Estimates Comparison",
        yaxis_title="MI",
        boxmode="group",  # group boxes side by side
        template="plotly_white",
    )
    # Add a horizontal line for the true MI
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
        text=f"True MI = {true_value:.4f}",
        showarrow=False,
        yshift=10,
        font=dict(color="red"),
    )

    fig.show()


if __name__ == "__main__":
    results, tv = test_gaussian_model(parallel=True)
    plot(results, tv)
