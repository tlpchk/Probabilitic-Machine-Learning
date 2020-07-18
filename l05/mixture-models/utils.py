import typing as t

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

import common
from gmm import MixtureModel

RNG = np.random.RandomState(1337)
sns.set()


def load_s_set_random_split() -> t.Tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset data containing two covering bowls.

    The split is performed randomly.
    :return: Tuple of train data and test data. Each element is a pandas
        DataFrame where first two columns correspond to coordinates on the 2D
        planes and the last column is a class.
    """
    data = pd.read_csv(
        common.S_SET_DATA / "data.txt", delimiter="\t", names=["x", "y"]
    )
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    indices = RNG.permutation(len(data))
    split_point = int(common.S_SET_SPLIT_RATIO * len(indices))

    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    return data.iloc[train_indices], data.iloc[test_indices]


def visualize(model: MixtureModel, x: torch.Tensor) -> plt.Axes:
    """Visualize results of clustering using Mixture Model.


    :param model: Learned mixture model that implements scoring each sample
        using `predict_proba` method and `score`. The `score` returns negative
        likelihood for each sample (the higher the better). `predict_proba`
        returns probability of sample belonging to a particular cluster. Both
        methods should return matrices N x K where K is number of components.
    :param x: torch.Tensor containing N x D matrix of samples where N is
        a number of samples and D is a number of dimensions.
    :return: plt.Axes that can be visualized through
        `matplotlib.pyplot.show()`.
    """
    color_palette = np.array(sns.color_palette("hls", model.num_components))
    probas = model.predict_proba(x)
    x = x.detach().numpy()

    classes = probas.argmax(dim=1).detach().numpy().astype(np.int)
    frame = pd.DataFrame(data={"x": x[:, 0], "y": x[:, 1], "cls": classes})

    xi, yi = np.meshgrid(
        np.linspace(frame["x"].min(), frame["x"].max(), num=100),
        np.linspace(frame["y"].min(), frame["y"].max(), num=100),
    )
    data = torch.stack(
        (
            torch.from_numpy(xi).float().flatten(),
            torch.from_numpy(yi).float().flatten(),
        ),
        dim=-1,
    )

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    zi = model.score(data).detach().numpy()
    for cls in range(model.num_components):
        subframe = frame[frame["cls"] == cls]
        if len(subframe) > 0:
            ax.scatter(
                subframe["x"], subframe["y"], color=color_palette[cls], s=6
            )
        sub_class_zi = zi[:, cls].reshape(xi.shape)
        ax.contour(
            xi,
            yi,
            sub_class_zi,
            levels=[0.4, 0.6, 0.8, 1.0],
            norm=Normalize(vmin=0, vmax=1),
            colors=[color_palette[cls]],
        )

    return ax
