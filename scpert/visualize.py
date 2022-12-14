# MIT License
#
# Copyright (c) 2022 Zhang.H.N
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scpert import config
from scipy.stats import pearsonr
from socube.utils import log
from typing import Sequence, Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_alignment(
    source: sc.AnnData, 
    target: sc.AnnData, 
    class_col: str = "cell_type",
    cluster_type: str = "umap",
    save_path: str = None) -> plt.Axes:
    """
    Visualize the data alignment between source domains
    and target domains

    Parameters
    ------------
    source: sc.AnnData
        Source domain data.
    target: sc.AnnData
        Target domain data.
    class_col: str
        Column name for cell type.
    cluster_type: str
        Clustering method for visualization.
        You should run clustering before calling this function.
        Any clustering method supported by `scanpy.tl`.
    save_path: str
        Path to save the figure.

    Returns
    -----------
    ax: plt.Axes
    """
    source_class = source.obs[class_col]
    target_class = target.obs[class_col]
    cluster_type = f"X_{cluster_type}"
    assert len(source) == len(target), "Inconsistent count of cells in source and target!"
    assert np.all(source_class.values == target_class.values), "Classes in source and target should be the same!"
    assert cluster_type in source.obsm and cluster_type in target.obsm, "Cluster type not existed!"
    merge = sc.concat([source, target])
    cell_types = source_class.unique()
    plt.cla()

    ax = plt.axes()
    for cell in cell_types:
        source = merge[(merge.obs[class_col]==cell) & (merge.obs[config.CONDITION_COL] == config.SOURCE_NAME)]
        target = merge[(merge.obs[class_col]==cell) & (merge.obs[config.CONDITION_COL] == config.TARGET_NAME)]
        start = np.array([source.obsm[cluster_type][:,0].mean(), source.obsm[cluster_type][:,1].mean()])
        direction = (target.obsm[cluster_type][:,:2] - source.obsm[cluster_type][:,:2]).mean(axis=0)
        plt.scatter(
            source.obsm[cluster_type][:,0], 
            source.obsm[cluster_type][:,1],
            label=f"{config.SOURCE_NAME}[{cell}]",
            )

        plt.scatter(
            target.obsm[cluster_type][:,0], 
            target.obsm[cluster_type][:,1],
            label=f"{config.TARGET_NAME}[{cell}]"
            )

        plt.arrow(start[0], start[1], direction[0], direction[1], width=0.2)
    if save_path:
        plt.savefig(save_path)
    return ax


def plot_regression_plot(
    X: sc.AnnData, 
    Y: sc.AnnData, 
    diff_genes: np.ndarray,
    n_obs:int = 100,
    x_label: str = "X",
    y_label: str = "Y", 
    figsize=(8, 4),
    save_path: str = None,
    random_state: int = None) -> Figure:
    sns.set(style="ticks", font_scale=1.0)
    fig, axs = plt.subplots(1, 2 if len(diff_genes) > 0 else 1, figsize=figsize)
    x_data, y_data = _pair_sample(X, Y, n_obs=n_obs, random_state=random_state)
    _plot_regression_plot(
        x=x_data.mean(), 
        y=y_data.mean(), 
        x_label=x_label, 
        y_label=y_label, 
        ax=axs[0] if len(diff_genes) > 0 else axs, 
        all_gene=True)
    if len(diff_genes) > 0:
        _plot_regression_plot(
            x=x_data[diff_genes].mean(), 
            y=y_data[diff_genes].mean(), 
            x_label=x_label, 
            y_label=y_label, 
            ax=axs[1])
    if save_path is not None:
        plt.savefig(
            save_path, 
            facecolor="auto", 
            edgecolor="auto",
            dpi=300, 
            bbox_inches="tight")
    return fig


def _pair_sample(
    X: sc.AnnData,
    Y: sc.AnnData,
    n_obs: int = 100,
    random_state: int = None) -> Tuple[Sequence]:
    tmp = min(n_obs, len(X), len(Y))
    if tmp < n_obs:
        log("Visualization", "Warning: n_obs is too large, using {} instead.".format(tmp), level="warn")
        n_obs = tmp
    x_data = sc.pp.subsample(X, n_obs=n_obs, copy=True, random_state=random_state).to_df()
    y_data = sc.pp.subsample(Y, n_obs=n_obs, copy=True, random_state=random_state).to_df()
    return x_data, y_data


def _plot_regression_plot(
    x: Sequence,
    y: Sequence,
    all_gene: bool = False,
    x_label: str = "X",
    y_label: str = "Y", 
    ax:Axes = None,
    scatter_kwargs: dict = None,
    **kwargs) -> Axes:
    r_value, _ = pearsonr(x, y)
    assert len(x) == len(y), "Length of x and y should be the same!"
    df = pd.DataFrame({x_label: x, y_label: y})
    ax = sns.regplot(x=x_label, y=y_label, data=df, scatter_kws=scatter_kwargs, ax=ax)
    if all_gene:
        ax.text(0.05, 0.9, r'$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$=' + f"{r_value ** 2:.2f}", transform=ax.transAxes, **kwargs)
    else:
        assert len(x) == 100, "Length of x and y should be 100!"
        ax.text(0.05, 0.9, r'$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEG}}}}$=' + f"{r_value ** 2:.2f}", transform=ax.transAxes, **kwargs)
    return ax

