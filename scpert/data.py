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
import scipy.sparse as sp
import torch

from lapjv import lapjv
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from socube.utils import log
from typing import Tuple
from torch.utils.data import Dataset, Subset, WeightedRandomSampler


def source_target_alignment(
    source: sc.AnnData, 
    target: sc.AnnData, 
    metric: str = "euclidean",
    top_frac: float = 0.5,
    seed: int = None) -> Tuple[sc.AnnData]:
    """
    Create data pairs between source domain and target domain.

    Parameters
    ------------
    source: sc.AnnData
        Source domain data.
    target: sc.AnnData
        Target domain data.
    metric: str
        Metric for calculating distance between two domains' cells.
        Default is "euclidean". Any metric supported by 
        `scipy.spatial.distance.cdist`
    top_frac: float
        Fraction of top cell pairs to be selected.
    seed: int
        Random seed for reproducibility.
    
    Returns
    -----------
    source: sc.AnnData
        Source domain data with selected cells.
    target: sc.AnnData
        Target domain data with selected cells.

    """
    g = np.random.default_rng(seed)
    def _weighted_subsample(data: sc.AnnData, size:int):
        if size >= data.shape[0]:
            return data
        else:
            value = data.X.toarray() if sp.issparse(data.X) else data.X
            mean = np.expand_dims(value.mean(axis=0), axis=0)
            prop = cdist(
                value, 
                mean, 
                metric=metric).flatten()
            prop = 1 / (prop + 1)
            prop = prop / prop.sum()
            indices = g.choice(data.shape[0], size=size, replace=False, p=prop)
            return data[indices].copy()
    if len(source) > len(target):
        source = _weighted_subsample(source, len(target))
    else:
        target = _weighted_subsample(target, len(source))

    source_loc = source.X.toarray() if sp.issparse(source.X) else source.X
    source_loc = source_loc - source_loc.mean(axis=0)
    target_loc = target.X.toarray() if sp.issparse(target.X) else target.X
    target_loc = (target_loc - target_loc.mean(axis=0))
    d = cdist(
        source_loc, 
        target_loc, 
        metric=metric)
    _, y, _ = lapjv(d)
    target = target[y]
    d = d[range(len(y)), y]
    top_k = int(len(y) * top_frac)
    if top_k < len(y):
        top_k_idx = np.argpartition(d, top_k)[:top_k]
        source = source[top_k_idx]
        target = target[top_k_idx]
    return source, target


def source_target_multiclass_alignment(
    source: sc.AnnData, 
    target: sc.AnnData, 
    class_col: str = "cell_type", 
    top_frac: float = 0.5,
    mertic: str = "euclidean",
    seed: int = None) -> Tuple[sc.AnnData]:
    """
    Create data pairs between source domain and target domain
    with cell type limited.

    Parameters
    ------------
    source: sc.AnnData
        Source domain data.
    target: sc.AnnData
        Target domain data.
    class_col: str
        Column name for cell type.
    top_frac: float
        Fraction of top cell pairs to be selected.
    metric: str
        Metric for calculating distance between two domains' cells.
        Default is "euclidean". Any metric supported by
        `scipy.spatial.distance.cdist`
    seed: int
        Random seed for reproducibility.

    Returns
    -----------
    source: sc.AnnData
        Source domain data with selected cells.
    target: sc.AnnData
        Target domain data with selected cells.
    """
    source_class = set(source.obs[class_col])
    target_class = set(target.obs[class_col])
    assert source_class == target_class, "Classes in source and target should be the same!"
    source_list = []
    target_list = []
    for cls in source_class:
        log("Preprocess", f"Sample for {cls}")
        s, t = source_target_alignment(
            source[source.obs[class_col] == cls].copy(),
            target[target.obs[class_col] == cls].copy(),
            seed= seed,
            metric=mertic,
            top_frac=top_frac
        )
        source_list.append(s)
        target_list.append(t)

    balance_source = sc.concat(source_list)
    balance_target = sc.concat(target_list)
    return balance_source, balance_target


def padding_data(data: sc.AnnData, padding_col_nums: int) -> sc.AnnData:
    """Padding the data with zeros to make it can be divided by the d_model."""
    if padding_col_nums == 0:
        return data
    else:
        padding_data = np.zeros((data.shape[0], padding_col_nums))
        if sp.issparse(data.X):
            if sp.isspmatrix_csr(data.X):
                padding_data = sp.csr_matrix(padding_data)
            elif sp.isspmatrix_csc(data.X):
                padding_data = sp.csc_matrix(padding_data)
        padding_data = sc.AnnData(
            X=padding_data, 
            var=data.var.iloc[:padding_col_nums], 
            obs=data.obs,
            dtype=data.X.dtype)
        padding_data.var_names = [f"padding_{i}" for i in range(padding_col_nums)]
        result = sc.concat([data, padding_data], axis=1)
        result.obs = data.obs
        result.uns["padding"] = padding_data.var_names.values
        return result


def unpadding_data(data: sc.AnnData):
    if "padding" in data.uns:
        old_var = data.var_names.drop(data.uns["padding"])
        data = data[:, old_var]
    return data


def clear_data(data: sc.AnnData) -> sc.AnnData:
    return sc.AnnData(X=data.X, obs=data.obs, var=data.var, dtype=data.X.dtype)

class PertDataset(Dataset):
    """
    Cell perturbation dataset.

    Parameters
    ------------
    source: sc.AnnData
        Source domain data.
    target: sc.AnnData
        Target domain data.
    class_col: str
        Column name for cell type.
    label_encoder: bool
        Whether encode non-numerical
        cell types as numerical labels.
    """
    def __init__(self, 
        source:sc.AnnData, 
        target:sc.AnnData, 
        class_col: str = "cell_type",
        label_encoder: bool = True):
        self._source = source
        self._target = target
        assert np.all(source.obs[class_col].values == target.obs[class_col].values), "Classes not matched!"
        self._class = source.obs[class_col].values
        if label_encoder:
            le = LabelEncoder()
            self._class = le.fit_transform(self._class)

        self._class_set = np.unique(self._class)

    def __len__(self) -> int:
        return self._source.shape[0]
    
    def __getitem__(self, index) -> dict:
        """
        Return a dict with soure data, target data
        and class numerical label.
        """
        return {
            "source": self._source[index].to_df().values.flatten(),
            "target": self._target[index].to_df().values.flatten(),
            "class": self._class[index]
        }


    def kFold(self, shuffle: bool = False, seed: int = None, k: int = 5):
        """
        Get generator for k-fold cross-validation dataset

        Returns
        ------------
        kFold: generator
            An generator for k-fold cross-validation dataset. Each iteration
            generates a tuple of two Subset objects for training and validating
        """
        if isinstance(seed, int):
            shuffle = True
        
        skf = StratifiedKFold(n_splits=k,
                              random_state=seed,
                              shuffle=shuffle)

        X = np.zeros((self._class.shape[0], 1))
        for train_index, valid_index in skf.split(X, self._class):
            yield (Subset(self, train_index), Subset(self, valid_index))


    def sampler(self, subset: Subset, seed: int = None) -> WeightedRandomSampler:
        """
        Generate weighted random sampler for a subset of this dataset

        Parameters
        ------------
        subset: Subset
            the subset of this dataset

        Returns
        ------------
        Weighted random sampler
        """
        assert subset.dataset == self, "Must be a subset of this dataset"
        labels = self._class[subset.indices]
        numSamples = labels.shape[0]
        labelWeights = numSamples / np.bincount(labels)
        sampleWeights = labelWeights[labels]
        generator = torch.Generator().manual_seed(
            seed) if seed is not None else None

        return WeightedRandomSampler(sampleWeights,
                                     numSamples,
                                     generator=generator)