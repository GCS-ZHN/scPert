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
import abc
import scanpy as sc
import torch
import torch.nn as nn

from scpert import config
from scpert.data import PertDataset, unpadding_data
from socube.utils import autoClearIter
from torch.utils.data import DataLoader

class NetBase(nn.Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super(NetBase, self).__init__()

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def criterion(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, source: sc.AnnData, class_col: str) -> sc.AnnData:
        pred_scPert = []
        for batch in autoClearIter(DataLoader(PertDataset(source, source, class_col), batch_size=256)):
            pred_scPert.append(self(batch["source"].to(self.device)).cpu().detach())
        pred_scPert = torch.cat(pred_scPert, dim=0).numpy()
        pred_scPert = sc.AnnData(
            X=pred_scPert, 
            obs=source.obs, 
            var=source.var, 
            dtype=source.X.dtype)
        pred_scPert.obs[config.CONDITION_COL] = self.__class__.__name__
        pred_scPert = unpadding_data(pred_scPert)
        return pred_scPert

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @device.setter
    def device(self, device: torch.device) -> None:
        self.to(device)


class PertNet(NetBase):
    """
    A transformer network for cell perturbation prediction.

    Parameters
    ------------
    d_model: int
        The dimension of virtural word embedding.
    n_heads: int
        The number of attention heads.
    n_layers: int
        The number of transformer encoder layers.
    n_features: int
        The number of features in the input data.
    """
    def __init__(
        self, 
        d_model: int, 
        n_features: int, 
        n_heads: int = 4, 
        n_layers: int = 4,
        padding_size: int = 0) -> None:
        super(PertNet, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_features % d_model == 0, "n_features must be divisible by d_model"
        self.d_model = d_model
        self.padding_size = padding_size
        self.n_features = n_features
        self._cross_head = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(d_model)
        )
        self._tranformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=1024, batch_first=True),
            num_layers=n_layers
            )
        self._weight = nn.Parameter(torch.randn((1, n_features)))
        
    def get_delta(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_features = x.shape
        assert n_features == self.n_features, "Input feature number not matched!"
        h = x.view(batch_size, n_features // self.d_model, self.d_model)
        h = h.swapaxes(1, 2)
        h = self._cross_head(h)
        h = h.swapaxes(1, 2)
        h = self._tranformer(h)
        return h.view(batch_size, n_features) * self._weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.get_delta(x)
        return  delta + x

    def criterion(self, p: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Model's relative MSE loss.

        Parameters
        ------------
        p: torch.Tensor
            Predicted data.
        t: torch.Tensor
            Target domain data.
        s: torch.Tensor
            Source domain data.

        Returns
        ------------
        loss: torch.Tensor
            Loss value.
        """
        p = p[:, :self.n_features - self.padding_size]
        s = s[:, :self.n_features - self.padding_size]
        t = t[:, :self.n_features - self.padding_size]
        loss_func = nn.MSELoss()
        return loss_func(p, t) / loss_func(s, t)


class FFNet(NetBase):
    
    def __init__(self, n_features:int) -> None:
        super(FFNet, self).__init__()

        self._delta = nn.Sequential(
            nn.Linear(n_features, n_features)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._delta(x)

    def criterion(self, p: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Model's relative MSE loss.

        Parameters
        ------------
        p: torch.Tensor
            Predicted data.
        t: torch.Tensor
            Target domain data.
        s: torch.Tensor
            Source domain data.

        Returns
        ------------
        loss: torch.Tensor
            Loss value.
        """
        loss_func = nn.MSELoss()
        return loss_func(p, t) / loss_func(s, t)
