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
import os
import scanpy as sc
import scpert.config as config
import torch
import torch.optim as optim
import numpy as np


from scpert.data import PertDataset
from scpert.model import FFNet, NetBase, PertNet
from scpert.visualize import plot_regression_plot
from scipy.stats import pearsonr
from socube.utils import log, autoClearIter, mkDirs, loadTorchModule, visualBytes
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


class EarlyStopping(object):
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Parameters
    ----------
    descend: bool, default True
        If True, the loss is minimized. If False, the loss is maximized.
    patience: int, default 7
        How long to wait after last time validation loss improved.
    verbose: int, default 0
        Prints a message which level great than `verbose` for each
         validation loss improvement.
    delta: float, default 0
        Minimum change in the monitored quantity to qualify as an improvement.
    path: str, default 'checkpoint.pt'
        Path for the checkpoint to be saved to.
    """

    def __init__(self,
                 descend: bool = True,
                 patience: int = 7,
                 threshold: float = 1e-5,
                 verbose: int = 0,
                 delta: float = 0,
                 path='checkpoint.pt'):
        self._flag = 1 if descend else -1
        self._patience = patience
        self._verbose = verbose
        self._counter = 0
        self._best = np.inf if descend else -np.inf
        self._stop = False
        assert delta >= 0, "delta should be not less than 0"
        self._delta = delta
        self._path = path
        self._threshold = threshold

    def __call__(self,
                 score: float,
                 model: torch.nn.Module) -> bool:
        """
        Object callable function to update record

        Parameters
        --------------
        score: float
            the loss score of model
        model: torch.nn.Module
            the model to be training

        Returns
        --------------
            Boolean value
        """
        if (self._best - score) * self._flag > self._delta:
            log(EarlyStopping.__name__,
                f"Score changes from {self._best} to {score}",
                quiet=self._verbose > 1)
            torch.save(model.state_dict(), self._path)
            self._best = score
            self._counter = 0
            return True
        else:
            self._counter += 1
            log(EarlyStopping.__name__,
                f'EarlyStopping counter: {self._counter} out of {self._patience}',
                quiet=self._verbose > 5)
            if self._counter >= self._patience:
                self._stop = True
            return False

    @property
    def earlyStop(self) -> bool:
        """
        Wether to reach early stopping point.

        Returns
        --------------
            Boolean value
        """
        return self._stop or (self._best - self._threshold) * self._flag <= 0


@torch.no_grad()
def validate(data_loader: DataLoader,
             model: PertNet,
             device: torch.device,
             with_progress: bool = False) -> dict:
    """
    Validate model performance basically

    Parameters
    ----------
    dataLoader: the torch dataloader object used for validation
    model: Network model implemented `NetBase`
        the model waited for validation
    device: the cpu/gpu device

    Returns
    ----------
    a quadra tuple of (average loss, average ACC, true label, predict score)
    """
    with torch.cuda.device(device if device.type == "cuda" else -1):
        model.to(device)
        model.eval()
        source_list = list()
        target_list = list()
        predict_list = list()
        itererate = autoClearIter(enumerate(data_loader, 1))
        if with_progress:
            itererate = tqdm(itererate, desc="Validate")

        for _, batch in itererate:
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            predict = model(source)
            target_list.append(target.cpu())
            predict_list.append(predict.cpu())
            source_list.append(source.cpu())
            # destroy useless object and recycle GPU memory
            del source, target, predict

        target = torch.cat(target_list)
        predict = torch.cat(predict_list)
        source = torch.cat(source_list)
        loss = model.criterion(predict, target, source)
        corr = pearsonr(predict.mean(axis=0), target.mean(axis=0))[0]
        return {
            "loss": loss.cpu().detach().item(),
            "predict": predict,
            "target": target,
            "pearson": corr,
        }


def train(
    jobid: str,
    dataset: PertDataset,
    device: torch.device,
    padding_size: int,
    lr: float = 1e-3,
    d_model: int = 40,
    n_features: int = 2000,
    n_layers: int = 6,
    n_heads: int = 10,
    n_epoches: int = 250,
    update_lr_step: int = 5,
    pretrain_path: str = None,
    train_batch_size: int = 64,
    valid_batch_size: int = 256,
    early_stop_patience: int = 7,
    early_stop_min_delta: float = 1e-4,
    exp_scheduler_gamma: float = 0.95,
    seed: int = None,
    model_name: str = PertNet.__name__,):
    """
    Train a perturbation network

    Parameters
    ------------
    jobid: str
        The job id for this training, 
        used for logging and checkpointing
    device: torch.device
        The device used for training
    lr: float
        The learning rate
    d_model: int
        The dimension of virtual word embedding
    n_features: int
        The number of features in the input data
    n_layers: int
        The number of transformer encoder layers
    n_heads: int
        The number of attention heads
    n_epoches: int
        The number of training epoches
    update_lr_step: int
        The step of learning rate update
    pretrain_path: str
        The path of pretrain model
    seed: int
        The random seed
    train_batch_size: int
        batch size of train data
    valid_batch_size: int
        batch size of valid data
    early_stop_patience: int
        The patience of early stopping
    early_stop_min_delta: float
        The minimum delta of early stopping
    """
    with torch.cuda.device(device if device.type == "cuda" else -1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        model_path = f"models/scPert/{jobid}"
        mkDirs(model_path)
        for fold, (train_set, valid_set) in enumerate(dataset.kFold(seed=seed), 1):
            trainloader = DataLoader(train_set, train_batch_size, sampler=dataset.sampler(train_set, seed=seed))
            validloader = DataLoader(valid_set, valid_batch_size, sampler=dataset.sampler(valid_set, seed=seed))
            earlystop = EarlyStopping(
                path=f"models/scPert/{jobid}/{model_name}_{fold}.pt",
                patience=early_stop_patience,
                delta=early_stop_min_delta,
                verbose=10)
            
            if model_name == PertNet.__name__:
                model = PertNet(d_model=d_model, n_features=n_features, n_heads=n_heads, n_layers=n_layers, padding_size=padding_size).to(device)
            elif model_name == FFNet.__name__:
                model = FFNet(n_features=n_features).to(device)
            else:
                raise ValueError("Unsupport model type: %s"% model_name)

            if isinstance(pretrain_path, str):
                model = loadTorchModule(model, pretrain_path, skipped=False)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                                gamma=exp_scheduler_gamma)
            loop = tqdm(range(n_epoches))
            corr = validate(validloader, model, device)
            writer = SummaryWriter(f"logs/scPert/{jobid}/fold{fold}")
            for epoch in autoClearIter(loop):
                model.train()
                epoch_loss = []
                for index, batch in autoClearIter(enumerate(trainloader, 1)):
                    source = batch["source"].to(device)
                    target = batch["target"].to(device)
                    optimizer.zero_grad()
                    predict = model(source)
                    loss = model.criterion(predict, target, source)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.cpu().detach().item())
                    loop.set_description("Fold %02d batch %03d/%03d" %
                                        (fold, index, len(trainloader)))
                    loop.set_postfix(
                        train_loss=loss.cpu().detach().item(),
                        valid_loss=corr["loss"],
                        pearson=corr["pearson"],
                        pid=str(os.getpid()),
                        memo_res=visualBytes(
                            torch.cuda.memory_reserved(device)),
                        max_memo_res=visualBytes(
                            torch.cuda.max_memory_reserved(device)),
                        lr="%.06f" % (scheduler.get_last_lr()[0]))

                    # del data, label, score, loss
                if (epoch + 1) % update_lr_step == 0:
                    scheduler.step()
                corr = validate(validloader, model, device)
                earlystop(corr["loss"], model)
                writer.add_scalar(
                    tag="Train Loss", 
                    scalar_value=np.mean(epoch_loss),
                    global_step=epoch)
                writer.add_scalar(
                    tag="Valid Loss",
                    scalar_value=corr["loss"],
                    global_step=epoch
                )
                writer.add_scalar(
                    tag="Pearson corr",
                    scalar_value=corr["pearson"],
                    global_step=epoch
                )
                if earlystop.earlyStop:
                    print(f"Early stopping at epoch {epoch} for fold {fold}")
                    break
            writer.close()
            break


def independent_validate(
    model: NetBase, 
    output_path: str, 
    source: sc.AnnData, 
    target: sc.AnnData,
    device: torch.device,
    class_col: str = "cell_type",
    valid_batch_size: int = 256,
    figure_path: str = None,
    random_state: int = None) -> sc.AnnData:
    """
    Model validation for independent dataset

    Parameters
    ------------
    model: PertNet
        The pretrained perturbation network
    output_path: str
        The path to save the validation result
    source: sc.AnnData
        The source domain data
    target: sc.AnnData
        The target domain data
    device: torch.device
        The device used for training
    valid_batch_size: int
        The batch size for validation
    figure_path: str
        The path to save the validation figure
    random_state: int
        The random state for validation

    Returns
    ------------
    sc.AnnData
        The prediction result
    """
    log("Inferrence", "Begin model inferrence")
    mkDirs(output_path)
    if figure_path is None:
        figure_path = output_path
    sc.settings.figdir = figure_path
    pred_scPert = source.copy()
    pred_scPert.X = validate(
        DataLoader(PertDataset(source, source, class_col), batch_size=valid_batch_size),
        model=model,
        device=device)["predict"]

    pred_scPert.obs[config.CONDITION_COL] = "scPert"
    if "padding" in pred_scPert.uns:
        old_var = pred_scPert.var_names.drop(pred_scPert.uns["padding"])
        source = source[:, old_var]
        target = target[:, old_var]
        pred_scPert = pred_scPert[:, old_var]

    sc.write(f"{output_path}/predict.h5ad", pred_scPert)
    log("Inferrence", "Begin PCA visualization")
    merge = sc.concat([source, pred_scPert, target])
    merge.obsm.clear()
    merge.obs_names_make_unique()
    sc.set_figure_params(fontsize=14)
    sc.pp.neighbors(merge)
    sc.tl.pca(merge)
    sc.pl.pca(merge, color=[config.CONDITION_COL],
            legend_fontsize=14,
            palette=[config.CTRL_COLOR, config.PRED_COLOR, config.TARGET_COLOR],
            save=f"_predict_condtions[all].pdf",
            show=False, 
            frameon=False)

    sc.pl.pca(merge[merge.obs[config.CONDITION_COL]!="predict"], color=[config.CONDITION_COL],
            legend_fontsize=14,
            palette=[config.CTRL_COLOR, config.TARGET_COLOR],
            save=f"_predict_condtions[no predict].pdf",
            show=False,
            frameon=False)

    sc.pl.pca(merge[merge.obs[config.CONDITION_COL]!=config.TARGET_NAME], color=[config.CONDITION_COL],
            legend_fontsize=14,
            palette=[config.CTRL_COLOR, config.PRED_COLOR],
            save=f"_predict_condtions[no target].pdf",
            show=False,
            frameon=False)

    log("Inferrence", "Plot gene regresssion")
    sc.tl.rank_genes_groups(merge, groupby=config.CONDITION_COL, method="wilcoxon")
    diff_genes = merge.uns["rank_genes_groups"]["names"][config.TARGET_NAME][:100]
    plot_regression_plot(
        pred_scPert, 
        target, 
        diff_genes,
        x_label="Predict by scPert",
        y_label=config.TARGET_NAME,
        save_path=os.path.join(figure_path, "reg_mean_plot.pdf"),
        random_state=random_state)
    
    return pred_scPert