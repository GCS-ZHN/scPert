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
import torch
import scpert.config as config

from argparse import ArgumentParser
from scpert.data import source_target_multiclass_alignment, PertDataset, padding_data
from scpert.visualize import plot_alignment
from scpert.model import FFNet, PertNet
from scpert.training import train, independent_validate
from socube.utils import getJobId, loadTorchModule, log


if __name__ == "__main__":
    parser = ArgumentParser("Perturbation Prediction Network")
    # parser.add_argument("input", help="Input data path", type=str)
    parser.add_argument("--jobid", help="Job ID", type=str, default=None)
    parser.add_argument("--predict-mode", help="Predict mode", action="store_true")
    parser.add_argument("--baseline", action="store_true", help="Run baseline FC net")
    parser.add_argument("--top-gene", type=int, default=7000, help="Top high variablely genes")
    args = parser.parse_args()
    if args.predict_mode!=(args.jobid is not None):
        print("Error: --predict-mode and --jobid must be used together")
        exit(1)
    config.N_TOP_GENES = args.top_gene
    jobid = getJobId() if args.jobid is None else args.jobid
    sc.settings.figdir = "figures/scPert/%s" % jobid
    log("Preprocess", "Begin preprocess with jobid: %s" % jobid)
    pbmc_train = sc.read_h5ad("datasets/scGEN/train_pbmc_clear.h5ad")
    cell_types = ["CD4T", "CD14+Mono", "FCGR3A+Mono", "Dendritic", "NK", "B", "CD8T"]
    if config.N_TOP_GENES >= pbmc_train.shape[1]:
        log("Preprocess", "Data padding")
        pbmc_train = padding_data(pbmc_train, config.N_TOP_GENES - pbmc_train.shape[1])
        padding_size = config.N_TOP_GENES - pbmc_train.shape[1]
    else:
        log("Preprocess", "Find highly variable genes")
        sc.pp.highly_variable_genes(pbmc_train, n_top_genes=config.N_TOP_GENES, subset=True)
        padding_size = 0
    log("Preprocess", "UMAP visualization")
    sc.set_figure_params(fontsize=14)
    sc.pp.neighbors(pbmc_train)
    sc.tl.umap(pbmc_train)
    sc.pl.umap(pbmc_train, color=[config.CONDITION_COL],
            legend_fontsize=14,
            palette=[config.CTRL_COLOR, config.TARGET_COLOR],
            save=f"_conditions.pdf", 
            show=False,
            frameon=False)

    sc.set_figure_params(fontsize=14)
    sc.pp.neighbors(pbmc_train)
    sc.pl.umap(pbmc_train, color=["cell_type"],
            legend_fontsize=14,
            save=f"_celltypes.pdf",
            show=False,
            frameon=False)

    pbmc_train_source = pbmc_train[pbmc_train.obs[config.CONDITION_COL] == config.SOURCE_NAME].copy()
    pbmc_train_target = pbmc_train[pbmc_train.obs[config.CONDITION_COL] == config.TARGET_NAME].copy()
    log("Preprocess", "Data alignment")
    if not args.predict_mode:
        balance_source, balance_target = source_target_multiclass_alignment(
            pbmc_train_source, 
            pbmc_train_target, 
            mertic=config.METRIC,
            top_frac=1.0,
            seed=config.GLOBAL_SEED
            )
        plot_alignment(
            balance_source,
            balance_target,
            save_path=f"figures/scPert/{jobid}/data_alignment.pdf")

        sc.write(f"datasets/scGEN/train_pbmc_balance_source[{jobid}].h5ad", balance_source)
        sc.write(f"datasets/scGEN/train_pbmc_balance_target[{jobid}].h5ad", balance_target)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for select_cell in cell_types:
        if not args.predict_mode:
            dataset = PertDataset(
                source=balance_source[balance_source.obs.cell_type != select_cell].copy(), 
                target=balance_target[balance_target.obs.cell_type != select_cell].copy())
            log("Train", f"Train model for {select_cell}")
            train(
                jobid=jobid+"/"+select_cell, 
                dataset=dataset,
                padding_size=padding_size,
                device=device,
                lr=config.LEARNING_RATE,
                d_model=config.D_MODEL,
                n_features=config.N_TOP_GENES,
                n_layers=config.N_LAYERS,
                n_heads=config.N_HEADS,
                n_epoches=config.N_EPOCHES,
                update_lr_step=config.UPDATE_LR_STEP,
                seed=config.GLOBAL_SEED,
                train_batch_size=config.TRAIN_BATCH_SIZE,
                valid_batch_size=config.VALID_BATCH_SIZE,
                early_stop_patience=config.EARLY_STOP_PATIENCE,
                early_stop_min_delta=config.EARLY_STOP_MIN_DELTA,
                exp_scheduler_gamma=config.EXP_SCHEDULER_GAMMA,
                model_name= FFNet.__name__ if args.baseline else PertNet.__name__
                )

        log("Inferrence", f"Prediction for {select_cell}")
        if args.baseline:
            model = FFNet(config.N_TOP_GENES)
        else:
            model = PertNet(config.D_MODEL, config.N_TOP_GENES, config.N_HEADS, n_layers=config.N_LAYERS)
        loadTorchModule(
            model, 
            f"models/scPert/{jobid}/{select_cell}/{model.__class__.__name__}_1.pt",
            skipped=False)
        independent_validate(
            model=model,
            output_path=f"outputs/scPert/{jobid}/{select_cell}",
            figure_path=f"figures/scPert/{jobid}/{select_cell}",
            source=pbmc_train_source[pbmc_train_source.obs.cell_type == select_cell].copy(),
            target=pbmc_train_target[pbmc_train_target.obs.cell_type == select_cell].copy(),
            device=device,
            valid_batch_size=config.VALID_BATCH_SIZE,
            random_state=config.GLOBAL_SEED)
