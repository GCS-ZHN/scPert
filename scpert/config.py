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
import torch
import scanpy as sc
import numpy as np


SOURCE_NAME = "control"
TARGET_NAME = "stimulated"
CTRL_COLOR = "#96A1A3"
PRED_COLOR = "#A4E804"
TARGET_COLOR = "#009FC2"
CONDITION_COL = "condition"
GLOBAL_SEED = 42
N_TOP_GENES = 7000
METRIC = "cosine"
LEARNING_RATE = 1e-3
N_EPOCHES = 250
N_HEADS = 10
N_LAYERS = 6
D_MODEL = 140
UPDATE_LR_STEP = 10
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 256
EARLY_STOP_PATIENCE = 25
EARLY_STOP_MIN_DELTA = 0.01
EXP_SCHEDULER_GAMMA = 0.99

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=150, dpi_save=300, format="pdf")
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed(GLOBAL_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True