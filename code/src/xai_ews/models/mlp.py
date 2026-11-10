"""M2: plain fully connected network (MLP) for PD (methodology 6).

Inputs
    numeric      quantile transform to a normal shape, fitted on train (robust to the heavy tails
                 of UPB changes and excess principal); missing values -> 0 plus a 0/1 indicator
                 for every feature that has missing values on train
    categorical  integer codes (CategoryEncoder fitted on train) -> learned embeddings
Network      [numeric, indicators, embeddings] -> n_layers x (Linear -> ReLU -> Dropout) -> logit
Training     AdamW on BCE with row weights omega x 1/r (normalised to mean 1), the same as CatBoost;
             after every epoch the omega-weighted log-loss on validation half A is computed and
             training stops when it has not improved for PATIENCE epochs (the best epoch is kept).
Tuned (Optuna, 40 trials): layers 2-4, width, dropout, learning rate, weight decay, batch size,
embedding size. The best configuration is trained with 5 seeds.
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from torch import nn

MAX_EPOCHS = 40
PATIENCE = 3
EVAL_BATCH = 65536


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Preprocessor:
    def __init__(self, n_quantiles: int = 1000, subsample: int = 500_000, seed: int = 0):
        self.qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal",
                                      subsample=subsample, random_state=seed)
        self.missing_cols: np.ndarray | None = None

    def fit(self, x_num: np.ndarray) -> "Preprocessor":
        self.qt.fit(x_num)
        self.missing_cols = np.flatnonzero(np.isnan(x_num).any(axis=0))
        return self

    def transform(self, x_num: np.ndarray) -> np.ndarray:
        z = self.qt.transform(x_num).astype(np.float32)
        ind = np.isnan(x_num[:, self.missing_cols]).astype(np.float32)
        return np.hstack([np.nan_to_num(z, nan=0.0), ind])


@dataclass
class Config:
    n_layers: int = 3
    width: int = 256
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 4096
    emb_dim: int = 8


def suggest(trial) -> Config:
    return Config(n_layers=trial.suggest_int("n_layers", 2, 4),
                  width=trial.suggest_categorical("width", [64, 128, 256, 512]),
                  dropout=trial.suggest_float("dropout", 0.0, 0.5),
                  lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                  weight_decay=trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True),
                  batch_size=trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192]),
                  emb_dim=trial.suggest_categorical("emb_dim", [4, 8, 16]))


class MLP(nn.Module):
    def __init__(self, n_num: int, cardinalities: list[int], cfg: Config):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, cfg.emb_dim) for c in cardinalities])
        layers, d = [], n_num + cfg.emb_dim * len(cardinalities)
        for _ in range(cfg.n_layers):
            layers += [nn.Linear(d, cfg.width), nn.ReLU(), nn.Dropout(cfg.dropout)]
            d = cfg.width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        e = [m(x_cat[:, i]) for i, m in enumerate(self.emb)]
        return self.net(torch.cat([x_num, *e], dim=1)).squeeze(1)


def _tensors(x_num, x_cat, dev):
    return torch.as_tensor(x_num, device=dev), torch.as_tensor(x_cat, device=dev)


@torch.no_grad()
def predict_logit(model: MLP, x_num: np.ndarray, x_cat: np.ndarray) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for s in range(0, len(x_num), EVAL_BATCH):
        xn, xc = _tensors(x_num[s:s + EVAL_BATCH], x_cat[s:s + EVAL_BATCH], dev)
        out.append(model(xn, xc).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def train(x_num, x_cat, y, w, cardinalities: list[int], cfg: Config, eval_data=None,
          seed: int = 0, max_epochs: int = MAX_EPOCHS, log=None) -> tuple[MLP, dict]:
    """eval_data = (x_num, x_cat, y, w) of half A for early stopping. Returns the best model and
    the history {'epoch', 'eval_loss', 'best_epoch'}."""
    from xai_ews.evaluation.metrics import log_loss, sigmoid

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = device()
    model = MLP(x_num.shape[1], cardinalities, cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    xn, xc = _tensors(x_num, x_cat, dev)
    yt = torch.as_tensor(np.asarray(y, np.float32), device=dev)
    wt = torch.as_tensor(np.asarray(w, np.float32) / np.mean(w), device=dev)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    best, best_state, best_epoch, hist = np.inf, None, -1, []
    for epoch in range(max_epochs):
        model.train()
        perm = torch.as_tensor(rng.permutation(len(yt)), device=dev)
        for s in range(0, len(perm), cfg.batch_size):
            i = perm[s:s + cfg.batch_size]
            loss = (lossf(model(xn[i], xc[i]), yt[i]) * wt[i]).sum() / wt[i].sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if eval_data is None:
            continue
        en, ec, ey, ew = eval_data
        ll = log_loss(ey, sigmoid(predict_logit(model, en, ec)), ew)
        hist.append(ll)
        if log:
            log(epoch, ll)
        if ll < best - 1e-6:
            best, best_epoch, best_state = ll, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"eval_loss": hist, "best_epoch": best_epoch, "best_loss": best}
