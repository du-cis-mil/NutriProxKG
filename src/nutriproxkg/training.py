from __future__ import annotations
from typing import *
from dataclasses import dataclass, field
import numpy as np
from .encoder import GNNConfig, EarlyStopping, RGCNHeteroEncoder
from .seed import set_seed
from .training_config import E2EConfig
import logging

logger = logging.getLogger(__name__)
RGCN_VARIANTS = {"rgcn_l3": {"num_layers": 3}}


def filter_edges(edge_index_dict: Dict, test_ing: np.ndarray, n_ing: int) -> Dict:
    """Drop every edge incident to a test ingredient node (both directions)."""
    test = np.zeros(n_ing, dtype=bool)
    test[test_ing] = True
    out = {}
    for (s, rel, d), ei in edge_index_dict.items():
        keep = np.ones(ei.shape[1], dtype=bool)
        if s == "Ingredient":
            keep &= ~test[ei[0]]
        if d == "Ingredient":
            keep &= ~test[ei[1]]
        out[(s, rel, d)] = np.ascontiguousarray(ei[:, keep])
    return out


def _build_general_model(model_type: str, gd, gnn_cfg, n_targets: int):
    import dataclasses as _dc
    import torch.nn as nn
    import torch.nn.functional as F

    encoders = {"rgcn": RGCNHeteroEncoder}
    if model_type in RGCN_VARIANTS:
        gnn_cfg = _dc.replace(gnn_cfg, **RGCN_VARIANTS[model_type])
        base = "rgcn"
    else:
        base = model_type
    if base not in encoders:
        raise ValueError(f"general model must be rgcn_l3, got {model_type!r}")
    encoder = encoders[base](gd, gnn_cfg)
    head = nn.Sequential(
        nn.Linear(gnn_cfg.dim, gnn_cfg.dim // 2),
        nn.ReLU(),
        nn.Dropout(gnn_cfg.dropout),
        nn.Linear(gnn_cfg.dim // 2, n_targets),
    )

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, edge_index_dict):
            z = self.encoder(edge_index_dict)
            ing = z["Ingredient"]
            return self.head(ing), ing

    return _Model()


def _train_loop(
    model,
    forward_train,
    forward_full,
    optimizer,
    tr_node,
    va_node,
    te_node,
    Y_tr,
    Y_va,
    is_a_edges,
    cfg: E2EConfig,
    fold: int,
):
    import torch
    import torch.nn.functional as F

    use_amp = bool(cfg.amp and str(cfg.device).startswith("cuda"))
    stopper = EarlyStopping(cfg.patience, cfg.min_delta)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            preds, repr_ = forward_train()
            sup = F.mse_loss(preds[tr_node].float(), Y_tr)
            aux = 0.0
            loss = (
                sup + cfg.aux_structural_weight * aux
                if (cfg.aux_structural_weight > 0 and is_a_edges is not None)
                else sup
            )
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                preds, _ = forward_train()
            val = F.mse_loss(preds[va_node].float(), Y_va).item()
        if not np.isfinite(val):
            raise ValueError(
                "Non-finite validation loss: check label units and training configuration"
            )
        if epoch % 100 == 0 or epoch <= 2:
            logger.info(
                "    [fold %d] epoch %4d val=%.4f best=%.4f pat=%d/%d",
                fold,
                epoch,
                val,
                stopper.best_loss,
                stopper.counter,
                cfg.patience,
            )
        if stopper.step(val, model):
            break
    stopper.restore(model)
    model.eval()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            preds, _ = forward_full()
    return preds[te_node].float().cpu().numpy()


def _inner_split(train_rows, cfg, fold):
    rng = np.random.default_rng(cfg.seed + fold)
    perm = rng.permutation(len(train_rows))
    n_val = max(1, int(len(train_rows) * cfg.inner_val_frac))
    return perm[n_val:], perm[:n_val]


def train_predict_general(
    gd,
    model_type,
    node_idx,
    train_rows,
    test_rows,
    Y_train_std,
    n_targets,
    cfg: E2EConfig,
    fold: int,
) -> np.ndarray:
    import torch

    set_seed(cfg.seed + fold)
    device = cfg.device
    gnn_cfg = GNNConfig(
        dim=cfg.hidden_dim,
        feature_dim=cfg.feature_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.n_ii_layers,
        heads=cfg.attn_heads,
        dropout=cfg.dropout,
        device=device,
        seed=cfg.seed,
    )
    model = _build_general_model(model_type, gd, gnn_cfg, n_targets).to(device)

    n_ing = gd.num_nodes_by_type["Ingredient"]
    test_ing = node_idx[test_rows]
    train_edges = filter_edges(gd.edge_index_dict, test_ing, n_ing)
    full_edges = gd.edge_index_dict

    inner_pos, val_pos = _inner_split(train_rows, cfg, fold)
    tr_node = torch.as_tensor(
        node_idx[train_rows[inner_pos]], dtype=torch.long, device=device
    )
    va_node = torch.as_tensor(
        node_idx[train_rows[val_pos]], dtype=torch.long, device=device
    )
    te_node = torch.as_tensor(node_idx[test_rows], dtype=torch.long, device=device)
    Y_tr = torch.as_tensor(Y_train_std[inner_pos], dtype=torch.float32, device=device)
    Y_va = torch.as_tensor(Y_train_std[val_pos], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    return _train_loop(
        model,
        lambda: model(train_edges),
        lambda: model(full_edges),
        optimizer,
        tr_node,
        va_node,
        te_node,
        Y_tr,
        Y_va,
        is_a_edges=None,
        cfg=cfg,
        fold=fold,
    )
