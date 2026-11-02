from __future__ import annotations
from typing import *
from dataclasses import dataclass, field
import numpy as np


@dataclass
class GNNConfig:
    """Hyper-parameters for heterogeneous GNN training."""

    dim: int = 128
    feature_dim: int = 64  # learnable input embedding size per node
    hidden_dim: int = 256
    num_layers: int = 3
    heads: int = 4  # attention heads (HGT / HAN)
    dropout: float = 0.3
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 30  # early stopping patience
    min_delta: float = 1e-4  # minimum val-loss improvement
    neg_ratio: int = 5  # negatives per positive in InfoNCE
    val_ratio: float = 0.1  # fraction of edges held out for val
    temperature: float = 0.07  # InfoNCE temperature
    max_edges_ic: int = 200_000  # cap on INGREDIENT_COMPOUND edges
    device: str = "cpu"
    seed: int = 42


@dataclass
class GraphData:
    """Parsed graph in GNN-ready format."""

    node_names_by_type: Dict[str, List[str]]  # {type → [name, ...]}
    num_nodes_by_type: Dict[str, int]
    # All edge types incl. reverses; values: np.int64 (2, E)
    edge_index_dict: Dict[Tuple[str, str, str], np.ndarray]
    # Forward (non-rev) edge types only
    forward_edge_types: List[Tuple[str, str, str]]
    # PyG-style metadata = (node_types, edge_types)
    metadata: Tuple[List[str], List[Tuple[str, str, str]]]
    # Active node types present in this graph (subset of _ALL_NODE_TYPES)
    node_types: List[str] = field(default_factory=list)


class EarlyStopping:
    """Patience-based early stopping with best-model restoration."""

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: Optional[Dict] = None

    def step(self, val_loss: float, model) -> bool:
        """Call after each epoch. Returns True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model) -> None:
        """Restore the weights from the best epoch."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class RGCNHeteroEncoder:
    """
    Relational Graph Convolutional Network (Schlichtkrull et al., 2018).

    Approach: collapses the heterogeneous graph into a single global node
    table, assigns each edge a relation ID, and uses relation-specific
    (or basis-decomposed) weight matrices in each RGCNConv layer.

    Architectural family: *spectral / relational message-passing*,
    no attention, global parameter sharing across node types.
    """

    def __new__(cls, graph_data: GraphData, config: GNNConfig):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import RGCNConv

        node_types = list(graph_data.metadata[0])

        # Cumulative offsets for global node indexing
        offsets: Dict[str, int] = {}
        offset = 0
        for t in node_types:
            offsets[t] = offset
            offset += graph_data.num_nodes_by_type[t]
        total_nodes = offset

        # Relation IDs (forward + reverse)
        all_et = list(graph_data.edge_index_dict.keys())
        rel_to_id = {et: i for i, et in enumerate(all_et)}
        num_relations = len(all_et)

        hidden = config.hidden_dim
        feat = config.feature_dim
        dim = config.dim
        dropout = config.dropout
        num_layers = config.num_layers

        class _RGCNEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                # Per-type learnable input embeddings
                self.emb = nn.ModuleDict(
                    {
                        t: nn.Embedding(graph_data.num_nodes_by_type[t], feat)
                        for t in node_types
                    }
                )
                self.in_proj = nn.Linear(feat, hidden)
                self.convs = nn.ModuleList(
                    [RGCNConv(hidden, hidden, num_relations) for _ in range(num_layers)]
                )
                self.out_proj = nn.Linear(hidden, dim)
                self.dropout = dropout
                self._offsets = offsets
                self._rel_to_id = rel_to_id
                self._num_nodes_by_type = graph_data.num_nodes_by_type

            def _build_global_edges(self, edge_index_dict, device):
                all_ei, all_et_ids = [], []
                for et, ei in edge_index_dict.items():
                    if et not in self._rel_to_id:
                        continue
                    src_type, _, dst_type = et
                    src_off = self._offsets[src_type]
                    dst_off = self._offsets[dst_type]
                    ei_t = torch.from_numpy(ei).long().to(device)
                    global_ei = ei_t.clone()
                    global_ei[0] = global_ei[0] + src_off
                    global_ei[1] = global_ei[1] + dst_off
                    all_ei.append(global_ei)
                    rid = self._rel_to_id[et]
                    all_et_ids.append(
                        torch.full((ei.shape[1],), rid, dtype=torch.long, device=device)
                    )
                if not all_ei:
                    return (
                        torch.zeros(2, 0, dtype=torch.long, device=device),
                        torch.zeros(0, dtype=torch.long, device=device),
                    )
                return torch.cat(all_ei, dim=1), torch.cat(all_et_ids)

            def forward(self, edge_index_dict) -> Dict[str, torch.Tensor]:
                device = next(self.parameters()).device

                # Build global feature tensor
                xs = [F.relu(self.in_proj(self.emb[t].weight)) for t in node_types]
                x = torch.cat(xs, dim=0)  # (total_nodes, hidden)

                # Build global edge tensors
                edge_index, edge_type = self._build_global_edges(
                    edge_index_dict, device
                )

                for conv in self.convs:
                    x = F.relu(conv(x, edge_index, edge_type))
                    x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.out_proj(x)

                # Split back by type
                z_dict: Dict[str, torch.Tensor] = {}
                for t in node_types:
                    off = self._offsets[t]
                    n = self._num_nodes_by_type[t]
                    z_dict[t] = x[off : off + n]
                return z_dict

        return _RGCNEncoder()
