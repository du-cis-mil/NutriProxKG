from __future__ import annotations
from typing import *
from dataclasses import dataclass, field
import numpy as np

REL = Tuple[str, str, str]


@dataclass
class MFGraph:
    key: str
    label: str
    num_nodes_by_type: Dict[str, int]
    ing_names: List[str]
    relations: Dict[REL, np.ndarray]  # forward typed edges (2, E) int64 (no rev_)
    type_offset: Dict[str, int]  # node type -> start index in the global id space
    n_total: int  # total nodes across all types

    # -- helpers ------------------------------------------------------------
    def ingredient_global(self, local_ids: np.ndarray) -> np.ndarray:
        return np.asarray(local_ids, dtype=np.int64) + self.type_offset["Ingredient"]

    def relation_names(self) -> List[str]:
        return [rel for (_, rel, _) in self.relations]

    def select(self, keep_rel_names: List[str]) -> "MFGraph":
        """Return a copy keeping only the relations whose name is in ``keep_rel_names``."""
        keep = set(keep_rel_names)
        rels = {et: ei for et, ei in self.relations.items() if et[1] in keep}
        return MFGraph(
            self.key,
            self.label,
            dict(self.num_nodes_by_type),
            list(self.ing_names),
            rels,
            dict(self.type_offset),
            self.n_total,
        )
