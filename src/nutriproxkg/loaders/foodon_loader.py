"""Loader for FoodOn ontology (Ingredient-Ingredient relations)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rdflib import Graph as RDFGraph
from rdflib import URIRef
from rdflib.namespace import RDFS
from tqdm import tqdm

from .base import LoaderBase, iter_table_rows, normalize_text


class FoodOnLoader(LoaderBase):
    """Parse FoodOn OWL/TSV files into ingredient hierarchy edges."""

    OWL_CANDIDATES = ["foodon.owl", "foodon-base.owl", "foodon_siren.owl"]
    TSV_CANDIDATES = ["foodon-synonyms.tsv"]

    def __init__(
        self, raw_dir: str | Path, logger: Optional[logging.Logger] = None
    ) -> None:
        super().__init__(raw_dir=raw_dir, logger=logger)

    @staticmethod
    def _is_foodon_uri(uri: str) -> bool:
        upper = uri.upper()
        return "FOODON_" in upper or "FOODON." in upper

    @staticmethod
    def _strip_uri_token(token: str) -> str:
        token = token.strip()
        if token.startswith("<") and token.endswith(">"):
            return token[1:-1]
        return token

    @staticmethod
    def _strip_literal(token: str) -> str:
        token = token.strip()
        match = re.match(r'"(.*)"(@[a-zA-Z\-]+)?$', token)
        if match:
            return match.group(1)
        return token.strip('"')

    def load(
        self,
        max_edges: Optional[int] = None,
    ) -> Tuple[Dict[str, Dict[str, str]], List[Tuple[str, str, Dict[str, str]]]]:
        """Load FoodOn ingredient nodes and subclass edges.

        Returns:
            nodes_by_uri, edges[(child_uri, parent_uri, attrs)]
        """
        owl_file = self.find_file(self.OWL_CANDIDATES)
        if owl_file:
            try:
                nodes, edges = self._load_from_owl(owl_file, max_edges=max_edges)
                if edges:
                    self.logger.info(
                        "FoodOn loaded from OWL: %d nodes, %d edges",
                        len(nodes),
                        len(edges),
                    )
                    return nodes, edges
            except Exception as exc:
                self.logger.warning(
                    "Failed parsing OWL %s (%s), trying TSV fallback", owl_file, exc
                )

        tsv_file = self.find_file(self.TSV_CANDIDATES)
        if tsv_file:
            nodes, edges = self._load_from_tsv(tsv_file, max_edges=max_edges)
            self.logger.info(
                "FoodOn loaded from TSV fallback: %d nodes, %d edges",
                len(nodes),
                len(edges),
            )
            return nodes, edges

        self.logger.warning("FoodOn files not found under %s", self.raw_dir)
        return {}, []

    def _load_from_owl(
        self,
        path: Path,
        max_edges: Optional[int],
    ) -> Tuple[Dict[str, Dict[str, str]], List[Tuple[str, str, Dict[str, str]]]]:
        rdf = RDFGraph()
        rdf.parse(str(path))

        label_by_uri: Dict[str, str] = {}
        for subject, label in tqdm(
            rdf.subject_objects(RDFS.label),
            desc="FoodOn labels",
            unit="triple",
        ):
            if not isinstance(subject, URIRef):
                continue
            uri = str(subject)
            if not self._is_foodon_uri(uri):
                continue
            clean = normalize_text(str(label))
            if clean:
                label_by_uri[uri] = clean

        edges: List[Tuple[str, str, Dict[str, str]]] = []
        nodes: Dict[str, Dict[str, str]] = {}

        for child, parent in tqdm(
            rdf.subject_objects(RDFS.subClassOf),
            desc="FoodOn subclass",
            unit="triple",
        ):
            if max_edges is not None and len(edges) >= max_edges:
                break
            if not isinstance(child, URIRef) or not isinstance(parent, URIRef):
                continue

            child_uri = str(child)
            parent_uri = str(parent)
            if not self._is_foodon_uri(child_uri) or not self._is_foodon_uri(
                parent_uri
            ):
                continue

            child_name = label_by_uri.get(child_uri)
            parent_name = label_by_uri.get(parent_uri)
            if not child_name or not parent_name:
                continue

            nodes.setdefault(
                child_uri,
                {
                    "name": child_name,
                    "FOODON.uri": child_uri,
                },
            )
            nodes.setdefault(
                parent_uri,
                {
                    "name": parent_name,
                    "FOODON.uri": parent_uri,
                },
            )
            edges.append(
                (
                    child_uri,
                    parent_uri,
                    {
                        "FOODON.child_uri": child_uri,
                        "FOODON.parent_uri": parent_uri,
                        "FOODON.relation": "subClassOf",
                    },
                )
            )

        return nodes, edges

    def _load_from_tsv(
        self,
        path: Path,
        max_edges: Optional[int],
    ) -> Tuple[Dict[str, Dict[str, str]], List[Tuple[str, str, Dict[str, str]]]]:
        label_by_uri: Dict[str, str] = {}
        parent_pairs: List[Tuple[str, str]] = []

        for row in iter_table_rows(path):
            class_uri = self._strip_uri_token(
                str(row.get("?class") or row.get("class") or "")
            )
            parent_uri = self._strip_uri_token(
                str(row.get("?parent") or row.get("parent") or "")
            )
            rel_type = normalize_text(row.get("?type") or row.get("type"))
            label = self._strip_literal(
                str(row.get("?label") or row.get("label") or "")
            )

            if class_uri and self._is_foodon_uri(class_uri):
                if rel_type.lower() == "label" and label:
                    label_by_uri[class_uri] = normalize_text(label)
                if parent_uri and self._is_foodon_uri(parent_uri):
                    parent_pairs.append((class_uri, parent_uri))

        nodes: Dict[str, Dict[str, str]] = {}
        edges: List[Tuple[str, str, Dict[str, str]]] = []
        for child_uri, parent_uri in parent_pairs:
            if max_edges is not None and len(edges) >= max_edges:
                break
            child_name = label_by_uri.get(child_uri)
            parent_name = label_by_uri.get(parent_uri)
            if not child_name or not parent_name:
                continue

            nodes.setdefault(child_uri, {"name": child_name, "FOODON.uri": child_uri})
            nodes.setdefault(
                parent_uri, {"name": parent_name, "FOODON.uri": parent_uri}
            )
            edges.append(
                (
                    child_uri,
                    parent_uri,
                    {
                        "FOODON.child_uri": child_uri,
                        "FOODON.parent_uri": parent_uri,
                        "FOODON.relation": "subClassOf",
                    },
                )
            )
        return nodes, edges
