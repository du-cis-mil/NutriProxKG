"""Semantic checks on synthetic data only; no external source files required."""

import dataclasses
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import yaml
from nutriproxkg.pipeline import (
    prepare,
    build,
    load_graph,
    load_labels,
    validate,
    taxonomy,
    edges,
)
from nutriproxkg.cli import configuration, task1, task2, summarize
from nutriproxkg.training import filter_edges
from nutriproxkg.cv import make_kfold
from nutriproxkg.nutrient_aliases import masked_compound_ids, targets_for_track
from nutriproxkg.eu_tolerance import compliance_matrix, load_tolerance_config
from nutriproxkg.retrieval.nutrient_gold import build_gold
from nutriproxkg.retrieval.metrics import ndcg_at_k, recall_at_k
from nutriproxkg.retrieval.graph_preprocess import typed_transition
from nutriproxkg.retrieval.load_graphs import MFGraph
from nutriproxkg.similarity import cosine_edges
from nutriproxkg.evaluation import aggregate_seed_blocks
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "synthetic", ROOT / "scripts/make_synthetic.py"
)
synthetic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(synthetic)


class ResearchInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = synthetic.generate(cls.tmp.name)
        (cls.root / "configs").mkdir()
        cfg = yaml.safe_load((ROOT / "configs/nutri19.yaml").read_text())
        cfg["tolerance"] = str(ROOT / "configs/eu_tolerance.yaml")
        cfg["training"].update(
            seeds=[42, 43], epochs=2, k_folds=2, feature_dim=8, hidden_dim=8, patience=2
        )
        (cls.root / "configs/test.yaml").write_text(yaml.safe_dump(cfg))
        cls.cfg = configuration(cls.root / "configs/test.yaml")
        prepare(cls.cfg)
        build(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_missing_input_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "Food.csv"):
            validate(self.root / "nonexistent")

    def test_bad_csv_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            synthetic.generate(directory)
            p = Path(directory) / "data/raw/foodb/Content.csv"
            p.write_text("food_id\n1\n")
            with self.assertRaisesRegex(ValueError, "missing columns"):
                validate(Path(directory) / "data/raw")

    def test_mask_covers_all_targets_but_not_unrelated_salts(self):
        self.assertEqual(
            masked_compound_ids(
                ["Protein", "Iodine", "Sodium chloride", "Retinol"], "full"
            ),
            {0, 1, 3},
        )
        self.assertEqual(masked_compound_ids(["Protein", "Iodine"], "n3"), {0})

    def test_genus_search_counts_start_and_obeys_limit(self):
        e = {
            "parent": {"1": "2", "2": "3", "3": "3"},
            "rank": {"1": "species", "2": "genus", "3": "family"},
            "anchors": {"0": "1", "1": "2"},
        }
        ids, fg, ss = taxonomy(e, "genus", 1, 1)
        self.assertEqual(fg[0].tolist(), [1])
        self.assertEqual(ids, ["2", "3"])
        self.assertEqual(ss.shape, (2, 1))
        ids, fg, ss = taxonomy(e, "genus", 0, 2)
        self.assertEqual(fg.shape, (2, 2))
        self.assertEqual(ids, ["2"])

    def test_symmetric_cosine_pairs_threshold(self):
        X = csr_matrix([[1, 1, 0], [1, 1, 0], [1, 0, 1]])
        np.testing.assert_array_equal(
            cosine_edges(X, np.ones(3, bool), 0.9), [[0], [1]]
        )

    def test_no_eligible_profiles_keeps_graph_constructible(self):
        result = cosine_edges(
            csr_matrix((3, 2), dtype=np.float32), np.zeros(3, bool), 0.9
        )
        self.assertEqual(result.shape, (2, 0))
        self.assertEqual(result.dtype, np.int64)

    def test_graph_target_edges_removed_and_inverse_correct(self):
        gd, e, info = load_graph(self.cfg, 2)
        masked = info["masked_compounds"]
        hc = gd.edge_index_dict["Ingredient", "has_compound", "Compound"]
        self.assertFalse(np.isin(hc[1], masked).any())
        for s, r, d in gd.forward_edge_types:
            np.testing.assert_array_equal(
                gd.edge_index_dict[d, "rev_" + r, s], gd.edge_index_dict[s, r, d][::-1]
            )

    def test_holdout_edges_removed_all_directions(self):
        gd, _, _ = load_graph(self.cfg, 2)
        filtered = filter_edges(
            gd.edge_index_dict, np.array([0, 2]), gd.num_nodes_by_type["Ingredient"]
        )
        for (s, r, d), ei in filtered.items():
            if s == "Ingredient":
                self.assertFalse(np.isin(ei[0], [0, 2]).any())
            if d == "Ingredient":
                self.assertFalse(np.isin(ei[1], [0, 2]).any())

    def test_fold_reuse_and_separation(self):
        first = list(make_kfold(24, 5, 42).split(np.arange(24)))
        second = list(make_kfold(24, 5, 42).split(np.arange(24)))
        seen = []
        for (tr, te), (tr2, te2) in zip(first, second):
            self.assertFalse(set(tr) & set(te))
            np.testing.assert_array_equal(te, te2)
            seen.extend(te)
        self.assertEqual(sorted(seen), list(range(24)))

    def test_typed_walk_balances_relation_mass(self):
        rels = {
            ("Ingredient", "a", "Compound"): edges([(0, 0), (0, 1)]),
            ("Ingredient", "b", "Species"): edges([(0, 0)]),
        }
        graph = MFGraph(
            "x",
            "x",
            {"Ingredient": 1, "Compound": 2, "Species": 1},
            ["x"],
            rels,
            {"Ingredient": 0, "Compound": 1, "Species": 3},
            4,
        )
        row = typed_transition(graph).toarray()[0]
        np.testing.assert_allclose(row, [0, 0.25, 0.25, 0.5])

    def test_metrics_perfect_rank_and_self_exclusion(self):
        gold = build_gold(["a", "b", "c"], np.array([[1], [2], [3]]), topk=[1, 2])
        self.assertTrue(np.isneginf(np.diag(gold.sim)).all())
        self.assertEqual(ndcg_at_k(gold.order[0], gold.relevance[0], 2), 1)
        self.assertEqual(recall_at_k(gold.order[0], gold.topk[2][0], 2), 1)

    def test_eu_asymmetry_and_negative_clipping(self):
        tol = load_tolerance_config(self.cfg["tolerance"])
        comp = compliance_matrix(
            np.array([[1.49], [1.51], [0.64]]), np.ones((3, 1)), ["vitamin_c_g"], tol
        )
        np.testing.assert_array_equal(comp[:, 0], [1, 0, 0])
        self.assertEqual(
            compliance_matrix(
                np.array([[0.0]]), np.array([[-3.0]]), ["protein_g"], tol
            )[0, 0],
            1,
        )

    def test_standard_deviation_uses_seeds(self):
        a = {
            "status": "ok",
            "euACC_mean_over_nutrients": 1.0,
            "rmse_mean_over_nutrients": 2.0,
            "per_nutrient": {},
        }
        b = dict(a, euACC_mean_over_nutrients=3.0, rmse_mean_over_nutrients=6.0)
        agg = aggregate_seed_blocks([a, b], [42, 43], [])
        self.assertEqual(agg["euACC_seed_std"], 1.0)
        self.assertEqual(agg["rmse_seed_std"], 2.0)

    def test_task1_end_to_end_no_training(self):
        result = task1(self.cfg)
        self.assertEqual(len(result["results"]), 5)
        for scores in result["results"].values():
            self.assertTrue(0 <= scores["mean"]["ndcg@10"] <= 1)
            for i, row in enumerate(scores["rankings"]):
                self.assertNotIn(i, row)

    def test_task2_trains_both_tracks_and_outputs_exact_targets(self):
        for track in ["n3", "full"]:
            cfg = dict(self.cfg, track=track)
            build(cfg)
            result = task2(cfg)
            self.assertEqual(result["targets"], targets_for_track(track))
            self.assertNotIn("water_g", result["targets"])
            self.assertEqual(result["result"]["n_seeds"], 2)
            for seed in result["per_seed"]:
                self.assertTrue(np.isfinite(seed["prediction"]).all())

    def test_mean_baseline_is_train_only(self):
        cfg = dict(self.cfg)
        cfg["training"] = dict(self.cfg["training"], model="train_mean")
        result = task2(cfg)
        for seed in result["per_seed"]:
            for fold in seed["folds"]:
                expected = seed["reference"][fold["train_rows"]].mean(axis=0)
                np.testing.assert_allclose(
                    seed["prediction"][fold["test_rows"]],
                    np.tile(expected, (len(fold["test_rows"]), 1)),
                    rtol=1e-5,
                    atol=1e-7,
                )

    def test_legacy_taxonomy_name_merge_preserves_attribute_semantics(self):
        from nutriproxkg.pipeline import legacy_taxonomy_identity

        parent = {"1": "1", "10": "1", "11": "1"}
        rank = {"1": "no rank", "10": "genus", "11": "genus"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "names.dmp"
            path.write_text(
                "10 | Shared genus | | scientific name |\n11 | shared GENUS | | scientific name |\n"
            )
            remap, collisions = legacy_taxonomy_identity(parent, rank, path)
        self.assertEqual(collisions, 1)
        self.assertEqual(remap["10"], remap["11"])
        self.assertEqual(remap["10"], "['10', '11']")
        self.assertEqual(rank[remap["10"]], "genus")
        self.assertNotIn("10", parent)

    def test_ics_and_relation_removal(self):
        cfg = dict(self.cfg, processed=str(self.root / "data/processed-ics"))
        cfg["graph"] = dict(self.cfg["graph"], variant="ics")
        prepare(cfg)
        build(cfg)
        gd, entities, info = load_graph(cfg, 1)
        self.assertNotIn(("Ingredient", "similar_to", "Ingredient"), gd.edge_index_dict)
        self.assertIn("100", info["species"])
        self.assertIn("1", info["species"])
        cfg["drop_relations"] = ["from_genus", "subtaxon_of"]
        gd, _, _ = load_graph(cfg, 2)
        self.assertFalse(
            any("genus" in et[1] or "subtaxon" in et[1] for et in gd.edge_index_dict)
        )

    def test_masked_profiles_and_label_independent_graph(self):
        from scipy.sparse import load_npz

        gd, _, info = load_graph(self.cfg, 1)
        X = load_npz(Path(self.cfg["processed"]) / "profiles_full.npz")
        self.assertEqual(X[:, info["masked_compounds"]].nnz, 0)
        label_path = Path(self.cfg["processed"]) / "labels_full.csv"
        original = label_path.read_text()
        import pandas as pd

        labels = pd.read_csv(label_path)
        labels.iloc[:, 1:] *= 1000
        labels.to_csv(label_path, index=False)
        try:
            build(self.cfg)
            rebuilt, _, _ = load_graph(self.cfg, 1)
            for et, ei in gd.edge_index_dict.items():
                np.testing.assert_array_equal(ei, rebuilt.edge_index_dict[et])
        finally:
            label_path.write_text(original)

    def test_changed_graph_config_requires_rebuild(self):
        cfg = dict(self.cfg, graph=dict(self.cfg["graph"], cosine_threshold=0.8))
        with self.assertRaisesRegex(ValueError, "run build again"):
            load_graph(cfg, 1)


if __name__ == "__main__":
    unittest.main()
