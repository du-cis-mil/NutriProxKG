# NutriProxKG

Research code for **NutriProxKG: Structuring Taxonomic and Chemical Proximity for
Nutrition-Aware Food Ingredient Representation**, by **Naoki Yoshimaru and Kenji
Hatano**, revised manuscript prepared for *IEICE Transactions on Information and
Systems*. Acceptance status and final publication metadata have not been verified;
the future journal header and DOI-like value in the working PDF are not treated
as a publication record. See [CITATION.cff](CITATION.cff).

NutriProxKG combines a FoodOn ingredient hierarchy with FooDB composition and
NCBI biological taxonomy. It connects mapped ingredients directly to a genus,
retains up to two parent steps above the genus, and adds unweighted ingredient
links at binary compound-profile cosine >=0.90.

This folder contains executable source preprocessing, NutriProxKG and ICS graph
construction, both evaluation tasks, and result aggregation. It is a locally
prepared release candidate. **It does not reproduce the entire paper:** the
FlavorGraph/FoodKG comparison builds, their historical common-food lists and
several diagnostic analyses are excluded. The default runs use all complete-case
FooDB foods, rather than the article's 149/275 common sets. Known source-unit and
preprocessing-order concerns are documented in
[reproducibility.md](docs/reproducibility.md); Task 1 cells have been checked against paper rounding using available local
historical common-food lists; full comparative and Task 2 numerical reproduction
have not been completed.

## Tasks

| Task | Input and purpose | Output / main metrics |
|---|---|---|
| Task 1: learning-free graph proximity (nutritional-similarity retrieval) | Rank other ingredients from graph structure; compare to a separate log1p/z-score/cosine gold ranking from FooDB nutrients | Per-query rankings, nDCG@k and Recall@k; vanilla PPR, typed PPR, Adamic–Adar projection, PathSim meta-path, plus direct compound cosine |
| Task 2: label-inductive nutrition prediction | Train a 3-layer R-GCN on training-food labels; remove test-food incident edges during training and restore them only for prediction | Out-of-fold nutrient predictions in g/100g, euACC and RMSE, fold and seed summaries; optional train-fold mean baseline |

Nutri-3 predicts carbohydrate, fat and protein. Nutri-19 adds ten minerals
(including iodine) and six vitamins; water is excluded from both. All targets are
explicitly selected. See [the protocol and scope table](docs/reproducibility.md)
for the exact targets, metrics, aggregation and manuscript correspondence.

## Environment and installation

Tested on Linux with Python 3.10.14, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3,
scikit-learn 1.7.2, RDFLib 7.5.0, PyYAML 6.0.3, PyTorch 2.9.1 and PyG 2.7.0.
Direct versions are pinned in `pyproject.toml`; `requirements-cpu.lock` also pins
the complete Linux/Python 3.10 CPU dependency resolution. Python 3.10 is the validated
choice; other declared compatible Python versions have not been exercised.

From this directory (the future repository root):

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m pip check
nutriproxkg --help
```

For a fully pinned CPU dependency environment, use this in place of the torch
and project install lines above (Linux x86_64, Python 3.10):

```bash
python -m pip install -r requirements-cpu.lock --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps -e .
```

For NVIDIA CUDA 12.8, replace the CPU torch install with:

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

Then set `training.device: cuda` and `training.amp: true` in your YAML. CUDA needs
a compatible NVIDIA driver; bfloat16 AMP also needs suitable hardware. A system
CUDA toolkit is not required for the prebuilt torch wheel. See the official
[PyTorch version instructions](https://pytorch.org/get-started/previous-versions/)
and [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html).
This pipeline uses basic `RGCNConv`; PyG's optional compiled scatter/sparse
extensions are not prerequisites. No language models, external experiment
trackers, credentials, online APIs or paid services are needed for execution.
Validation details and environment limits are in [validation.md](docs/validation.md).

## Supply the three data sources

Download and extract the data yourself; acquisition instructions, official terms,
version limits, citations and field requirements are in [data.md](docs/data.md).
Do not commit source or derived files.

```text
data/raw/
  foodb/   Food.csv  Compound.csv  Nutrient.csv  Content.csv
  foodon/  foodon.owl
  ncbi/    nodes.dmp  names.dmp  merged.dmp
```

FooDB uses the 2020-04-07 CSV dump (bibliography access date 2026-05-14). The available
local FoodOn OWL identifies release 2023-02-17; its historical input identity and
the NCBI snapshot remain author-confirmation items. Changes to
these sources may change the graph and experiment results. No actual source rows,
processed labels, graphs, embeddings, trained weights or experiment logs are
included. The empty data/output directories contain only `.gitkeep` files.

## Execute the pipeline

All relative data paths in a config resolve against its repository root, defined
as the parent of the directory containing the YAML. Keep configs directly under
`configs/`. Run commands from the repository root.

```bash
nutriproxkg validate --config configs/nutri19.yaml
nutriproxkg preprocess --config configs/nutri19.yaml
nutriproxkg build --config configs/nutri19.yaml
nutriproxkg task1 --config configs/nutri19.yaml
nutriproxkg task2 --config configs/nutri19.yaml
nutriproxkg build --config configs/nutri3.yaml
nutriproxkg task1 --config configs/nutri3.yaml
nutriproxkg task2 --config configs/nutri3.yaml
nutriproxkg aggregate --outputs outputs
```

Preprocessing generates label tables for both tracks once. Builds produce a
separate masked graph per track. Task 1 is learning-free and requires no training
command. Task 2 performs learning, restored-graph inference and evaluation in the
same command. The default 600-epoch, 5-fold, 5-seed Task 2 runs are full experiments;
the short synthetic checks below do not validate their numeric paper results.

To use the paper's evaluation intersection, independently reconstruct it using
the additional comparison sources and set `foods` to its JSON path; the schema is
in data.md. No generated list in this release should be mistaken for the
historical 149/275-food sets.

For the train-mean baseline, copy a config and set `training.model: train_mean`
and a distinct `output`. For ICS set `graph.variant: ics`, with distinct
`processed` and `output` paths, then rerun preprocessing/build. Relation-removal
experiments use `drop_relations`, for example `[from_genus, subtaxon_of]`.
Threshold, rank, upper parent count, profile eligibility, seeds, folds, dimensions,
optimization and output paths are explicit YAML settings. Changing experimental
conditions defines a new run; use separate processed/output roots. The graph
fingerprint is checked to reject stale preprocessing inputs.

## Synthetic smoke check

The generator creates independent fictional FooDB-like CSV, FoodOn-like OWL and
NCBI-like text. It does not extract rows from the source databases. Use an empty
scratch directory; it is kept outside this release:

```bash
scratch_dir=$(mktemp -d)
python scripts/make_synthetic.py --root "$scratch_dir"
nutriproxkg preprocess --config "$scratch_dir/configs/full.yaml"
for track in n3 full; do
  nutriproxkg validate --config "$scratch_dir/configs/$track.yaml"
  nutriproxkg build --config "$scratch_dir/configs/$track.yaml"
  nutriproxkg task1 --config "$scratch_dir/configs/$track.yaml"
  nutriproxkg task2 --config "$scratch_dir/configs/$track.yaml"
done
nutriproxkg aggregate --outputs "$scratch_dir/outputs"
python -m unittest discover -s tests -v
python scripts/audit_release.py --clean-generated
```

The smoke config has 24 fictional foods, two folds, seeds 42/43, two epochs and
width 8. It verifies executable training/inference and output generation, not
scientific validity on real food data. Tests cover masks, typed transitions,
cosine thresholding, genus search, fold separation, train-only statistics and
metrics. `python -m nutriproxkg` is equivalent to the console command.

## Outputs and inspection

| Location | Contents |
|---|---|
| `data/processed/preprocess.json` | Raw file hashes/sizes, dependency versions, source-unit/join counts, label coverage |
| `data/processed/base.npz`, `profiles.npz`, `entities.json` | Containment/hierarchy arrays, binary positive-content profiles, identity and taxonomy maps |
| `data/processed/labels_n3.csv`, `labels_full.csv` | Complete-case nutrient labels, no estimated water |
| `data/processed/graph_<track>.npz/.json`, `profiles_<track>.npz` | Track-specific graph, mask, construction config, provenance and masked profiles |
| `outputs/main/<track>/task1/result.json` | Food order, per-method averages, query metrics and rankings |
| `outputs/main/<track>/task2/result.json` | Exact targets, every seed's folds/reference/prediction, aggregate euACC/RMSE |
| each task's `resolved_config.json` | Effective config; result metadata also includes seed settings, dependencies, code hashes and input hashes |
| `outputs/summary.csv`, `summary.json` | Aggregated results, with one row per task/method/run |

Task 2 means and standard deviations are computed over seed-level CV scores
(population std, ddof=0). Each seed averages folds and nutrients equally. RMSE is
in original units and uses unclipped predictions. euACC alone clips negative
predictions to zero before constructing asymmetric tolerance bounds.

## Resources and troubleshooting

Small CPU smoke tests take seconds to tens of seconds depending on installation.
On the available local sources, preprocessing took about 215 seconds with
2.76 GiB peak resident RAM, and the Nutri-19 masked graph build took about eight
seconds with 1.04 GiB peak resident RAM. These are observations on this host, not
minimum resource requirements. Full-training runtime and peak VRAM remain unmeasured. It parses the full NCBI parent/rank map and retains all joined
FooDB content records, so source preprocessing can need substantial RAM. Task 2
caps containment to top-256 per ingredient; Task 1 remains uncapped. Full-batch
R-GCN on the real graph is intended for a GPU. There is no measured minimum GPU
capacity for this release; choose a reduced smoke configuration before allocating
a full run. Do not interpret unrelated FoodKG-full timings as NutriProxKG timings.

* Missing files/columns: run `validate`; use the exact extracted filenames and
  schema from data.md. The command names the required path or column.
* RDFLib date/time-literal warnings: the available historical FoodOn file has
  malformed annotation literals. They are not used for subclass/label extraction;
  distinguish annotation warnings from a fatal OWL parse failure.
* No complete cases / unexpectedly sparse targets: verify the 2020-04-07 dump and
  units. The release refuses to silently drop an expected Nutri-19 target.
* CUDA unavailable / unsupported bfloat16: select CPU and disable AMP, or install
  the matching wheel and verify `torch.cuda.is_available()`.
* Stale graph/config: rebuild after changing preprocessing or graph conditions.
* Existing `result.json`: set a new output root. Successful results are not
  overwritten. A failed task may leave an empty directory; it is not a result.
* Out of memory: do not silently lower graph caps or model dimensions and call
  the result a paper reproduction. A smaller config is a separately labelled run.
* Undefined auxiliary correlations on tiny/constant synthetic profiles become
  JSON `null`; the main retrieval metrics remain defined.

## License and citation

Authors' code is prepared under the [MIT License](LICENSE). Copyright-holder
confirmation remains required before publication. External data and dependencies
retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Cite the manuscript using the verified authors/title in [CITATION.cff](CITATION.cff)
and cite the three data sources. Do not assign an unconfirmed DOI or publication
status. [Reproducibility notes](docs/reproducibility.md) list excluded experiments,
known inconsistencies and the remaining author decisions.
