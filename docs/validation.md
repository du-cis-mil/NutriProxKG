# Local validation record

Prepared on 2026-11-02 (Asia/Tokyo). No GitHub upload, push, repository setting
change or publication operation was performed. All test artifacts were written
to disposable scratch directories outside this release. Raw files and historical
research sources were not edited.

## Environment and dependency checks

* Linux x86_64, Python 3.10.14.
* A completely new CPU virtual environment was created; all 42 external packages
  in `requirements-cpu.lock` were installed from their package indexes, including
  PyTorch 2.9.1+cpu and PyG 2.7.0. The release was then installed editable with
  `python -m pip install --no-deps -e .`. `python -m pip check` passed. This clean
  environment has no access to the research environment's site-packages.
* A separate existing CUDA environment was used for an additional short GPU
  check: PyTorch 2.9.1+cu128, PyG 2.7.0, NVIDIA GeForce RTX 4070 Ti. This does
  not validate an independent clean CUDA installation.
* The main CLI and all six subcommand `--help` paths passed. Missing raw inputs
  returned exit code 2 with the required source filename/path.
* CITATION.cff passed the official CFF 1.2.0 JSON schema. Undefined-name checks
  on all packaged Python files passed.

The CPU lock is a complete dependency version resolution for this platform, not
a cross-platform guarantee or a hash-verified wheel archive. CUDA direct package
versions are pinned, but no full CUDA transitive lock is supplied.

## Synthetic execution

The independent fixture generator supplied 24 fictional foods, a labelled OWL
hierarchy, original species/genus/parent taxon records, non-target compounds and
all 19 positive nutrient targets. It ran outside the release folder.

Both Nutri-3 and Nutri-19 completed source validation, source preprocessing,
track-specific masked graph construction, all five Task 1 inference methods,
and Task 2 training/restored-graph inference/evaluation. Smoke learning used two
seeds (42/43), two folds and two epochs, input/hidden width 8, CPU without AMP.
Results and resolved configs were produced as JSON; aggregation generated
`summary.csv` and `summary.json`. No fixture records or generated result files
are included here.

A clean CPU command sequence (including help checks and failure-path checking)
completed 18 commands in approximately 23 seconds. Individual timings vary with
process/import overhead and are not estimates for full-size experiments.
The separate CUDA/bfloat16 Nutri-19 synthetic Task 2 check completed all four
seed/fold training runs in approximately 1.9 seconds after imports.

The 20 semantic tests passed in the clean CPU environment. They cover:

* Missing sources and malformed CSV headers.
* Exact/family nutrient masks, target-column removal from profiles, and graph
  construction independent of the numerical label CSV.
* Genus search including its start node and search cutoff; the legacy scientific-name
  merge of distinct taxon IDs; original ICS anchors,
  ancestor closure and relation removal; similarity thresholds, canonical pairs and the no-eligible-profile case.
* Forward/inverse consistency and removal of every test-food incident edge.
* Shared shuffled folds, disjoint train/test rows and once-per-seed test coverage.
* Equal probability across active typed-walk channels.
* Self-excluded perfect retrieval metrics, asymmetric euACC bounds and clipping.
* Seed-level population standard deviation and training-fold-only mean prediction.
* Both target tracks executing actual R-GCN optimization and returning finite
  predictions with exact target dimensions and no water output.
* Rejection of stale graph configuration.

These tests establish executable pathways and selected semantic invariants. They
do not establish numerical reproduction, optimal preprocessing, correct biological
labels, complete alias coverage or the absence of every possible indirect signal.

## Real-source checks

The available local FooDB CSV dump, FoodOn OWL with versionIRI 2023-02-17 and
NCBI taxonomy files were inspected read-only and passed the source validator.
The portable preprocessor completed on the full available source files. It
recovered 22,784 ingredients, 70,516 compounds, 4,800,089 containment records and
23,417 FoodOn hierarchy records. Its complete-case labels matched the existing
327-food Nutri-19 and 755-food Nutri-3 CSVs exactly in values and row order.
The legacy-compatible taxonomy projection recovered 763 retained nodes,
559 genus anchors and 621 subtaxon edges. The Nutri-19 masked cosine relation
contained 2,420 pairs and matched the existing masked-similarity artifact exactly
when compared by ingredient name.

The full-source preprocessor took 214.56 seconds with maximum RSS 2,895,064 KiB
(about 2.76 GiB); the Nutri-19 masked build took 8.05 seconds with maximum RSS
1,086,744 KiB (about 1.04 GiB). These timings include this host's I/O/cache state
and must not be treated as universal requirements. RDFLib warned about malformed
date/time annotations in the historical OWL; the consumed labelled subclass
hierarchy was parsed successfully.

For additional local validation only, the existing historical 149/275 common-food
JSON lists were read as extra inputs. They were not reconstructed from the three
sources and are not distributed. This is a material limitation of the following
checks: using those lists does not establish fresh reconstruction of the paper's
comparison intersection.

On both local common-food lists (149 Nutri-19 / 275 Nutri-3), all four graph
Task 1 methods and direct cosine ran on the rebuilt inputs. Their nDCG@10/Recall@10
matched the corresponding rounded NutriProxKG/direct-cosine cells in Table 3(a):
20 cells in total, including both metrics for five methods on two tracks. Task 1
took about 92 seconds for Nutri-19 and 191 seconds for Nutri-3. Real-graph
CUDA/bfloat16 Task 2 checks completed two seeds, two folds, two epochs and width 8
on each list, with finite three-/19-output predictions (about nine seconds per
track). Result aggregation also completed for these real-data checks. This short run is not the paper's 600-epoch, five-seed,
five-fold, width-128 training and is not used to claim reproduced Task 2 scores.

No raw sources, historical food lists, labels, graphs, predictions, experiment
metrics files or runtime logs from these checks are included in this release.

## Release-folder inspection

The final folder audit passed with 53 text or empty files. No raw-source files,
real-data derived artifacts, binary/large files, symlinks, personal mounted paths,
private-key blocks or known credential-token patterns were flagged. Disposable
Python/build metadata was removed. This is a heuristic and manual inspection,
not a guarantee that every possible secret format can be detected.

The release .gitignore was tested as the root ignore file in a separate local
scratch repository: raw CSV, processed outputs, model files and .env were ignored,
while the data/output .gitkeep files remained includable. The parent research
repository's ignore rules were not used as a substitute for this check.

## Commands to repeat

Run the full synthetic command block in README.md and:

```bash
python -m unittest discover -s tests -v
python scripts/audit_release.py --clean-generated
```

With your own sources in the documented layout:

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

The full 600-epoch/5-seed/5-fold experiments and the paper's 149/275-food
comparison have not been rerun with this release. The partial Task 1 rounding checks above must not be extended to a claim of
reproducing every paper result. See reproducibility.md for excluded comparisons,
known preprocessing concerns and author decisions.
