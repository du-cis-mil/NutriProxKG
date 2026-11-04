# Reproducibility scope and manuscript correspondence

This release was prepared from `IEICE-DEIM---N.Yoshimaru/main_revise.pdf`
and the local implementation, with priority given to `src_4_revised`. The PDF
includes reviewer-response material; manuscript section numbers below refer to
the research article, not the appended responses. Neither the PDF nor historical
experiment outputs are redistributed. `source_manifest.json` identifies original
code files and their source hashes; files in the release have been extracted,
renamed, formatted, and connected with new portable adapters.

## What is executable

| Paper section / experiment | Research implementation inspected | Release implementation / config | Command / status |
|---|---|---|---|
| §3 alignment, FoodOn backbone, FooDB links | `src/graph_construction/loaders/{foodb,foodon}_loader.py`, `graph_builder.py` | `loaders/`, `pipeline.prepare`; exact normalized-name matching | `nutriproxkg preprocess --config configs/nutri19.yaml`; synthetic verified |
| §3.2 condensation | `src_chemtaxfoodkg/build_graphs.py::_find_genus`, `build_genus_cos90`; revised B-3 taxonomy variants | `pipeline.taxonomy`; rank genus, first 20 inspected nodes, parent steps L=2 | `nutriproxkg build --config configs/nutri19.yaml`; synthetic verified |
| §3.3 chemical proximity | original `_build_compound_sets`, `e2e/build_similar_to_variants.py`; revised B-3 `cosine_edges` | `similarity.py`, `pipeline.build`; binary positive-content sets, theta=0.90 | same build command; recomputed after target masking |
| §4.1 Nutri-3 / Nutri-19 labels | `src/evaluation/build_nutrition_track.py`, FooDB nutrient extractor; revised full-scale validation `canonical_labels` | `pipeline.prepare`, `nutrient_aliases.targets_for_track` | preprocess makes both CSVs; exact 3/19 outputs enforced |
| §4.2 / Table 3(a), Fig. 2: Task 1 | `experiments/model_free_nutritional_similarity_retrieval/src/` | `retrieval/`, `cli.task1`, both YAML configs | `nutriproxkg task1 --config configs/nutri19.yaml`; all four methods runnable |
| Table 3(a) non-graph calibration | revised B-4 `review_b_04_non_graph_baselines.py` | `cli.task1`, direct cosine on all remaining profile columns | included by default, zero profiles retained |
| §4.2 / Table 3(b): Task 2 | `e2e/inductive.py::train_predict_general`, `common/hetero_gnn_trainer.py::RGCNHeteroEncoder`, `e2e/evaluate.py`; revised full-scale protocol | `training.py`, `encoder.py`, `evaluation.py`, `cli.task2` | `nutriproxkg task2 --config configs/nutri19.yaml`; short synthetic training verified |
| Table 3(b) train-fold mean | revised B-4 non-graph baseline | `training.model: train_mean` | copy YAML, change model and output; same task2 command |
| ICS comparator | `e2e/graph_build.py::build_ics_typed` | `graph.variant: ics`, source-taxonomy lineages (at most 50 nodes, including the original anchor), no similar_to | use a separate processed/output directory; synthetic verified; historical node ordering not reproduced |
| §5.1 / final Fig. 3: masked relation-removal probes | track-specific masked Task 1 ablation outputs in the research tree | `drop_relations` accepts any of five forward relation names | masked build/task commands can express the conditions; synthetic edge-filter semantics verified; numerical reproduction of all nine conditions on both tracks is not recorded for this release |
| §5 sensitivity | revised B-3 and `sensitivity_validation/` | configurable threshold, rank, L, minimum compounds, search limit | copy YAML per condition, use separate processed/output roots; grid orchestration / paper figure generation not included |
| Experiment summaries | old dashboards/recompute scripts | `cli.summarize` | `nutriproxkg aggregate --outputs outputs`; CSV and JSON verified |

Module paths under `src/nutriproxkg/` are used in the release column.
Configurations for both tracks use the same model, fold generator and seed list.
A supplied evaluation list is filtered in nutrient CSV order, so graph variants
use identical row indices if they use the same preprocessed labels and list.
Every Task 2 result retains explicit per-seed fold train/test rows, reference
values, predictions and train statistics; each food is predicted once per seed.

## Final Fig. 3 versus historical unmasked ablations

The final revised manuscript's **Fig. 3 uses target-nutrient-masked Task 1
retrieval on both Nutri-3 and Nutri-19**, with typed PPR, nDCG@10 and Recall@10.
All nine conditions use the corresponding track's target exclusion. `Full`
retains all four relation families and matches the NutriProxKG typed-PPR row in
Table 3(a) of the manuscript. This manuscript correspondence is not a claim that
the released code has been numerically validated for every Fig. 3 bar.

The four families are Compound (`has_compound`), Genus (`from_genus` together with
`subtaxon_of`), Comp-sim (`similar_to`), and Ontology (`is_a`). An `only` condition
retains the named family and drops the other three; a `w/o` condition drops the
named family or families and retains the others. The nine conditions are Full,
Compound only, Genus only, Comp-sim only, Ontology only, w/o Genus, w/o Comp-sim,
w/o Ontology, and w/o Genus+Sim. In particular, w/o Genus drops both `from_genus`
and `subtaxon_of`; w/o Genus+Sim also drops `similar_to`. All conditions retain
the nodes. The release's `drop_relations` filters the five forward relation
names before adding inverse relations (`pipeline.load_graph`); `pipeline.build`
recomputes similarity after the track-specific target mask. These interfaces
can express the masked conditions using separate processed/output roots and the
same supplied evaluation food list. Grid orchestration, the paper's saved
nine-condition outputs and figure export are not included.

Earlier manuscript versions used **unmasked historical Task 1 ablations** for
Fig. 3. Those results are superseded by the final masked figure and must not be
used as its inputs or described as its evaluation protocol. Separately, the
structural coherence diagnostic in Fig. 4(a) retains its pre-masking saved graph
and 327-food nutrient-CSV scope; it is not the masked Fig. 3 experiment.

The executed checks remain those recorded in [validation.md](validation.md).
They establish synthetic relation-filter semantics and, using locally supplied
historical common-food lists, rounded agreement for **20 Table 3(a) cells**:
four graph retrieval methods plus direct cosine, two metrics, and two tracks,
for NutriProxKG/direct cosine. Those food lists were not reconstructed or
redistributed. The record does **not** establish numerical reproduction of all
nine Fig. 3 conditions, the other graphs' Table 3 cells, or the full Task 2
published scores. The short Task 2 runs check execution and finite outputs,
not the paper's full training results. No historical graphs, labels, rankings,
predictions or experiment-result files are distributed in this repository.

## Task definitions and scoring

**Task 1: learning-free graph proximity / nutritional-similarity retrieval.**
Inputs are typed graph structure and an ordered complete-case food set. Nutrient
values are used only to create the evaluation gold ranking: log1p, per-column
z-score (population std; zero variance becomes zero after centering), and cosine.
Self candidates are excluded. Gold rank r in 1–20 receives linear gain 21−r,
otherwise zero. nDCG uses those gains directly, not an exponential gain. Recall@k
compares the retrieved k foods with gold top-k (clipped to the candidate count).
Outputs include nDCG@5/10/20, Recall@5/10/20, auxiliary metrics, per-query scores
and candidate-index rankings (indices refer to the saved food list). There is no
trained model, fold or seed variance for deterministic Task 1.

Vanilla PPR uses a symmetrized binary adjacency; typed PPR divides probability
equally across relation families present at each node and then across neighbours
within each family. Restart=0.15, maximum 100 iterations, tolerance=1e-6.
Projection uses Adamic–Adar with global entity degree, equally averaged after
z-scoring the off-diagonal scores per relation. Meta-path uses PathSim for each
available I–X–I relation, also an equal z-mean. Direct compound cosine uses the
same masked positive-content profiles, no threshold, top-k cap or dimension
reduction. Ties use the fixed candidate order.

**Task 2: label-inductive nutrition prediction (label-inductive GNN).**
Inputs are the masked typed graph and training-food labels in g/100g. A 3-layer
R-GCN uses learned 64-dimensional node embeddings, hidden/output width 128,
a 128→64→3/19 regression head and dropout 0.4. There are no nutrient values or
compound quantities passed as GNN input features. Edges incident to outer-test
ingredients are removed in both directions for training and inner validation,
and restored for prediction. Held-out nodes remain in the embedding tables;
this is the article's label-inductive setting with auxiliary relations available
at inference, not a strict node-inductive or zero-information evaluation.

Defaults: seeds 42–46, 5 shuffled folds, full-batch Adam, lr=.002,
weight_decay=1e-4, maximum 600 epochs, inner validation 15%, patience 60,
min_delta=1e-4. Model initialization and inner split use seed+fold. Labels are
standardized using the outer training fold, including its inner validation rows,
as in the research implementation. Complete-case inputs are enforced before CV,
so the retained train-only imputer has nothing to fill. Inference inverse-scales
the predictions to g/100g. CPU defaults disable AMP; to match the historical
CUDA/bfloat16 execution, explicitly set `device: cuda` and `amp: true`.

The euACC score treats max(prediction,0) as a declared value, and checks whether
the true value lies within its lower/upper band. Vitamins use −35%/+50%, non-sodium
minerals (including iodine) −35%/+45%. Carbohydrate/protein: ±2 g below 10 g,
±20% from 10 through 40 g, ±8 g above 40 g. Fat has ±1.5 g below 10 g and the
same other intervals. Sodium: ±0.15 g below 0.5 g and ±20% otherwise. Lower bounds
are clipped to zero. This is a tolerance-inspired prediction metric, not a full
regulatory conformity test. RMSE uses unclipped predictions. Both metrics average
nutrients and folds equally within each seed, then report the five-seed mean and
population std (ddof=0). No standard deviation treats 25 folds as independent.

## Unresolved differences and effects on numerical reproduction

1. **Historical input identity and comparison coverage.** The paper's 149/275
   common food sets depend on FlavorGraph and FoodKG. Their data and derived
   food lists are omitted. Default `foods: null` evaluates all complete cases and
   cannot be identified with Table 3. Neither exact FoodOn/NCBI snapshots nor
   complete historical per-graph label snapshots are established. Updated inputs
   can change counts and results. The old 327/755 label counts, 22,784 ingredients,
   763 retained taxonomic nodes and 559 anchors were matched on the available local sources. They are not universal assertions
   for different source versions, and they do not authenticate all historical inputs.
2. **Taxonomy identity compatibility.** The original `MultiTypeGraphBuilder`
   merges Species nodes by normalized scientific name, just as it merges other
   node types by name. Different NCBI IDs with the same name can consequently
   share a node whose ID/parent/rank attributes become string lists. The portable
   adapter reproduces that attribute representation without materializing the
   whole NCBI NetworkX graph. On the inspected local files, 1,250 name-collision
   groups were found. An initial ID-only adapter yielded 785 retained nodes,
   570 anchors and 644 subtaxon edges; reproducing the legacy name merge yields
   the paper's 763/559/621. The ID-only adapter was replaced before finalization;
   its outputs are not the release's construction policy. This is compatibility
   with the existing implementation, not a claim that homonymous taxa should
   biologically be merged. Authors should consider this limitation separately
   from any future corrected, ID-preserving experiment.
3. **Source ordering.** The adapter assigns deterministic local indices by
   sorted normalized ingredient/compound names and taxon IDs. The old pipeline
   allocated graph IDs on insertion and sorted those IDs. Random embedding
   initialization and tied rankings can consequently differ even with identical
   source files. This is a documented portability choice, not numeric parity.
   It requires a full-source comparison before claiming reproduction.
4. **Five-compound eligibility.** The final manuscript's Section 3.3 and the
   historical masked-similarity builder choose >=5 distinct positive compounds
   **before** masking at both endpoints, then recompute cosine after masking.
   The release preserves that order. Applying eligibility after masking would
   define different inputs and may change both tasks; no such change is made.
5. **Top-256 before masking.** The existing Task 2 loader caps containment edges
   before dropping target-nutrient edges. The release preserves this ordering,
   quantity-prioritized selection and graph seed 42. Consequently target-associated
   records may consume cap positions; target amounts can indirectly affect which
   other edges survive. Direct target edges are absent, but strict absence of
   target-dependent preprocessing cannot be claimed. Authors should decide whether
   to change the order and rerun every affected comparator. Such a correction would
   define different experimental inputs and is not presented as a cleanup or a
   reproduction of existing scores. Task 1 and direct cosine have no such cap.
6. **Containment versus positive profiles.** All joined content records, including
   missing/zero quantities, remain containment edges, as in the old IC tensors.
   Only finite raw `standard_content > 0` enters cosine profiles. The manuscript's
   phrase “one per measured pair” is narrower than these stored containment
   records. Duplicate records remain in containment arrays, while profile pairs
   are binary. PPR binarizes adjacency; R-GCN can retain multiplicities.
7. **Native similarity record count.** The stored native graph contains 3,390
   similarity records but only 3,370 unique ingredient pairs. Inspection found
   3,370 `compound_cosine` records and 20 additional `genus_jaccard` records on
   already represented pairs. Thus the native Table 2 count is a multigraph
   record count, whereas the formal one-pair definition describes unique pairs.
   The existing target-masked recomputation and this release use the recomputed
   unique cosine pairs. The native records are not reintroduced into the masked
   evaluation. The final manuscript distinguishes stored records from unique pairs
   in Section 3.3 and describes binary PPR adjacency and unique masked R-GCN pairs
   in Section 4.2.
8. **Label units.** The retained FooDB extractor prefers `standard_content` but
   converts it according to `orig_unit` (falling back to `orig_content` when blank).
   Missing units use a value-dependent heuristic; only supported units are kept.
   If `standard_content` is already a standardized mg quantity regardless of the
   original unit, this can mis-scale labels. The local research code does not
   establish a universal interpretation across source records. Author verification
   against the dump documentation and old label snapshots is necessary; unit
   conversion was not silently rewritten and no label-correctness claim is made.
9. **Model configuration.** The paper's rgcn_l3 architecture and default numerical
   hyperparameters are retained. CPU versus CUDA/AMP, library updates, graph IDs,
   edge order and nondeterministic GPU reductions may change results. The tests
   check semantics and executable training, not equivalence to reported scores.
10. **Publication and copyright.** The PDF inspected during release preparation
   had future journal/header metadata and a DOI-like value. Those template values
   are not independently established publication facts.
   CITATION.cff records the manuscript without assigning an invented year, DOI,
   volume or pages. Code copyright names are inferred from the two authors and
   require their confirmation before publication.

## Excluded experiments found in the research tree

| Existing implementation / article coverage | Release status and reason |
|---|---|
| FlavorGraph; FoodKG-50k and FoodKG-full, Tables 2/3 | Not packaged: historical multi-source build, exact comparison intersection and raw-source provenance need a separate validation effort. Official acquisition projects are linked in data.md. |
| HAN/HGT, older KGE encoders, 2-layer/hidden-256 runs | Present in research tree; excluded to keep the release focused on the article's reported R-GCN. No claim to reproduce these auxiliary runs. |
| Type-specialized learned encoder and branch ablations (Fig. 4c) | Present in `e2e/model.py`, `branches.py`, `inductive.py`; not packaged. Attention/quantity branches are a separate diagnostic model; R-GCN relation drops are not equivalent to those branch ablations. |
| Final masked Task 1 ablation grid (Fig. 3) | Individual conditions are expressible with the released masked pipeline and relation filters. Grid orchestration, saved nine-condition outputs and figure export are not packaged; full Fig. 3 numerical reproduction is not established by the release validation record. |
| Legacy unmasked Task 1 ablations; structural coherence bootstrap and anchored-food confounding | The unmasked Task 1 bars belong to earlier manuscript versions, not final Fig. 3. The separate diagnostics are present in the research model-free runner and revised B-5/B-9; their saved graphs, pair sets and historical CSV scope are not packaged. |
| Paired Wilcoxon / Holm tests, Table of paired comparisons | Revised B-1/B-8 and full FoodKG revalidation retain analyses. Not packaged: complete four-baseline outputs and shared provenance are required; testing only a subset would change the Holm family. |
| Full sensitivity grid / plotting | Revised B-3 and `sensitivity_validation` exist. Individual conditions are runnable via config; grid orchestration, statistics and figure export are excluded and not claimed verified. |
| Independent nutrient-source validation | Revised B-2 exists but requires an independent source and explicit alignment/provenance; unavailable here, not replaced by FooDB reuse. |

The newer full-scale FoodKG validation report (2026-10-16) was inspected after the
older blue-revision audit: it records completed masked three-/19-output reruns,
replacing the older mixture of unmasked runs and a water-containing 20-output run.
This release enforces exact targets globally, preventing that class of accidental
extra-output evaluation. This safeguard does not recalculate or validate the
historical comparison results, and their data/logs are not copied.

## Author review before publication

Confirm the code copyright grant, publication status, exact FoodOn/NCBI releases,
source-unit semantics, pre-mask candidate eligibility, top-k/masking order, and
whether the reduced scope meets the reviewer commitment. If exact Table 3
reproduction is required, supply validated comparison builders and reconstruct
the common food lists locally; then rerun all methods with shared split manifests.
Only the checks in validation.md have been executed for this release.

Publication-oriented guards were added for empty eligible-profile sets (returning
no similarity edges rather than raising inside normalization), non-finite
validation losses/predictions, explicit tolerance files, stale input/config
fingerprints, conflicting anchors and exact target dimensions. None of these
guards changes the successful nonempty real-source computations checked above.
The empty-profile guard has no effect on the paper inputs; no revised paper
score is inferred from these defensive changes.
