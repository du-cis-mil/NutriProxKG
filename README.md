# NutriProxKG

Research code for **NutriProxKG: Structuring Taxonomic and Chemical Proximity for Nutrition-Aware Food Ingredient Representation** by Naoki Yoshimaru and Kenji Hatano.

NutriProxKG combines taxonomic relationships and chemical similarity in a food ingredient knowledge graph. This repository includes graph construction and two evaluation tasks:

- **Task 1 — Nutritional similarity retrieval.** Rank ingredients by graph proximity and evaluate against nutrient-based rankings.
- **Task 2 — Nutrient prediction.** Use a relational graph convolutional network (R-GCN) to predict nutrient content for held-out ingredients.

## Installation

Use Python 3.10. From the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

## Data

Obtain FooDB, FoodOn, and NCBI Taxonomy and place the required files under `data/raw/` as described in the [data setup guide](docs/data.md). The source datasets are distributed by their respective providers.

## Run the experiments

From the repository root, preprocess the inputs once, then build and evaluate the graph for each nutrient set:

```bash
nutriproxkg validate --config configs/nutri19.yaml
nutriproxkg preprocess --config configs/nutri19.yaml

for track in nutri19 nutri3; do
  nutriproxkg build --config "configs/${track}.yaml"
  nutriproxkg task1 --config "configs/${track}.yaml"
  nutriproxkg task2 --config "configs/${track}.yaml"
done

nutriproxkg aggregate --outputs outputs
```

Results are written to `outputs/`. The default configurations evaluate all FooDB foods with complete labels for each nutrient set. The [reproducibility notes](docs/reproducibility.md) describe the paper's shared food subsets and the scope of the included experiments.

## Citation and license

See [CITATION.cff](CITATION.cff) for citation details and [LICENSE](LICENSE) for the MIT license. Third-party terms are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
