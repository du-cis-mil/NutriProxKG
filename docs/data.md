# Data acquisition and layout

Users must obtain the following three sources themselves. No download, login or
paid API call is performed by the pipeline. Source files and derived artifacts
must remain outside the distributed code. Official pages were checked on
2026-11-02; availability and terms may change.

| Source | Official acquisition point | Paper version / known provenance | Required local files |
|---|---|---|---|
| FooDB: The Food Database | [Downloads](https://foodb.ca/downloads) | CSV dump dated 2020-04-07; manuscript bibliography says accessed 2026-05-14 | `data/raw/foodb/Food.csv`, `Compound.csv`, `Nutrient.csv`, `Content.csv` |
| FoodOn food ontology | [Project](https://foodon.org/), [official repository and releases](https://github.com/FoodOntology/foodon) | Available local OWL identifies release 2023-02-17; exact historical paper input/acquisition date not proven | `data/raw/foodon/foodon.owl` |
| NCBI Taxonomy | [Official taxonomy FTP index](https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/) | Exact snapshot and acquisition date not established | `data/raw/ncbi/nodes.dmp`, `names.dmp`, `merged.dmp` |

FooDB's indexed official downloads page lists the April 7, 2020 CSV tar.gz.
A direct page fetch returned HTTP 403 during this preparation; use the page in
your browser, rather than assuming that an unverified mirror is equivalent.
Extract the archive locally and copy the four CSV tables into the exact layout
above, retaining capitalization. Other archive tables are not required.

The inspected local FoodOn OWL has `owl:versionIRI` ending in
`foodon/releases/2023-02-17/foodon.owl`. The local NCBI dump files have filesystem
modification dates of 2024-10-25; these dates alone do not authenticate the snapshot
or its acquisition date. The real-source validation used these available local
files without redistributing them.

For FoodOn, choose the author's historical release if available; otherwise record
which release you selected. Download the OWL artifact and save it as `foodon.owl`.
This adapter reads labelled FOODON URI classes connected by explicit
`rdfs:subClassOf` triples. It does not perform OWL reasoning, follow imports,
expand synonyms, or infer restrictions. Both endpoints need FoodOn identifiers
and labels. A TSV fallback exists in the retained research loader, but the release adapter
calls its OWL parser explicitly and never silently substitutes a TSV input.

For NCBI, download `taxdump.tar.gz` from the official index and extract the three
`.dmp` files into `data/raw/ncbi/`. They are pipe-delimited text, with whitespace
around fields. `merged.dmp` must exist but may be empty. The parser uses taxon ID,
parent ID and rank from `nodes.dmp`; `names.dmp` provides scientific names for compatibility with the original
node-identity merge, described in reproducibility.md. Names are not used to infer
ingredient-to-taxon links. Ingredient taxon IDs come
from FooDB. The retained legacy policy resolves merged IDs one step; the release
reports unavailable anchors rather than inventing a name-based mapping.

## Required schema and joins

| File | Required fields | Other fields used when available |
|---|---|---|
| Food.csv | `id`, `name`, `ncbi_taxonomy_id` | `name_scientific`, public/category metadata in retained loader |
| Compound.csv | `id`, `name` | `public_id`, `cas_number`, chemical-class metadata |
| Nutrient.csv | `id`, `name` | `dfc_name`, `dfc_id` |
| Content.csv | `food_id`, `source_id`, `source_type`, `standard_content`, `orig_unit` | `id`, `orig_content`, `orig_unit_expression` |
| foodon.owl | URI subjects/objects for `rdfs:subClassOf`, `rdfs:label` | Other triples are ignored |
| nodes.dmp | fields 1–3: taxon ID, parent ID, rank | Remaining fields ignored |
| names.dmp | taxon ID, name, unique name, name class | Scientific-name node identity; not ingredient alignment |
| merged.dmp | old ID, current ID | One-step resolution |

CSV uses a header, comma separator and UTF-8. Internal FooDB identifiers join
`Content.food_id` to `Food.id`; `source_type` distinguishes `Compound` and
`Nutrient`, and `source_id` joins the matching table. Graph identity merges exact
names after lowercasing and collapsing whitespace, within each node type.
Unknown source types or unmatched joins are skipped as in the research loader;
counts and source-unit frequencies are recorded in `preprocess.json`.

The nutrient extractor takes the median of converted content records per
food/nutrient. It disables water estimation, filters missing/mostly-zero nutrients
at 50%, and retains complete cases. The exact 3/19 target lists are enforced:
no water, and iodine retained for Nutri-19. A different dump that cannot supply
these targets fails rather than silently creating a different task. The legacy
unit conversion deserves author review; see reproducibility.md. There is no
missing-label zero imputation in this release.

## Conditions and citations

* FooDB's official [compliance page](https://foodb.ca/compliance) states CC
  Attribution-NonCommercial 4.0 for data and metadata. The downloads page also
  directs users to usage/distribution terms. Check the terms applicable to the
  chosen dump yourself; this repository supplies no redistribution permission.
  Cite FooDB, *FooDB: The Food Database*, version 2020-04-07,
  [official site](https://foodb.ca/), as in the manuscript bibliography.
* FoodOn's official [license](https://github.com/FoodOntology/foodon/blob/master/LICENSE.txt)
  is CC BY 4.0. Inspect notices for imported ontology terms in the chosen release.
  Cite Dooley et al., *FoodOn: a harmonized food ontology to increase global food
  traceability, quality control and data integration*, npj Science of Food (2018),
  [DOI](https://doi.org/10.1038/s41538-018-0032-6).
* Consult [NCBI policies](https://www.ncbi.nlm.nih.gov/home/about/policies/),
  which distinguish US-government information from contributed material and
  request acknowledgment. Do not equate every linked resource with unrestricted
  data. Cite Schoch et al., *NCBI Taxonomy: a comprehensive update on curation,
  resources and tools*, Database (2020),
  [DOI](https://doi.org/10.1093/database/baaa062).

Changed FoodOn classes, NCBI parents/ranks/merged IDs or FooDB records can change
coverage, graph structure, folds, labels and results. The pipeline stores SHA-256
hashes and sizes of every source file in `preprocess.json`; graph and experiment
metadata link back to this provenance. Keep the selected source release dates
alongside these files locally. Hashes identify files; they do not establish that
a file is the historical paper input.

## Additional inputs for the paper's comparison

The three sources are sufficient to build NutriProxKG/ICS and run both tasks on
all complete-case FooDB foods. They are **not** sufficient to reconstruct the
four-graph intersection used in Table 3 (275 Nutri-3 / 149 Nutri-19 foods).
That requires FlavorGraph and official FoodKG as well, with the historical name
alignment and versions. The research tree contains their loaders and builds,
but they are not packaged here: their multi-source build and historical snapshot
provenance have not been independently validated for this release.

The official source projects are [FlavorGraph](https://github.com/lamypark/FlavorGraph)
and [FoodKG](https://foodkg.github.io/foodkg.html). Their original inputs, recipe and
USDA data, and their separate terms must be checked before extending this release.
No comparison files or food-list excerpts are bundled. If you independently
reconstruct the intersection, set `foods: data/raw/evaluation_foods.json` in a
config. The format is `{"common_foods": ["exact food name", "..."]}` or a JSON
list. This is a schema example, not a runnable paper food list. The loader
preserves nutrient CSV order, rejects missing/duplicate identities, and saves the
actual ordered foods with every result. Supplying a list alone does not verify
its provenance or guarantee the paper's numeric results.
