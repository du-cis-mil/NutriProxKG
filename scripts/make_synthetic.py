"""Generate independent fictional source files, never excerpts of external data."""

import argparse
import csv
from pathlib import Path
import yaml

MICROS = [
    "calcium",
    "copper",
    "iodine",
    "iron",
    "manganese",
    "phosphorus",
    "potassium",
    "selenium",
    "sodium",
    "zinc",
    "retinol",
    "thiamine",
    "riboflavin",
    "folic acid",
    "ascorbic acid",
    "tocopherol",
]


def table(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def generate(root):
    root = Path(root)
    raw = root / "data/raw"
    table(
        raw / "foodb/Food.csv",
        ["id", "name", "ncbi_taxonomy_id"],
        [
            {
                "id": i,
                "name": f"fictional food {i:02d}",
                "ncbi_taxonomy_id": 100 + i % 3,
            }
            for i in range(24)
        ],
    )
    comps = [{"id": i, "name": f"fictional compound {i}"} for i in range(12)] + [
        {"id": 100 + i, "name": n} for i, n in enumerate(MICROS)
    ]
    table(raw / "foodb/Compound.csv", ["id", "name"], comps)
    table(
        raw / "foodb/Nutrient.csv",
        ["id", "name"],
        [
            {"id": i, "name": n}
            for i, n in enumerate(["carbohydrate", "fat", "protein"])
        ],
    )
    rows = []
    for i in range(24):
        for j in range(3):
            rows.append(
                {
                    "food_id": i,
                    "source_id": j,
                    "source_type": "Nutrient",
                    "standard_content": (i + 2) * (j + 1) * 80,
                    "orig_unit": "mg/100g",
                }
            )
        for j in range(16):
            rows.append(
                {
                    "food_id": i,
                    "source_id": 100 + j,
                    "source_type": "Compound",
                    "standard_content": (i + 1) * (j + 1) * 0.001,
                    "orig_unit": "mg/100g",
                }
            )
        for j in range(6):
            rows.append(
                {
                    "food_id": i,
                    "source_id": (j + (i % 3) * 3) % 12,
                    "source_type": "Compound",
                    "standard_content": 1 + j,
                    "orig_unit": "mg/100g",
                }
            )
    table(
        raw / "foodb/Content.csv",
        ["food_id", "source_id", "source_type", "standard_content", "orig_unit"],
        rows,
    )
    ncbi = raw / "ncbi"
    ncbi.mkdir(parents=True)
    nodes = (
        [("1", "1", "no rank"), ("2", "1", "order"), ("3", "2", "family")]
        + [(str(10 + i), "3", "genus") for i in range(3)]
        + [(str(100 + i), str(10 + i), "species") for i in range(3)]
    )
    (ncbi / "nodes.dmp").write_text(
        "".join(f"{a}\t|\t{b}\t|\t{r}\t|\n" for a, b, r in nodes)
    )
    (ncbi / "names.dmp").write_text(
        "".join(
            f"{a}\t|\tfictional taxon {a}\t|\t\t|\tscientific name\t|\n"
            for a, b, r in nodes
        )
    )
    (ncbi / "merged.dmp").write_text("")
    foodon = raw / "foodon"
    foodon.mkdir(parents=True)
    classes = [
        '<owl:Class rdf:about="http://purl.obolibrary.org/obo/FOODON_999999"><rdfs:label>fictional root</rdfs:label></owl:Class>'
    ]
    for i in range(24):
        classes.append(
            f'<owl:Class rdf:about="http://purl.obolibrary.org/obo/FOODON_{900000 + i}"><rdfs:label>fictional food {i:02d}</rdfs:label><rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/FOODON_999999"/></owl:Class>'
        )
    (foodon / "foodon.owl").write_text(
        '<?xml version="1.0"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" xmlns:owl="http://www.w3.org/2002/07/owl#">'
        + "".join(classes)
        + "</rdf:RDF>"
    )
    return root


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", required=True, help="Empty scratch directory outside the release"
    )
    a = p.parse_args()
    root = Path(a.root)
    if root.exists() and any(root.iterdir()):
        p.error("--root must be empty to avoid overwriting data")
    generate(root)
    configdir = root / "configs"
    configdir.mkdir()
    release = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((release / "configs/nutri19.yaml").read_text())
    cfg["output"] = "outputs/smoke"
    cfg["tolerance"] = str(release / "configs/eu_tolerance.yaml")
    cfg["training"].update(
        seeds=[42, 43],
        k_folds=2,
        epochs=2,
        patience=2,
        feature_dim=8,
        hidden_dim=8,
        device="cpu",
        amp=False,
    )
    for track in ["n3", "full"]:
        cfg["track"] = track
        (configdir / f"{track}.yaml").write_text(yaml.safe_dump(cfg))
    print(configdir)
