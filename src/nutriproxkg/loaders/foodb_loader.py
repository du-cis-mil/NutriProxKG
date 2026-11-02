"""Loader for FooDB raw dumps and preprocessed tables."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ..models import (
    CompoundEntry,
    FoodDBData,
    IngredientCompoundLink,
    IngredientEntry,
    IngredientSpeciesLink,
)
from .base import LoaderBase, iter_table_rows, normalize_text, parse_float


class FooDBLoader(LoaderBase):
    """Parse FooDB files for ingredient-compound graph construction."""

    FOOD_CANDIDATES = ["Food.csv", "food.csv"]
    COMPOUND_CANDIDATES = ["Compound.csv", "compound.csv"]
    NUTRIENT_CANDIDATES = ["Nutrient.csv", "nutrient.csv"]
    CONTENT_CANDIDATES = ["Content.csv", "content.csv"]
    PREPROCESSED_EDGE_CANDIDATES = ["food_compound_content.csv"]
    DEFAULT_MICRONUTRIENTS = [
        "vitamin_c_g",
        "vitamin_a_retinol_g",
        "vitamin_b1_thiamine_g",
        "vitamin_b2_riboflavin_g",
        "vitamin_b9_folate_g",
        "vitamin_e_tocopherol_g",
        "vitamin_d_g",
        "zinc_g",
        "iron_g",
        "magnesium_g",
        "calcium_g",
        "potassium_g",
        "sodium_g",
        "phosphorus_g",
        "copper_g",
        "manganese_g",
        "selenium_g",
        "iodine_g",
    ]

    def __init__(
        self, raw_dir: str | Path, logger: Optional[logging.Logger] = None
    ) -> None:
        super().__init__(raw_dir=raw_dir, logger=logger)

    def load(
        self,
        max_food_rows: Optional[int] = None,
        max_content_rows: Optional[int] = None,
        include_preprocessed_fallback: bool = True,
    ) -> FoodDBData:
        """Load FooDB entity and edge records.

        Args:
            max_food_rows: Optional limit for number of Food rows.
            max_content_rows: Optional limit for Content rows.
            include_preprocessed_fallback: Use preprocessed name-based file if needed.
        """
        food_file = self.find_file(self.FOOD_CANDIDATES)
        compound_file = self.find_file(self.COMPOUND_CANDIDATES)
        nutrient_file = self.find_file(self.NUTRIENT_CANDIDATES)
        content_file = self.find_file(self.CONTENT_CANDIDATES)

        if not food_file:
            self.logger.warning("Food.csv not found under %s", self.raw_dir)
            return FoodDBData()

        ingredients, ingredient_species_links = self._load_foods(
            food_file, max_food_rows
        )

        compounds: Dict[str, CompoundEntry] = {}
        nutrient_name_by_id: Dict[str, str] = {}

        if compound_file:
            compounds.update(self._load_compounds(compound_file))
        else:
            self.logger.warning(
                "Compound.csv not found; Ingredient-Compound edges may be sparse"
            )

        if nutrient_file:
            nutrient_compounds, nutrient_name_by_id = self._load_nutrients(
                nutrient_file
            )
            compounds.update(nutrient_compounds)
        else:
            self.logger.warning("Nutrient.csv not found")

        ingredient_compound_links: List[IngredientCompoundLink] = []
        if content_file:
            ingredient_compound_links = self._load_content_edges(
                content_file=content_file,
                ingredients=ingredients,
                compounds=compounds,
                max_content_rows=max_content_rows,
            )
        else:
            self.logger.warning("Content.csv not found")

        if include_preprocessed_fallback and not ingredient_compound_links:
            pre_file = self.find_file(self.PREPROCESSED_EDGE_CANDIDATES)
            if pre_file:
                self.logger.info(
                    "Using preprocessed fallback edges from %s because Content edges are empty",
                    pre_file,
                )
                edges = self._load_preprocessed_edges(pre_file, ingredients, compounds)
                ingredient_compound_links.extend(edges)

        self.logger.info(
            "FooDB loaded: %d ingredients, %d compounds, %d ingredient-compound edges, %d ingredient-species links",
            len(ingredients),
            len(compounds),
            len(ingredient_compound_links),
            len(ingredient_species_links),
        )
        return FoodDBData(
            ingredients=ingredients,
            compounds=compounds,
            ingredient_compound_links=ingredient_compound_links,
            ingredient_species_links=ingredient_species_links,
            nutrient_name_by_id=nutrient_name_by_id,
        )

    def _load_foods(
        self,
        path: Path,
        max_rows: Optional[int],
    ) -> Tuple[Dict[str, IngredientEntry], List[IngredientSpeciesLink]]:
        ingredients: Dict[str, IngredientEntry] = {}
        species_links: List[IngredientSpeciesLink] = []

        for idx, row in enumerate(iter_table_rows(path)):
            if max_rows is not None and idx >= max_rows:
                break

            food_id = normalize_text(row.get("id"))
            if not food_id:
                continue
            name = normalize_text(row.get("name") or row.get("name_scientific"))
            if not name:
                continue

            attrs = {
                "FOODB.id": food_id,
                "FOODB.public_id": normalize_text(row.get("public_id")),
                "FOODB.name_scientific": normalize_text(row.get("name_scientific")),
                "FOODB.category": normalize_text(row.get("category")),
                "FOODB.food_group": normalize_text(row.get("food_group")),
                "FOODB.food_subgroup": normalize_text(row.get("food_subgroup")),
                "FOODB.food_type": normalize_text(row.get("food_type")),
                "FOODB.ncbi_taxonomy_id": normalize_text(row.get("ncbi_taxonomy_id")),
                "FOODB.source_record_id": food_id,
            }
            ingredients[food_id] = IngredientEntry(
                food_id=food_id, name=name, attrs=attrs
            )

            tax_id = normalize_text(row.get("ncbi_taxonomy_id"))
            if tax_id and tax_id.lower() not in {"0", "nan", "none"}:
                species_links.append(
                    IngredientSpeciesLink(
                        food_id=food_id,
                        tax_id=tax_id,
                        attrs={
                            "FOODB.food_id": food_id,
                            "FOODB.ncbi_taxonomy_id": tax_id,
                        },
                    )
                )
        return ingredients, species_links

    def _load_compounds(self, path: Path) -> Dict[str, CompoundEntry]:
        compounds: Dict[str, CompoundEntry] = {}
        for row in iter_table_rows(path):
            comp_id = normalize_text(row.get("id"))
            if not comp_id:
                continue
            name = normalize_text(
                row.get("name") or row.get("public_id") or f"compound_{comp_id}"
            )
            key = f"Compound:{comp_id}"
            compounds[key] = CompoundEntry(
                compound_key=key,
                name=name,
                attrs={
                    "FOODB.id": comp_id,
                    "FOODB.public_id": normalize_text(row.get("public_id")),
                    "FOODB.cas_number": normalize_text(row.get("cas_number")),
                    "FOODB.kingdom": normalize_text(row.get("kingdom")),
                    "FOODB.superklass": normalize_text(row.get("superklass")),
                    "FOODB.klass": normalize_text(row.get("klass")),
                    "FOODB.subklass": normalize_text(row.get("subklass")),
                    "FOODB.source_record_id": comp_id,
                    "FOODB.source_table": "Compound",
                },
            )
        return compounds

    def _load_nutrients(
        self, path: Path
    ) -> Tuple[Dict[str, CompoundEntry], Dict[str, str]]:
        compounds: Dict[str, CompoundEntry] = {}
        nutrient_name_by_id: Dict[str, str] = {}

        for row in iter_table_rows(path):
            nid = normalize_text(row.get("id"))
            if not nid:
                continue
            name = normalize_text(
                row.get("name") or row.get("dfc_name") or f"nutrient_{nid}"
            )
            nutrient_name_by_id[nid] = name
            key = f"Nutrient:{nid}"
            compounds[key] = CompoundEntry(
                compound_key=key,
                name=name,
                attrs={
                    "FOODB.id": nid,
                    "FOODB.public_id": normalize_text(row.get("public_id")),
                    "FOODB.dfc_id": normalize_text(row.get("dfc_id")),
                    "FOODB.dfc_name": normalize_text(row.get("dfc_name")),
                    "FOODB.source_record_id": nid,
                    "FOODB.source_table": "Nutrient",
                    "FOODB.is_nutrient": "true",
                },
            )
        return compounds, nutrient_name_by_id

    def _load_content_edges(
        self,
        content_file: Path,
        ingredients: Dict[str, IngredientEntry],
        compounds: Dict[str, CompoundEntry],
        max_content_rows: Optional[int],
    ) -> List[IngredientCompoundLink]:
        edges: List[IngredientCompoundLink] = []

        iterator = iter_table_rows(content_file)
        for idx, row in enumerate(
            tqdm(iterator, desc="Parsing FooDB Content", unit="row")
        ):
            if max_content_rows is not None and idx >= max_content_rows:
                break

            food_id = normalize_text(row.get("food_id"))
            source_type = normalize_text(row.get("source_type")).lower()
            source_id = normalize_text(row.get("source_id"))

            if not food_id or food_id not in ingredients or not source_id:
                continue

            if source_type == "compound":
                compound_key = f"Compound:{source_id}"
            elif source_type == "nutrient":
                compound_key = f"Nutrient:{source_id}"
            else:
                continue

            if compound_key not in compounds:
                continue

            edge_attrs = {
                "FOODB.content_id": normalize_text(row.get("id")),
                "FOODB.source_id": source_id,
                "FOODB.source_type": source_type,
                "FOODB.standard_content": normalize_text(row.get("standard_content")),
                "FOODB.orig_unit": normalize_text(row.get("orig_unit")),
                "FOODB.source_record_id": normalize_text(row.get("id")),
            }
            edges.append(
                IngredientCompoundLink(
                    food_id=food_id,
                    compound_key=compound_key,
                    attrs=edge_attrs,
                )
            )

        return edges

    def _load_preprocessed_edges(
        self,
        path: Path,
        ingredients: Dict[str, IngredientEntry],
        compounds: Dict[str, CompoundEntry],
    ) -> List[IngredientCompoundLink]:
        name_to_food_id = {v.name.lower(): k for k, v in ingredients.items()}
        name_to_compound_key = {v.name.lower(): k for k, v in compounds.items()}
        edges: List[IngredientCompoundLink] = []

        for idx, row in enumerate(iter_table_rows(path), start=1):
            ingredient_name = normalize_text(row.get("food_name"))
            compound_name = normalize_text(row.get("compound_name"))
            if not ingredient_name or not compound_name:
                continue

            food_id = name_to_food_id.get(ingredient_name.lower())
            if not food_id:
                food_id = f"pre_{idx}"
                ingredients[food_id] = IngredientEntry(
                    food_id=food_id,
                    name=ingredient_name,
                    attrs={
                        "FOODB.id": food_id,
                        "FOODB.source_table": "food_compound_content",
                        "FOODB.generated_from_preprocessed": "true",
                    },
                )
                name_to_food_id[ingredient_name.lower()] = food_id

            comp_key = name_to_compound_key.get(compound_name.lower())
            if not comp_key:
                comp_key = f"Preprocessed:{idx}"
                compounds[comp_key] = CompoundEntry(
                    compound_key=comp_key,
                    name=compound_name,
                    attrs={
                        "FOODB.source_table": "food_compound_content",
                        "FOODB.generated_from_preprocessed": "true",
                    },
                )
                name_to_compound_key[compound_name.lower()] = comp_key

            edges.append(
                IngredientCompoundLink(
                    food_id=food_id,
                    compound_key=comp_key,
                    attrs={
                        "FOODB.source_table": "food_compound_content",
                        "FOODB.standard_content": normalize_text(
                            row.get("standard_content")
                        ),
                        "FOODB.orig_unit": normalize_text(row.get("orig_unit")),
                    },
                )
            )
        return edges

    def extract_nutrition_table(
        self,
        max_rows: Optional[int] = None,
        include_micronutrients: bool = False,
        micronutrients: Optional[List[str]] = None,
        estimate_water: bool = True,
    ) -> List[Dict[str, float | str]]:
        """Build nutrient table per ingredient from FooDB Content + Nutrient files.

        Returns rows with keys:
            ingredient_name, carbs_g, fat_g, protein_g, (water_g,) (+ optional micronutrients)

        Args:
            estimate_water: When True (default), add a ``water_g`` column estimated
                by difference from a 100 g edible portion (FooDB has no measured
                water values). When False, ``water_g`` is omitted entirely and only
                carbs/fat/protein are required for an ingredient to be kept.
        """
        food_file = self.find_file(self.FOOD_CANDIDATES)
        nutrient_file = self.find_file(self.NUTRIENT_CANDIDATES)
        content_file = self.find_file(self.CONTENT_CANDIDATES)
        compound_file = self.find_file(self.COMPOUND_CANDIDATES)

        if not food_file or not nutrient_file or not content_file:
            self.logger.warning(
                "Nutrition table skipped: missing Food/Nutrient/Content files"
            )
            return []

        food_name_by_id: Dict[str, str] = {}
        for row in iter_table_rows(food_file):
            fid = normalize_text(row.get("id"))
            name = normalize_text(row.get("name") or row.get("name_scientific"))
            if fid and name:
                food_name_by_id[fid] = name

        nutrient_canonical: Dict[str, str] = {}
        for row in iter_table_rows(nutrient_file):
            nid = normalize_text(row.get("id"))
            name = normalize_text(row.get("name") or row.get("dfc_name"))
            canonical = self._map_nutrient_name(name)
            if nid and canonical:
                nutrient_canonical[nid] = canonical

        selected_micronutrients = set(micronutrients or self.DEFAULT_MICRONUTRIENTS)
        compound_micronutrients: Dict[str, str] = {}
        if include_micronutrients:
            if compound_file:
                compound_micronutrients = self._load_compound_micronutrient_map(
                    compound_file=compound_file,
                    selected_micronutrients=selected_micronutrients,
                )
            else:
                self.logger.warning(
                    "Compound.csv not found; micronutrient extraction disabled"
                )

        value_buckets: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for idx, row in enumerate(iter_table_rows(content_file)):
            if max_rows is not None and idx >= max_rows:
                break

            source_type = normalize_text(row.get("source_type")).lower()
            nutrient_key: Optional[str] = None

            food_id = normalize_text(row.get("food_id"))
            source_id = normalize_text(row.get("source_id"))
            if food_id not in food_name_by_id or not source_id:
                continue

            if source_type == "nutrient":
                nutrient_key = nutrient_canonical.get(source_id)
            elif include_micronutrients and source_type == "compound":
                nutrient_key = compound_micronutrients.get(source_id)

            if not nutrient_key:
                continue

            raw_val = parse_float(
                row.get("standard_content") or row.get("orig_content")
            )
            if raw_val is None:
                continue

            unit = normalize_text(
                row.get("orig_unit") or row.get("orig_unit_expression") or ""
            )
            converted = self._to_g_per_100g(raw_val, unit, nutrient_key=nutrient_key)
            if converted is None:
                continue
            value_buckets[(food_id, nutrient_key)].append(converted)

        rows: List[Dict[str, float | str]] = []
        macros = ["carbs_g", "fat_g", "protein_g"]
        required = macros + ["water_g"] if estimate_water else list(macros)
        for food_id, ingredient_name in food_name_by_id.items():
            row: Dict[str, float | str] = {"ingredient_name": ingredient_name}
            for nutrient in required:
                values = value_buckets.get((food_id, nutrient), [])
                row[nutrient] = float(median(values)) if values else np.nan

            # FooDB nutrient table frequently lacks explicit water values.
            # Fallback: estimate moisture by difference from 100 g edible portion.
            if estimate_water and not np.isfinite(row["water_g"]):
                if all(np.isfinite(row[n]) for n in macros):
                    estimated_water = (
                        100.0
                        - float(row["carbs_g"])
                        - float(row["fat_g"])
                        - float(row["protein_g"])
                    )
                    row["water_g"] = max(0.0, estimated_water)

            if include_micronutrients:
                for micro in sorted(selected_micronutrients):
                    micro_vals = value_buckets.get((food_id, micro), [])
                    row[micro] = float(median(micro_vals)) if micro_vals else np.nan

            if all(np.isfinite(row[n]) for n in required):
                rows.append(row)

        if include_micronutrients:
            for micro in sorted(selected_micronutrients):
                coverage = sum(
                    1
                    for row in rows
                    if micro in row
                    and isinstance(row[micro], (float, int))
                    and np.isfinite(row[micro])
                )
                self.logger.info("Micronutrient coverage %s: %d rows", micro, coverage)

        self.logger.info("Nutrition table rows: %d", len(rows))
        return rows

    @staticmethod
    def _map_nutrient_name(name: str) -> Optional[str]:
        text = normalize_text(name).lower()
        if not text:
            return None

        if "carbo" in text or "carb" in text:
            return "carbs_g"
        if "fat" in text or "lipid" in text:
            return "fat_g"
        if "protein" in text:
            return "protein_g"
        if "water" in text or "moisture" in text:
            return "water_g"
        return FooDBLoader._map_compound_to_micronutrient(text)

    @staticmethod
    def _map_compound_to_micronutrient(name: str) -> Optional[str]:
        text = normalize_text(name).lower()
        if not text:
            return None

        if re.search(r"\bascorbic acid\b", text):
            return "vitamin_c_g"
        if re.search(r"\bretinol\b", text):
            return "vitamin_a_retinol_g"
        if re.search(r"\bthiamine?\b", text):
            return "vitamin_b1_thiamine_g"
        if re.search(r"\briboflavin(e)?\b", text):
            return "vitamin_b2_riboflavin_g"
        if (
            "folic acid" in text
            or "tetrahydrofolic" in text
            or "methyltetrahydrofolic" in text
        ):
            return "vitamin_b9_folate_g"
        if re.search(r"\btocopherol\b", text):
            return "vitamin_e_tocopherol_g"
        if re.search(r"\bvitamin d(2|3)?\b", text) or "hydroxyvitamin d" in text:
            return "vitamin_d_g"
        if re.search(r"\bzinc\b", text):
            return "zinc_g"
        if re.search(r"\b(iron|ferrous|ferric)\b", text):
            return "iron_g"
        if re.search(r"\bmagnesium\b", text):
            return "magnesium_g"
        if re.search(r"\bcalcium\b", text):
            return "calcium_g"
        if re.search(r"\bpotassium\b", text):
            return "potassium_g"
        if re.search(r"\bsodium\b", text):
            return "sodium_g"
        if re.search(r"\bphosphorus\b", text):
            return "phosphorus_g"
        if re.search(r"\bcopper\b", text):
            return "copper_g"
        if re.search(r"\bmanganese\b", text):
            return "manganese_g"
        if re.search(r"\bselenium\b", text):
            return "selenium_g"
        if re.search(r"\biodine\b", text) or re.search(r"\biodide\b", text):
            return "iodine_g"
        return None

    def _load_compound_micronutrient_map(
        self,
        compound_file: Path,
        selected_micronutrients: set[str],
    ) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for row in iter_table_rows(compound_file):
            cid = normalize_text(row.get("id"))
            if not cid:
                continue
            key = self._map_compound_to_micronutrient(normalize_text(row.get("name")))
            if key and key in selected_micronutrients:
                mapping[cid] = key
        return mapping

    @staticmethod
    def _to_g_per_100g(
        value: float, unit: str, nutrient_key: Optional[str] = None
    ) -> Optional[float]:
        u = unit.lower().replace(" ", "")
        if not u:
            # Heuristic: macro nutrients in this dump are often mg/100g when unit is missing.
            return value / 1000.0 if value > 100.0 else value
        if "iu/100g" in u or u == "iu":
            if nutrient_key == "vitamin_d_g":
                return value * 0.025 / 1_000_000.0
            if nutrient_key == "vitamin_a_retinol_g":
                return value * 0.3 / 1_000_000.0
            return None
        if "mg/100g" in u:
            return value / 1000.0
        if "ug/100g" in u or "µg/100g" in u:
            return value / 1_000_000.0
        if "kg/100g" in u:
            return value * 1000.0
        if "g/100g" in u:
            return value
        if u == "g":
            return value
        if u == "mg":
            return value / 1000.0
        if u in {"ug", "µg"}:
            return value / 1_000_000.0
        return None
