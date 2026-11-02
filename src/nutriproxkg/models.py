"""Data models for graph construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class IngredientEntry:
    """Ingredient entity from source datasets."""

    food_id: str
    name: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompoundEntry:
    """Compound (incl. nutrient) entity from source datasets."""

    compound_key: str
    name: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngredientCompoundLink:
    """Ingredient-Compound edge record."""

    food_id: str
    compound_key: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngredientSpeciesLink:
    """Ingredient-Species edge record."""

    food_id: str
    tax_id: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeciesEntry:
    """Species entity from NCBI taxonomy."""

    tax_id: str
    name: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FoodDBData:
    """Aggregated FooDB records needed for graph construction and evaluation."""

    ingredients: Dict[str, IngredientEntry] = field(default_factory=dict)
    compounds: Dict[str, CompoundEntry] = field(default_factory=dict)
    ingredient_compound_links: List[IngredientCompoundLink] = field(
        default_factory=list
    )
    ingredient_species_links: List[IngredientSpeciesLink] = field(default_factory=list)
    nutrient_name_by_id: Dict[str, str] = field(default_factory=dict)
