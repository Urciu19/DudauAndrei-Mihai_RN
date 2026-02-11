from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPECIES_INFO_PATH = PROJECT_ROOT / "config" / "species_info.json"


def _load_species_info() -> Dict[str, Any]:
    """Încarcă species_info.json de pe disc la runtime (mereu actual)."""
    if not SPECIES_INFO_PATH.exists():
        return {}
    with open(SPECIES_INFO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def enrich_prediction(prediction: dict) -> dict:
    """
    Adaugă numele comun, numele latin și descrierea la rezultatul inferenței.
    Reîncarcă JSON-ul la fiecare apel ca să reflecte imediat modificările.
    """
    species_info = _load_species_info()
    label = prediction["label"]

    info = species_info.get(label, {
        "common_name": label,
        "latin_name": "N/A",
        "description": "Descriere indisponibilă."
    })

    return {
        **prediction,
        "common_name": info.get("common_name", label),
        "latin_name": info.get("latin_name", "N/A"),
        "description": info.get("description", "Descriere indisponibilă."),
    }