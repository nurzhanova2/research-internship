"""Column schema of the Fannie Mae Single-Family Loan Performance files.

Raw quarterly files are pipe-delimited, have no header and 113 columns; the order is given by
the field dictionary. The leakage register assigns every field a usage decision for modelling.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from xai_ews.paths import DEFAULT_DICTIONARY, LEAKAGE_REGISTER

N_FIELDS = 113
DELIMITER = "|"


@dataclass(frozen=True)
class Field:
    position: int
    name: str
    official_name: str
    type: str
    applicable: bool  # populated in the SF Loan Performance dataset (√) vs CRT-only (NA)


def load_fields(path: Path = DEFAULT_DICTIONARY) -> list[Field]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fields = [
        Field(
            position=int(r["field_position"]),
            name=r["field_name"],
            official_name=r["official_field_name"],
            type=r["type"],
            applicable=r["sf_loan_performance_applicability"].strip() == "√",
        )
        for r in rows
    ]
    if [f.position for f in fields] != list(range(1, N_FIELDS + 1)):
        raise ValueError(f"{path}: expected positions 1..{N_FIELDS}")
    return fields


def column_names(path: Path = DEFAULT_DICTIONARY) -> list[str]:
    return [f.name for f in load_fields(path)]


def load_leakage_register(path: Path = LEAKAGE_REGISTER) -> dict[str, str]:
    """Map field name -> v01 usage decision (candidate_feature, label_only_or_exclude, ...)."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["field_name"]: r["v01_usage_decision"] for r in csv.DictReader(f)}
