"""Freezing the test protocol (reports/modeling/test_protocol.md, section 12).

    frozen_files   configs, disaster flags, validation halves, LGD table and every file of every
                   stage-3 model (CatBoost / XGBoost / LightGBM folds, MLP seeds, encoders, meta, B0 pickles)
    hash_table     markdown table "| `path` | sha256 |" for section 12
    check_frozen   called by the first cell of 03_6: the protocol must say ЗАМОРОЖЕН and every
                   listed file must still have its sha256; otherwise it raises
"""

import hashlib
import re

from xai_ews.paths import (
    CONFIGS_DIR,
    DATA_AUDIT_DIR,
    DISASTER_FLAGS,
    MODELING_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    VALIDATION_HALVES,
)

PROTOCOL = MODELING_DIR / "test_protocol.md"
FROZEN_MARK = "ЗАМОРОЖЕН"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen_files(models: list[str]) -> list:
    files = sorted(CONFIGS_DIR.glob("*.yaml")) + [DISASTER_FLAGS, VALIDATION_HALVES,
                                                  DATA_AUDIT_DIR / "lgd_variants.csv"]
    for m in models:
        p = MODELS_DIR / m
        files += sorted(f for f in p.iterdir() if f.suffix in (".cbm", ".model", ".pt", ".pkl", ".json")) \
            if p.is_dir() else [p.with_suffix(".pkl")]          # B0 and single stress models: one pickle
    return files


def hash_table(models: list[str]) -> str:
    rows = ["| Файл | sha256 |", "| --- | --- |"]
    rows += [f"| `{f.relative_to(PROJECT_ROOT)}` | `{sha256(f)}` |" for f in frozen_files(models)]
    return "\n".join(rows)


def check_frozen() -> str:
    text = PROTOCOL.read_text(encoding="utf-8")
    status = next((ln for ln in text.splitlines() if ln.startswith("**Статус:")), "")
    if FROZEN_MARK not in status:
        raise RuntimeError(f"test protocol is not frozen: {status!r}")
    pairs = re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", text)
    if not pairs:
        raise RuntimeError("no sha256 table in the protocol")
    bad = [p for p, h in pairs if not (PROJECT_ROOT / p).exists() or sha256(PROJECT_ROOT / p) != h]
    if bad:
        raise RuntimeError(f"files differ from the frozen protocol: {bad}")
    return f"protocol frozen, {len(pairs)} files verified: {status}"
