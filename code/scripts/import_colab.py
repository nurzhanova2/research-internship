"""Bring the Colab (GPU) CatBoost results into the project:

    uv run python scripts/import_colab.py <unpacked models folder from Google Drive>

Copies every study of the Colab optuna.db that the local journal does not have (the local one
keeps the MLP studies) and every model folder (fold_*.cbm, encoder.pkl, meta.json, oof.parquet)
into data/processed/models/. Existing local folders are not overwritten.
"""

import shutil
import sys
from pathlib import Path

import optuna

from xai_ews.paths import MODELS_DIR, OPTUNA_DB


def main(src: str) -> int:
    src = Path(src)
    remote = f"sqlite:///{src / 'optuna.db'}"
    local = f"sqlite:///{OPTUNA_DB}"
    have = {s.study_name for s in optuna.get_all_study_summaries(local)}
    for s in optuna.get_all_study_summaries(remote):
        if s.study_name in have:
            print(f"study {s.study_name}: already local, skipped")
            continue
        optuna.copy_study(from_study_name=s.study_name, from_storage=remote, to_storage=local)
        print(f"study {s.study_name}: {s.n_trials} trials copied")
    for d in sorted(p for p in src.iterdir() if p.is_dir() and (p / "meta.json").exists()):
        dst = MODELS_DIR / d.name
        if dst.exists():
            print(f"model {d.name}: already local, skipped")
            continue
        shutil.copytree(d, dst)
        print(f"model {d.name}: {len(list(d.glob('fold_*.cbm')))} folds copied")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
