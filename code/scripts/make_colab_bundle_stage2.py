"""Bundle for notebook 02_8 (steps 8-9 with other models) on Colab:

    uv run python scripts/make_colab_bundle_stage2.py      # -> data/colab_bundle_stage2.zip

Run it after 02_6 (it needs importance.csv, the step-9 ablation and the selected set). The bundle
keeps the repository layout, so xai_ews resolves its paths unchanged: src/xai_ews, configs, the
two notebooks, the stage-2 inputs of 02_8, and data/processed with only what 02_8 loads: the Y1
train sample (all events + negatives with u_neg < r) and the Y1 validation rows, candidate columns
only. Test and stress rows are not in the bundle.
"""

import shutil
import sys
import zipfile

import polars as pl

from xai_ews.config import feature_config
from xai_ews.data.features import candidate_features
from xai_ews.features.data import TASKS
from xai_ews.paths import (
    DATA_DIR,
    FEATURE_SELECTION_DIR,
    NOTEBOOKS_DIR,
    PANEL_DIR,
    PROJECT_ROOT,
    STAGE2_INTERIM_DIR,
    TRAIN_FULL_DIR,
)

OUT = DATA_DIR / "colab_bundle_stage2"
NOTEBOOKS = ["02_8_other_models.ipynb", "colab_stage2_other_models.ipynb"]
REPORTS = ["redundancy.csv", "importance.csv", "importance_models_auc.csv"]


def main() -> int:
    r = feature_config()["data"]["negatives_rate"]
    keys = ["loan_identifier", "m"]
    feats = candidate_features()
    labels = [c for t, pop in TASKS.items() for c in (t, pop)]
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ("src", "configs"):
        shutil.copytree(PROJECT_ROOT / d, OUT / d, ignore=shutil.ignore_patterns("__pycache__"))
    (OUT / "notebooks").mkdir()
    for f in NOTEBOOKS:
        shutil.copy(NOTEBOOKS_DIR / f, OUT / "notebooks")
    rep = OUT / FEATURE_SELECTION_DIR.relative_to(PROJECT_ROOT)
    rep.mkdir(parents=True)
    for f in REPORTS:
        shutil.copy(FEATURE_SELECTION_DIR / f, rep)
    s2 = OUT / STAGE2_INTERIM_DIR.relative_to(PROJECT_ROOT)
    s2.mkdir(parents=True)
    for t in TASKS:
        for f in (f"woe_binners_{t}.json", f"ablation_step9_{t}.csv"):
            shutil.copy(STAGE2_INTERIM_DIR / f, s2)
    # train: only the rows train_sample() can select for these tasks
    keep = pl.any_horizontal([pl.col(pop) & pl.col(t).is_not_null() & ((pl.col(t) == 1) | (pl.col("u_neg") < r))
                              for t, pop in TASKS.items()])
    for part in sorted(TRAIN_FULL_DIR.glob("vintage=*")):
        df = pl.read_parquet(part / "*.parquet", columns=[*keys, *feats, *labels, "u_neg"]).filter(keep)
        (OUT / TRAIN_FULL_DIR.relative_to(PROJECT_ROOT) / part.name).mkdir(parents=True)
        df.write_parquet(OUT / TRAIN_FULL_DIR.relative_to(PROJECT_ROOT) / part.name / "part.parquet",
                         compression="zstd")
    va_cols = [*keys, *feats, *labels, "split_main", "val_hurricane_2017"]
    for part in sorted(PANEL_DIR.glob("vintage=*")):
        df = pl.read_parquet(part / "*.parquet", columns=va_cols).filter(pl.col("split_main") == "validation")
        if df.height:
            (OUT / PANEL_DIR.relative_to(PROJECT_ROOT) / part.name).mkdir(parents=True)
            df.write_parquet(OUT / PANEL_DIR.relative_to(PROJECT_ROOT) / part.name / "part.parquet",
                             compression="zstd")
    zpath = DATA_DIR / "colab_bundle_stage2.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(OUT))
    print(f"{zpath}: {zpath.stat().st_size / 1e6:,.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
