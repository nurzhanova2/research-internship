"""Raw logits of the XGBoost / LightGBM comparators on a split, in a process of their own:

    uv run python scripts/score_gbdt.py test          # called by 03_6 (validation for its dry run)

XGBoost and LightGBM bring their own OpenMP runtime; in one process with PyTorch (the MLP M2) the
second one to start hangs. So 03_6 runs this script first and joins its output by (loan, month).
Output: data/processed/predictions/gbdt_<split>.parquet with loan_identifier, m, z_<model>; every
row of the Y1 population is kept (no disaster rule), as in the 03_6 frame.
"""

import sys

from xai_ews.config import feature_config
from xai_ews.models import registry
from xai_ews.paths import PREDICTIONS_DIR

MODELS = ["xgb_y1", "lgb_y1"]


def main(split: str) -> int:
    sel = feature_config()["selected"]
    d = registry.score_by_month(split, "y1", MODELS, sel["features"], sel["categorical"],
                                keep=["loan_identifier", "m"], calibrate=False, exclusion="all")
    d.write_parquet(PREDICTIONS_DIR / f"gbdt_{split}.parquet")
    print(split, d.height, "rows scored by", ", ".join(MODELS), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
