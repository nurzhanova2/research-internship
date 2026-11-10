"""Copy the results of notebooks/colab_stage2_other_models.ipynb back into the project:

    uv run python scripts/import_colab_stage2.py ~/Downloads/results

the executed 02_8 notebook -> notebooks/, other_models_*.csv -> reports/feature_selection/,
step8_* / step9_* caches -> data/interim/stage2/.
"""

import shutil
import sys
from pathlib import Path

from xai_ews.paths import FEATURE_SELECTION_DIR, NOTEBOOKS_DIR, STAGE2_INTERIM_DIR


def main(src: Path) -> int:
    moved = []
    for f in sorted(src.iterdir()):
        if f.name == "02_8_other_models.ipynb":
            dst = NOTEBOOKS_DIR
        elif f.name.startswith("other_models_") and f.suffix == ".csv":
            dst = FEATURE_SELECTION_DIR
        elif f.name.startswith(("step8_", "step9_")) and f.suffix == ".csv":
            dst = STAGE2_INTERIM_DIR
        else:
            continue
        shutil.copy(f, dst / f.name)
        moved.append(f"{f.name} -> {dst.name}/")
    print("\n".join(moved) or "nothing to import")
    return 0 if moved else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).expanduser()))
