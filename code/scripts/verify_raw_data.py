"""Verify raw Fannie Mae archives against the per-quarter manifests.

Checks file size and SHA-256 of every zip in data/raw/<quarter>/ and that the archive holds the
expected CSV member with the expected uncompressed size. Row counts are taken from the manifest
(counting them again means decompressing ~150 GB); pass --count-rows to recount anyway.

--write-missing first creates data/manifests/<quarter>_manifest.csv for every archive that has no
manifest yet: one CSV member with 113 fields, full read (zipfile checks the CRC), row count, sha256.

    uv run python scripts/verify_raw_data.py [--write-missing] [--count-rows] [--quarters 2016Q1 2018Q3]
"""

import argparse
import csv
import hashlib
import sys
import zipfile
from datetime import date
from pathlib import Path

from xai_ews.data.schema import DELIMITER, N_FIELDS
from xai_ews.paths import MANIFEST_DIR, RAW_DIR, REPORTS_DIR

CHUNK = 8 * 1024 * 1024
SOURCE = "Fannie Mae Data Dynamics, Single-Family Loan Performance Primary Dataset"
MANIFEST_FIELDS = ["manifest_date", "relative_path", "source", "bytes", "sha256", "zip_member",
                   "uncompressed_bytes", "rows", "status"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def count_rows(zf: zipfile.ZipFile, member: str) -> int:
    n = 0
    with zf.open(member) as f:
        while block := f.read(CHUNK):
            n += block.count(b"\n")
    return n


def load_manifests(quarters: list[str] | None) -> list[dict]:
    """One entry per quarter from all manifests (per-quarter files and data_manifest.csv)."""
    by_quarter = {}
    for path in sorted(MANIFEST_DIR.glob("*manifest.csv")):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                quarter = Path(r["relative_path"]).stem
                if not quarters or quarter in quarters:
                    by_quarter[quarter] = {"quarter": quarter, **r}
    return [by_quarter[q] for q in sorted(by_quarter)]


def write_manifest(zip_path: Path) -> Path:
    quarter = zip_path.stem
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        if len(members) != 1:
            raise ValueError(f"{zip_path}: expected one member, got {[m.filename for m in members]}")
        member = members[0]
        with zf.open(member) as f:
            n_fields = f.readline().count(DELIMITER.encode()) + 1
        if n_fields != N_FIELDS:
            raise ValueError(f"{zip_path}: {n_fields} fields in the first row, expected {N_FIELDS}")
        rows = count_rows(zf, member.filename)

    entry = {
        "manifest_date": date.today().isoformat(),
        "relative_path": f"fannie_mae/data/raw/{quarter}/{quarter}.zip",
        "source": SOURCE,
        "bytes": zip_path.stat().st_size,
        "sha256": sha256(zip_path),
        "zip_member": member.filename,
        "uncompressed_bytes": member.file_size,
        "rows": rows,
        "status": "pass",
    }
    out = MANIFEST_DIR / f"{quarter}_manifest.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerow(entry)
    return out


def verify(entry: dict, recount: bool) -> dict:
    quarter = entry["quarter"]
    zip_path = RAW_DIR / quarter / f"{quarter}.zip"
    result = {"quarter": quarter, "zip": str(zip_path.relative_to(RAW_DIR.parent.parent))}
    problems = []

    if not zip_path.exists():
        return {**result, "status": "missing", "problems": "zip not found"}

    size = zip_path.stat().st_size
    if size != int(entry["bytes"]):
        problems.append(f"size {size} != {entry['bytes']}")
    digest = sha256(zip_path)
    if digest != entry["sha256"]:
        problems.append("sha256 mismatch")

    with zipfile.ZipFile(zip_path) as zf:
        info = {i.filename: i for i in zf.infolist()}
        member = entry["zip_member"]
        if member not in info:
            problems.append(f"member {member} not in archive ({list(info)})")
        else:
            if info[member].file_size != int(entry["uncompressed_bytes"]):
                problems.append("uncompressed size mismatch")
            if recount:
                rows = count_rows(zf, member)
                result["rows_recounted"] = rows
                if rows != int(entry["rows"]):
                    problems.append(f"rows {rows} != {entry['rows']}")

    return {
        **result,
        "bytes": size,
        "uncompressed_bytes": int(entry["uncompressed_bytes"]),
        "rows_manifest": int(entry["rows"]),
        "status": "fail" if problems else "pass",
        "problems": "; ".join(problems),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-missing", action="store_true")
    parser.add_argument("--count-rows", action="store_true")
    parser.add_argument("--quarters", nargs="*")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "data_audit" / "raw_integrity.csv")
    args = parser.parse_args()

    if args.write_missing:
        known = {e["quarter"] for e in load_manifests(None)}
        for zip_path in sorted(RAW_DIR.glob("*/*.zip")):
            if zip_path.stem in known or (args.quarters and zip_path.stem not in args.quarters):
                continue
            print(f"{zip_path.stem}: writing manifest -> {write_manifest(zip_path).name}", flush=True)

    entries = load_manifests(args.quarters)
    if not entries:
        print("no manifests found", file=sys.stderr)
        return 1

    results = []
    for entry in entries:
        r = verify(entry, args.count_rows)
        print(f"{r['quarter']}: {r['status']} {r.get('problems', '')}", flush=True)
        results.append(r)

    fields = list(dict.fromkeys(k for r in results for k in r))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    failed = [r for r in results if r["status"] != "pass"]
    total_rows = sum(r.get("rows_manifest", 0) for r in results)
    print(f"{len(results) - len(failed)}/{len(results)} pass, {total_rows:,} rows total -> {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
