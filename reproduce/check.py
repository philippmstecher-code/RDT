#!/usr/bin/env python3
"""Compare the rebuilt Source Data with the published Source Data workbooks (source_data/*.xlsx).

reproduce/source_files.csv names, per table of each workbook, the file the builders write and its sha256. A rebuilt
file is 'identical' when its sha256 matches, 'same values' when it differs only in formatting (every number within a
relative 1e-9, every label equal to the table in the workbook), and 'differs' otherwise. Tables whose builder has not
run are listed as 'not rebuilt'; tables kept in the repository that the last run of reproduce/figures.sh did not
rewrite are listed as 'stored' (Supplementary Table 3 needs the stored SAE activations, which are not deposited).
The workbooks name runs, arms and conditions in words; labels.py maps the code's short names to them.

Usage: python reproduce/check.py            (after bash reproduce/figures.sh)
Exit status 1 if any rebuilt table differs.
"""
import csv
import hashlib
import math
import re
import sys
from pathlib import Path

import openpyxl

from labels import relabel

ROOT = Path(__file__).resolve().parents[1]


def sheet_tables(ws):
    """Blocks of a panel sheet: {block title: (column names, rows)}."""
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    out, i = {}, 3
    while i < len(rows):
        if rows[i][0] is None:
            i += 1
            continue
        title, cols, j = rows[i][0], rows[i + 1], i + 2
        n = max(k for k, c in enumerate(cols) if c is not None) + 1
        body = []
        while j < len(rows) and any(c is not None for c in rows[j]):
            body.append(rows[j][:n])
            j += 1
        out[title] = ([str(c) for c in cols[:n]], body)
        i = j
    return out


def same(a, b):
    if a is None or a == "":
        return b is None or b == "" or (isinstance(b, float) and math.isnan(b))
    try:
        x = float(a)
    except ValueError:
        return str(b) == a
    if b is None or b == "":
        return math.isnan(x)
    if isinstance(b, str):
        try:
            b = float(b)
        except ValueError:
            return False
    y = float(b)
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12)


def compare(path, table):
    cols, body = table
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    head, rows = relabel([c.strip() for c in rows[0]], rows[1:])
    if head != cols or len(rows) != len(body):
        return False
    return all(len(r) == len(b) and all(same(x, y) for x, y in zip(r, b)) for r, b in zip(rows, body))


def main():
    counts = {"identical": 0, "same values": 0, "differs": 0, "stored": 0, "not rebuilt": 0}
    stamp = ROOT / "reproduce/out/.started"
    t0 = stamp.stat().st_mtime if stamp.exists() else float("inf")
    with open(ROOT / "reproduce/source_files.csv", newline="") as fh:
        files = list(csv.DictReader(fh))
    for wb_path in sorted((ROOT / "source_data").glob("Source_Data_*.xlsx")):
        wb = openpyxl.load_workbook(wb_path, read_only=False)
        status = []
        for r in (r for r in files if r["workbook"] == wb_path.name):
            sheet, block, src, digest = r["sheet"], r["table"], r["file"], r["sha256"]
            p = ROOT / src
            if not p.exists():
                s = "not rebuilt"
            elif not src.startswith("reproduce/out/") and p.stat().st_mtime < t0:
                s = "stored"
            elif digest and hashlib.sha256(p.read_bytes()).hexdigest() == digest:
                s = "identical"
            else:
                s = "same values" if compare(p, sheet_tables(wb[sheet]).get(block, ([], []))) else "differs"
            counts[s] += 1
            status.append((s, src))
        item = re.sub(r"(Fig|Table)(\d)", r"\1. \2", wb_path.stem.replace("Source_Data_", "").replace("_", " "))
        item = item.replace("Table. ", "Table ")
        worst = next((s for s in ("differs", "not rebuilt", "stored", "same values") if any(x == s for x, _ in status)),
                     "identical")
        print(f"{item:30s} {len(status):3d} table{'s' if len(status) != 1 else ' '}  {worst}")
        for s, src in status:
            if s in ("differs",):
                print(f"    {s}: {src}")
    print("\n" + ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 1 if counts["differs"] else 0


if __name__ == "__main__":
    sys.exit(main())
