#!/usr/bin/env bash
# Rebuild every figure and table of the paper and the Supplementary Information.
# Output: reproduce/out/<item>.{png,pdf,tif} and reproduce/out/source_data/<item>_<panel>.csv
# Items marked per-run also read the per-run data deposit (README.md, "Data"): set RDT_DATA to the unpacked folder.
# Without it they are skipped and the others are built from this repository alone.
# Usage: RDT_DATA=/path/to/rdt_per_run_data bash reproduce/figures.sh [item ...]   (e.g. fig5 ed7)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
OUTDIR="$ROOT/reproduce/out"
RDT_DATA="${RDT_DATA:-/data/rdt}"
export RDT_DATA
mkdir -p "$OUTDIR"
touch "$OUTDIR/.started"  # reproduce/check.py tells rebuilt tables from stored ones by this time stamp
F=scripts/paper/figures

ITEMS=(  # key | item | builder | per-run data needed
  "fig1|Fig. 1|$F/fig1.py|1"
  "fig2|Fig. 2|$F/fig2.py|1"
  "fig3|Fig. 3|$F/fig3.py|1"
  "fig4|Fig. 4|$F/fig4.py|0"
  "fig5|Fig. 5|$F/interventions/fig5.py|0"
  "fig6|Fig. 6|$F/interventions/fig6.py|1"
  "ed1|Extended Data Fig. 1|$F/extended/ed1.py|1"
  "ed2|Extended Data Fig. 2|$F/interventions/ed2.py|0"
  "ed3|Extended Data Fig. 3|$F/extended/ed3.py|0"
  "ed4|Extended Data Fig. 4|$F/routing/ed4.py|0"
  "ed5|Extended Data Fig. 5|$F/ed5.py|0"
  "ed6|Extended Data Fig. 6|$F/interventions/ed6.py|0"
  "ed7|Extended Data Fig. 7|$F/interventions/ed7.py|1"
  "ed8|Extended Data Fig. 8|$F/interventions/ed8.py|1"
  "ed9|Extended Data Fig. 9|$F/interventions/ed9.py|1"
  "edtable1|Extended Data Table 1|scripts/paper/edtable1_runs_all.py|1"
  "supp1|Supplementary Fig. 1|$F/supp1.py|0"
  "supp2|Supplementary Fig. 2|$F/supp2.py|0"
  "supp3|Supplementary Fig. 3|$F/extended/supp3.py|1"
  "supp4|Supplementary Fig. 4|$F/extended/supp4.py|1"
  "supp5|Supplementary Fig. 5|$F/supp5.py|0"
  "supp6|Supplementary Fig. 6|$F/supp6.py|0"
  "supp7|Supplementary Fig. 7|$F/interventions/supp7.py|0"
  "supp_table1|Supplementary Table 1|scripts/paper/supp_table1_ordering.py|0"
  "supp_tables2_3|Supplementary Tables 2 and 3|scripts/paper/si_source_files.py|1"
)

HAVE_DATA=0
if [ -d "$RDT_DATA/core/experiments" ] && [ -d "$RDT_DATA/extended/experiments" ]; then
  HAVE_DATA=1
  # runs the scripts read from data/experiments inside the repository
  if [ -d "$RDT_DATA/repository/data/experiments" ]; then
    mkdir -p "$ROOT/data/experiments"
    for d in "$RDT_DATA"/repository/data/experiments/*; do
      [ -e "$ROOT/data/experiments/$(basename "$d")" ] || ln -s "$d" "$ROOT/data/experiments/"
    done
  fi
  echo "Per-run data: $RDT_DATA"
else
  echo "Per-run data not found under $RDT_DATA: items marked per-run are skipped (see README.md, \"Data\")."
fi
echo

built=0; skipped=0; failed=0
for entry in "${ITEMS[@]}"; do
  IFS="|" read -r key item builder perrun <<<"$entry"
  if [ $# -gt 0 ] && [[ " $* " != *" $key "* ]]; then continue; fi
  if [ "$perrun" = 1 ] && [ "$HAVE_DATA" = 0 ]; then
    printf "%-30s skipped (per-run data)\n" "$item"; skipped=$((skipped + 1)); continue
  fi
  t0=$SECONDS
  if (cd "$ROOT/$(dirname "$builder")" && ES_FIG_OUT="$OUTDIR" MPLBACKEND=Agg "$PY" "$ROOT/$builder") \
      >"$OUTDIR/$key.log" 2>&1; then
    printf "%-30s built (%ds)\n" "$item" $((SECONDS - t0)); built=$((built + 1))
  else
    printf "%-30s FAILED, see reproduce/out/%s.log\n" "$item" "$key"; failed=$((failed + 1))
  fi
done

echo
echo "$built built, $skipped skipped, $failed failed; outputs in reproduce/out/"
[ "$failed" -eq 0 ]
