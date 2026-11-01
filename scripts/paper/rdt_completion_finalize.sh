#!/bin/zsh
# RDT completion: all analyses on the complete lane set (called by rdt_completion_driver.sh), sequentially.
# Logs in analysis_outputs/paper/RDT_completion/final_*.log.
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}; O=analysis_outputs/paper/RDT_completion
ALL=sham,gradsc,gradjoint,gradenergy,gradrand,gradrg,gradsc_l3,gradjoint_l3
echo "[finalize] $(date)"
$PY scripts/paper/rdt_completion_analysis.py --workers 3 > $O/final_part1.log 2>&1
# not part of this release
$PY scripts/paper/rdt_completion_nesting.py > $O/final_part2.log 2>&1
# not part of this release
$PY scripts/paper/rdt_completion_lineage.py --workers 3 > $O/final_part3.log 2>&1
$PY scripts/paper/rdt_completion_patterns.py routing --arms $ALL > $O/final_routing.log 2>&1
$PY scripts/paper/rdt_completion_patterns.py dose --arms sham,gradsc,gradjoint,gradenergy,gradrand,gradrg > $O/final_dose.log 2>&1
$PY scripts/paper/rdt_completion_patterns.py turnover --arms $ALL > $O/final_turnover.log 2>&1
# not part of this release
$PY scripts/paper/rdt_completion_noise.py > $O/final_part4.log 2>&1
echo "[finalize] done $(date)"
