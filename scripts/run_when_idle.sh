#!/usr/bin/env bash
# Wait until the GPUs have had no processes for IDLE_MIN minutes in a row and the CPU load
# is below LOAD_MAX, then execute the whole notebook in place (outputs saved into it).
# Usage: scripts/run_when_idle.sh        (IDLE_MIN=10 LOAD_MAX=8 by default)
set -u
cd "$(dirname "$0")/.."
IDLE_MIN=${IDLE_MIN:-10}
LOAD_MAX=${LOAD_MAX:-8}
LOG=outputs/run.log

echo "$(date '+%F %T') waiting: GPUs idle for ${IDLE_MIN} min and load < ${LOAD_MAX}" >> "$LOG"
idle=0
while [ "$idle" -lt "$IDLE_MIN" ]; do
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  load=$(cut -d' ' -f1 /proc/loadavg)
  if [ "$procs" -eq 0 ] && awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN { exit !(l < m) }'; then
    idle=$((idle + 1))
  else
    idle=0
  fi
  sleep 60
done

echo "$(date '+%F %T') starting notebook" >> "$LOG"
HF_HUB_OFFLINE=1 .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=rag_assignment --ExecutePreprocessor.timeout=-1 \
  rag_reranking.ipynb >> "$LOG" 2>&1
status=$?
echo "$(date '+%F %T') finished, exit code $status" >> "$LOG"
exit $status
