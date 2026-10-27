#!/usr/bin/env bash
# Thin wrapper so the semifinal chain can be started with nohup.
#
#   bash run_sf.sh plan      # print paths/commands, run nothing
#   bash run_sf.sh prep      # find + unzip + verify the dataset
#   bash run_sf.sh smoke     # gates; the last two need a free GPU
#   bash run_sf.sh train     # ~2 h, refuses to start without a fresh smoke
#   bash run_sf.sh infer     # 8-view + 16-view CSV, each verified then zipped
#   bash run_sf.sh all       # the three stages above, in order
#
#   nohup bash run_sf.sh all > sf_all.log 2>&1 &
#
# All the logic and every assertion live in sf_run.py / sf_gate.py (Python, so it
# is testable off the GPU box); this file only picks the interpreter and the stage.
set -euo pipefail
cd "$(dirname "$0")"

# Find a Python that has torch + open_clip.  On AutoDL the conda interpreter is
# NOT on the non-interactive PATH (`ssh host 'python -V'` says "not found"),
# which is exactly how a 4-hour run dies at the first line.
#
# The probe uses `importlib.util.find_spec`, not `import`: importing torch costs
# 60-90 s on a throttled (no-GPU, 0.5-core) container, and a 2-minute `plan` is a
# dry run nobody runs.
PROBE='import importlib.util as u, sys; sys.exit(0 if u.find_spec("torch") and u.find_spec("open_clip") else 1)'
PY="${PY:-}"
if [ -z "$PY" ]; then
  for cand in python /root/miniconda3/bin/python python3 /usr/bin/python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      if "$cand" -c "$PROBE" >/dev/null 2>&1; then PY="$cand"; break; fi
      [ -z "${PY_FALLBACK:-}" ] && PY_FALLBACK="$cand"
    fi
  done
  if [ -z "$PY" ]; then
    echo "FATAL: no interpreter with torch + open_clip. Try:" >&2
    echo "  export PATH=/root/miniconda3/bin:\$PATH" >&2
    echo "  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 'open_clip_torch==3.3.0'" >&2
    [ -n "${PY_FALLBACK:-}" ] && echo "  (a python exists at $PY_FALLBACK but it is missing packages)" >&2
    exit 1
  fi
fi
echo "[run_sf] interpreter: $PY"

STAGE="${1:-all}"
case "$STAGE" in
  plan|prep|smoke|train|infer|all) shift || true ;;
  *) echo "usage: bash run_sf.sh {plan|prep|smoke|train|infer|all} [--root DIR]" >&2
     exit 2 ;;
esac

exec "$PY" -u sf_run.py "$STAGE" "$@"
