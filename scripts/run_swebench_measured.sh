#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .env.swebench
TASK="${1:?Usage: run_swebench_measured.sh django|sphinx|nodebb}"
SWE_WORK_ROOT="${SWE_WORK_ROOT:-$SWE_STORAGE/work}"
export PYTHONPATH="$SWE_WORK_ROOT/$TASK${PYTHONPATH:+:$PYTHONPATH}"
case "$TASK" in django|sphinx) export PATH="$SWE_STORAGE/$TASK-env/bin:$PATH";; nodebb) ;; *) exit 2;; esac
OUT="$PWD/results/swebench-measured/$TASK-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"
export PROV_CCR_URL=http://127.0.0.1:3456
export PROV_CCR_TOKEN=ccr
curl -fsS "$VLLM_URL/metrics" > "$OUT/metrics-before.txt"
curl -fsS "$VLLM_URL/server_info" > "$OUT/server-info.json"
"$SWE_STORAGE/venv/bin/python" scripts/swebench_proxy.py --upstream "$VLLM_URL" --output "$OUT/http" > "$OUT/proxy.log" 2>&1 &
PROXY_PID=$!
trap 'kill "$PROXY_PID" 2>/dev/null || true' EXIT
sleep 1
kill -0 "$PROXY_PID"
TRACE="${SWE_TRACE:-1}"
TRACER_PID=""
if [ "$TRACE" = 1 ]; then
  SWE_START_PAUSED=1 "$SWE_STORAGE/venv/bin/python" scripts/swebench_local.py run --task "$TASK" --work-root "$SWE_WORK_ROOT" --output "$OUT/agent" --execute --permission-mode bypassPermissions &
  AGENT_PID=$!
  sudo -n env PYTHONPATH="$PWD/src" /usr/bin/python3 -m agent_io_tracing.tracing.bcc_tracer --root-pid "$AGENT_PID" --output "$OUT/ebpf_events.log" --include-net --perf-pages 4096 > "$OUT/bcc.out" 2> "$OUT/bcc.err" &
  TRACER_PID=$!
  for ((i=0; i<120; i++)); do
    [ -s "$OUT/ebpf_events.log" ] && break
    if ! kill -0 "$TRACER_PID" 2>/dev/null; then
      kill -KILL "$AGENT_PID"
      cat "$OUT/bcc.err" >&2
      exit 1
    fi
    sleep 1
  done
  if [ ! -s "$OUT/ebpf_events.log" ]; then
    kill -KILL "$AGENT_PID"
    sudo -n kill -INT "$TRACER_PID" || true
    exit 1
  fi
  kill -CONT "$AGENT_PID"
  set +e
  wait "$AGENT_PID"
  RC=$?
  sudo -n pkill -INT -P "$TRACER_PID" 2>/dev/null || true
  wait "$TRACER_PID"
  set -e
else
  set +e
  "$SWE_STORAGE/venv/bin/python" scripts/swebench_local.py run --task "$TASK" --work-root "$SWE_WORK_ROOT" --output "$OUT/agent" --execute --permission-mode bypassPermissions
  RC=$?
  set -e
fi
sleep 6
curl -fsS "$VLLM_URL/metrics" > "$OUT/metrics-after.txt"
echo "$RC" > "$OUT/exit_code.txt"
"$SWE_STORAGE/venv/bin/python" scripts/summarize_swebench.py "$OUT" > "$OUT/summary.stdout.json"
if [ "$TRACE" = 1 ]; then
  PYTHONPATH="$PWD/src" "$SWE_STORAGE/venv/bin/python" -m agent_io_tracing.parsing.ebpf "$OUT" > "$OUT/parse.log" 2>&1
  "$SWE_STORAGE/venv/bin/python" scripts/postprocess_swebench.py "$OUT"
fi
echo "$OUT"
exit "$RC"
