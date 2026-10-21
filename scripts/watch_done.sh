#!/usr/bin/env bash
# Watch the remote log and PID printed by launch_run.sh.
# Usage: bash scripts/watch_done.sh genomas logs/<run>.log <pid>

set -uo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: bash scripts/watch_done.sh <genomas|scilink> <remote_log> <pid>" >&2
    exit 2
fi
WORKFLOW="$1"
REMOTE_LOG="$2"
RUN_PID="$3"
case "$WORKFLOW" in
    genomas|scilink) ;;
    *) echo "Unknown workflow: $WORKFLOW" >&2; exit 2 ;;
esac
case "$RUN_PID" in
    ''|*[!0-9]*|0) echo "PID must be a positive integer." >&2; exit 2 ;;
esac
INTERVAL="${WATCH_INTERVAL:-30}"
case "$INTERVAL" in
    ''|*[!0-9]*|0) echo "WATCH_INTERVAL must be a positive integer." >&2; exit 2 ;;
esac
: "${SSH_USER:?source cloudlab_env.sh first}"
: "${CLIENT_NODE:?source cloudlab_env.sh first}"
START=$(date +%s)

remote_read() {
    local remote_args
    printf -v remote_args '%q ' "$REMOTE_LOG" "$RUN_PID" "$1"
    ssh "$SSH_USER@$CLIENT_NODE" "bash -s -- $remote_args" <<'REMOTE'
set -uo pipefail
log="$1"
pid="$2"
case "$log" in
    '~/'*) log="$HOME/${log#\~/}" ;;
    /*) ;;
    *) log="$HOME/$log" ;;
esac
if [ ! -f "$log" ]; then
    echo "Log not found: $log" >&2
    exit 2
fi
if [ "$3" = tail ]; then
    tail -n 5 -- "$log"
    exit
fi
alive=no
state=$(ps -p "$pid" -o stat= 2>/dev/null || true)
state="${state#"${state%%[![:space:]]*}"}"
case "$state" in
    ''|Z*|X*) ;;
    *) alive=yes ;;
esac
done_flag=no
if grep -q '^Results in:' "$log"; then done_flag=yes; fi
last=$(grep -aE 'Cell [0-9]+/|Processing:|run complete|Exit code|Results in:' "$log" | tail -1 || true)
printf '%s|%s|%s' "$alive" "$done_flag" "$last"
REMOTE
}

printf '[watch_done] %s PID=%s LOG=%s\n' "$WORKFLOW" "$RUN_PID" "$REMOTE_LOG"
while true; do
    if ! STATUS=$(remote_read status); then
        echo "Could not read the remote run status." >&2
        exit 2
    fi
    ALIVE="${STATUS%%|*}"
    REST="${STATUS#*|}"
    DONE="${REST%%|*}"
    LAST="${REST#*|}"
    ELAPSED=$(( ($(date +%s) - START) / 60 ))
    if [ "$ALIVE" = no ]; then
        printf '\n\a'
        if [ "$DONE" = yes ]; then
            printf '[%s] 已结束，监看 %s 分钟。\n' "$WORKFLOW" "$ELAPSED"
            remote_read tail || exit 2
            exit 0
        fi
        printf '[%s] 进程已退出，日志缺少 Results in: 完成标记。\n' "$WORKFLOW" >&2
        remote_read tail || exit 2
        exit 1
    fi
    printf '\r  [%s] %sm | %s\033[K' "$(date '+%H:%M:%S')" "$ELAPSED" "${LAST:-等待日志}"
    sleep "$INTERVAL"
done
