#!/bin/bash
#
# watch_done.sh — poll a detached remote trace run and notify (bell + line)
# when it finishes.  Run LOCALLY in a spare terminal tab after launching a
# nohup'd trace_script_bcc_*.sh; it only watches, never touches the run.
#
#   bash scripts/watch_done.sh genomas     # watches ~/genomas_run.log
#   bash scripts/watch_done.sh scilink     # watches ~/scilink_run.log
#
# Prints a live progress line each poll so it never looks dead; rings the
# terminal bell and exits by itself when the run finishes (or crashes).
# Needs SSH_USER / CLIENT_NODE (source cloudlab_env.sh first).

set -uo pipefail

WHICH="${1:-genomas}"
case "$WHICH" in
    genomas) LOG='~/genomas_run.log'; PROC="trace_script_bcc_genomas"; DONE="matrix run complete" ;;
    scilink) LOG='~/scilink_run.log'; PROC="trace_script_bcc_scilink"; DONE="All done. Results in" ;;
    *) echo "usage: watch_done.sh {genomas|scilink}" >&2; exit 2 ;;
esac

: "${SSH_USER:?source cloudlab_env.sh first}"
: "${CLIENT_NODE:?source cloudlab_env.sh first}"
INTERVAL="${WATCH_INTERVAL:-30}"
START=$(date +%s)

echo "[watch_done] $WHICH on $CLIENT_NODE — poll ${INTERVAL}s (Ctrl-C 只停止盯着看，不影响任务)"
SEEN_ALIVE=0
while true; do
    # One SSH round-trip per poll: alive? done? + last meaningful progress line.
    STATUS=$(ssh "$SSH_USER@$CLIENT_NODE" "
        a=no; pgrep -qf '$PROC' && a=yes
        d=no; grep -q '$DONE' $LOG 2>/dev/null && d=yes
        l=\$(grep -aE 'Cell [0-9]+/|Processing:|run complete|Exit code' $LOG 2>/dev/null | tail -1)
        printf '%s|%s|%s' \"\$a\" \"\$d\" \"\$l\"
    " 2>/dev/null)
    ALIVE="${STATUS%%|*}"; rest="${STATUS#*|}"; DFLAG="${rest%%|*}"; LAST="${rest#*|}"
    [ "$ALIVE" = "yes" ] && SEEN_ALIVE=1
    ELAPSED=$(( ($(date +%s) - START) / 60 ))

    if [ "$DFLAG" = "yes" ] && [ "$ALIVE" != "yes" ]; then
        printf '\a'
        echo ""
        echo "🔔 [$WHICH] 跑完了  $(date '+%H:%M:%S')  (盯了 ${ELAPSED} 分钟)"
        ssh "$SSH_USER@$CLIENT_NODE" "tail -n 5 $LOG" 2>/dev/null
        exit 0
    fi
    if [ "$SEEN_ALIVE" = "1" ] && [ "$ALIVE" != "yes" ] && [ "$DFLAG" != "yes" ]; then
        printf '\a'
        echo ""
        echo "⚠️  [$WHICH] 进程没了但没有完成标记 —— 可能崩了。看日志:"
        echo "    ssh \"\$SSH_USER@\$CLIENT_NODE\" \"tail -n 30 $LOG\""
        exit 1
    fi

    # Live progress line (overwrite in place with \r).
    printf '\r  [%s] %sm | %s\033[K' "$(date '+%H:%M:%S')" "$ELAPSED" "${LAST:-启动中…}"
    sleep "$INTERVAL"
done
