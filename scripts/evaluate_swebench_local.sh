#!/usr/bin/env bash
# Run the dataset regression tests in a separate checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
source .env.swebench
TASK="${1:?task required}"
PATCH="$(realpath "${2:?agent patch path required}")"
MODE="${3:-patched}"
case "$MODE" in patched|baseline) ;; *) exit 2;; esac
BASE=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_commit"])' "workloads/swebench/$TASK/task_context.json")
EVAL="$SWE_STORAGE/evaluation/$TASK-$MODE-$(date -u +%Y%m%dT%H%M%S)"
git clone --quiet --shared "$SWE_STORAGE/work/$TASK" "$EVAL"
git -C "$EVAL" checkout -q --detach "$BASE"
if [ "$MODE" = patched ]; then git -C "$EVAL" apply "$PATCH"; fi
git -C "$EVAL" apply "$ROOT/workloads/swebench_evaluation/$TASK.test.patch"
export PYTHONPATH="$EVAL"
cd "$EVAL"
set +e
case "$TASK" in
 django) "$SWE_STORAGE/django-env/bin/python" tests/runtests.py migrations.test_operations --parallel 1 --verbosity 1 > "$EVAL.test.log" 2>&1;;
 sphinx) "$SWE_STORAGE/sphinx-env/bin/python" -m pytest -q tests/test_ext_autodoc_autoclass.py > "$EVAL.test.log" 2>&1;;
 nodebb)
   cp "$SWE_STORAGE/work/nodebb/package.json" .
   cp "$SWE_STORAGE/work/nodebb/config.json" .
   ln -s "$SWE_STORAGE/work/nodebb/node_modules" node_modules
   ./node_modules/.bin/mocha test/database.js test/database/sorted.js > "$EVAL.test.log" 2>&1;;
esac
RC=$?
set -e
echo "$RC" > "$EVAL.exit_code"
tail -25 "$EVAL.test.log"
echo "evaluation=$EVAL exit_code=$RC"
exit "$RC"
