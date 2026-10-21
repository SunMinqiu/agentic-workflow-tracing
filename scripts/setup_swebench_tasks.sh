#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .env.swebench
[ -x "$SWE_STORAGE/django-env/bin/python" ] || uv venv --python 3.9 "$SWE_STORAGE/django-env"
uv pip install --python "$SWE_STORAGE/django-env/bin/python" -e "$SWE_STORAGE/work/django" 'asgiref<4' pytz sqlparse PyYAML pillow tblib
[ -x "$SWE_STORAGE/sphinx-env/bin/python" ] || uv venv --python 3.9 "$SWE_STORAGE/sphinx-env"
uv pip install --python "$SWE_STORAGE/sphinx-env/bin/python" -e "$SWE_STORAGE/work/sphinx" 'pytest<8' 'Jinja2<3.1' 'MarkupSafe<2.1' 'docutils<0.17' 'sphinxcontrib-applehelp==1.0.2' 'sphinxcontrib-devhelp==1.0.2' 'sphinxcontrib-htmlhelp==1.0.3' 'sphinxcontrib-qthelp==1.0.3' 'sphinxcontrib-serializinghtml==1.1.4' 'alabaster<0.8' 'babel<3' html5lib pytest-cov 'setuptools<70'
cd "$SWE_STORAGE/work/nodebb"
[ -f package.json ] || cp install/package.json package.json
npm install --no-audit --no-fund
if ! command -v redis-server >/dev/null; then sudo dnf install -y redis; fi
if ! redis-cli -p 16379 ping >/dev/null 2>&1; then
  redis-server --port 16379 --bind 127.0.0.1 --save "" --appendonly no --daemonize yes --pidfile /tmp/swebench-redis.pid --logfile /tmp/swebench-redis.log
fi
python - <<'PY'
import json
from pathlib import Path
p = Path('config.json')
if not p.exists():
    p.write_text(json.dumps({'url':'http://127.0.0.1:4567','secret':'swebench-local-test-only',
        'database':'redis','redis':{'host':'127.0.0.1','port':16379,'database':0},
        'test_database':{'host':'127.0.0.1','port':16379,'database':1}}, indent=2))
PY
