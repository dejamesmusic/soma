#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
mkdir -p run/logs

if python3 -c "import torch" >/dev/null 2>&1; then
    py=python3
else
    if [ ! -x .cloud-venv/bin/python ]; then
        python3 -m venv .cloud-venv
    fi
    py=.cloud-venv/bin/python
fi

"$py" -m pip install -q --upgrade pip
"$py" -m pip install -q -r requirements.txt
"$py" -m pip install -q -r cloud/requirements-cloud.txt

if [ -f run/curriculum.pid ] \
        && kill -0 "$(cat run/curriculum.pid)" 2>/dev/null; then
    echo "curriculum already running as pid $(cat run/curriculum.pid)"
    exit 0
fi

nohup "$py" cloud/run_cloud_curriculum.py \
    --root "$root" --workdir run --data data \
    > run/logs/supervisor.out 2>&1 &
echo "$!" > run/curriculum.pid
echo "started soma curriculum as pid $!"
echo "$py" > run/python_command
echo "stats:  $py cloud/monitor_cloud.py --workdir run --follow"
echo "dreams: $py cloud/monitor_cloud.py --workdir run --dreams"
