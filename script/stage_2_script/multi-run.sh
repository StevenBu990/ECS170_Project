#!/bin/bash
# runs script_mlp.py once for every config file in configs/
cd "$(dirname "$0")"        # script_mlp.py uses paths relative to this folder
export PYTHONPATH=../..     # so "from local_code..." imports work

for cfg in ../../configs/5-epoch_runs/*.json; do
    echo "===== $cfg ====="
    python script_mlp.py "$cfg"
done
