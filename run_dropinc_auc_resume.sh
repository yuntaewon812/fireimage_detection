#!/bin/bash
cd "$(dirname "$0")"

run_script() {
    echo ""
    echo "############################################################"
    echo "  Running: $1"
    echo "############################################################"
    python -u "$1"
    if [ $? -eq 0 ]; then
        echo "[OK] $1 finished successfully"
    else
        echo "[ERROR] $1 failed with return code $?"
    fi
}

run_script dropIncLimeShap.py
run_script aucGradSaliency.py
run_script aucLimeShap.py

echo ""
echo "All DropInc/AUC scripts complete!"
