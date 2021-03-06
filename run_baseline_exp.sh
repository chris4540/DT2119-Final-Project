#!/bin/bash
mkdir -p ./experiment
for part_labeled in 0.01 0.03 0.05 0.1 0.2 0.3 0.5; do
    export part_labeled
    python -u train_baseline.py | tee ./experiment/baseline_plbl_${part_labeled}.log
done