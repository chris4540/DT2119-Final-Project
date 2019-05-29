#!/bin/bash
mkdir -p ./experiment
for part_labeled in 0.1 0.3; do
    export part_labeled
    for optim in adam stepLR cyclicLR; do
        export optim
        python -u cmp_optim.py |tee ./experiment/cmp_optim_${optim}_plbl_${part_labeled}.log
    done
done