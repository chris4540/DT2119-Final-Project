#!/bin/bash
for part_labeled in 0.1 0.3; do
    for optim in adam stepLR cyclicLR; do
        export optim
        python -u cmp_optim.py |tee cmp_optim_${optim}_plbl_${part_labeled}.log
    done
done