#!/bin/bash
# export part_labeled=0.1
for optim in adam stepLR; do
    export optim
    python -u cmp_optim.py |tee cmp_optim_${optim}.log
done