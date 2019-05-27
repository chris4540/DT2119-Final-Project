#!/bin/bash
# Setting of the precentage of the labeled data
export part_labeled=0.3

# run semi-supervised learning
for temp in 1 2 4 8 10; do
    export temp=${temp}
    python -u semi_main.py | tee log_plbl_${part_labeled}_T${temp}.log
done
