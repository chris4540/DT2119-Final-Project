#!/bin/bash
export part_labeled=0.3
for temp in 1 2 4 8 10; do
    export temp=${temp}
    python semi_main.py | tee log_plbl_${part_labeled}_T${temp}.log
done