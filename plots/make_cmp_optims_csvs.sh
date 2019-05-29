#!/bin/bash
#
# The whole experiment is done with Config.part_labeled = 0.3
#
set -e

# mark the project root dir
root_dir=`readlink -f ..`

# cyclicLR_loss.txt
echo "cyclicLR" > cyclicLR_loss.txt
grep Loss ${root_dir}/experiment/teacher_plbl_0.3.log | awk '{print $6}' >> cyclicLR_loss.txt

# adam_loss.txt
echo "adam" > adam_loss.txt
grep Loss ${root_dir}/experiment/cmp_optim_adam.log | awk '{print $6}' >> adam_loss.txt
# stepLR_loss.txt
echo "stepLR" > stepLR_loss.txt
grep Loss ${root_dir}/experiment/cmp_optim_stepLR.log | awk '{print $6}' >> stepLR_loss.txt
paste -d"," cyclicLR_loss.txt adam_loss.txt stepLR_loss.txt > cmp_loss.csv
rm cyclicLR_loss.txt adam_loss.txt stepLR_loss.txt
# ==============================================================================
# compare the training acc.
# cyclicLR_acc.txt
echo "cyclicLR" > cyclicLR_tacc.txt
grep Train ${root_dir}/experiment/teacher_plbl_0.3.log | awk '{print $10}' >> cyclicLR_tacc.txt

# adam_tacc.txt
echo "adam" > adam_tacc.txt
grep Train ${root_dir}/experiment/cmp_optim_adam.log | awk '{print $10}' >> adam_tacc.txt
# stepLR_tacc.txt
echo "stepLR" > stepLR_tacc.txt
grep Train ${root_dir}/experiment/cmp_optim_stepLR.log | awk '{print $10}' >> stepLR_tacc.txt
paste -d"," cyclicLR_tacc.txt adam_tacc.txt stepLR_tacc.txt > cmp_tacc.csv
rm cyclicLR_tacc.txt adam_tacc.txt stepLR_tacc.txt