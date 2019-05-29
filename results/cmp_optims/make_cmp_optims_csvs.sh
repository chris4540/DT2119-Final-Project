#!/bin/bash
#
# The whole experiment is done with Config.part_labeled = 0.3
#
set -e

# mark the project root dir
root_dir=`readlink -f ../..`

# cyclicLR_loss.txt
echo "cyclicLR" > cyclicLR_loss.txt
grep Loss ${root_dir}/experiment/baseline_plbl_0.3.log | awk '{print $6}' >> cyclicLR_loss.txt

# adam_loss.txt
echo "adam" > adam_loss.txt
grep Loss ${root_dir}/experiment/cmp_plbl_0.3_adam.log | awk '{print $6}' >> adam_loss.txt
# stepLR_loss.txt
echo "stepLR" > stepLR_loss.txt
grep Loss ${root_dir}/experiment/cmp_plbl_0.3_stepLR.log | awk '{print $6}' >> stepLR_loss.txt
paste -d"," cyclicLR_loss.txt adam_loss.txt stepLR_loss.txt > cmp_loss.csv
rm cyclicLR_loss.txt adam_loss.txt stepLR_loss.txt
# ==============================================================================
# compare the training acc.
# cyclicLR_train_acc.txt
echo "cyclicLR" > cyclicLR_train_acc.txt
grep Train ${root_dir}/experiment/baseline_plbl_0.3.log | awk '{print $10}' >> cyclicLR_train_acc.txt

# adam_train_acc.txt
echo "adam" > adam_train_acc.txt
grep Train ${root_dir}/experiment/cmp_plbl_0.3_adam.log | awk '{print $10}' >> adam_train_acc.txt
# stepLR_train_acc.txt
echo "stepLR" > stepLR_train_acc.txt
grep Train ${root_dir}/experiment/cmp_plbl_0.3_stepLR.log | awk '{print $10}' >> stepLR_train_acc.txt
paste -d"," cyclicLR_train_acc.txt adam_train_acc.txt stepLR_train_acc.txt > cmp_train_acc.csv
rm cyclicLR_train_acc.txt adam_train_acc.txt stepLR_train_acc.txt

# ====================================================
# compare the Validation Acc.
# cyclicLR_valid_acc.txt
echo "cyclicLR" > cyclicLR_valid_acc.txt
grep Eval ${root_dir}/experiment/baseline_plbl_0.3.log | grep Valid | awk '{print $5}' >> cyclicLR_valid_acc.txt

# adam_valid_acc.txt
echo "adam" > adam_valid_acc.txt
grep Eval ${root_dir}/experiment/cmp_plbl_0.3_adam.log | grep Valid | awk '{print $5}' >> adam_valid_acc.txt
# stepLR_valid_acc.txt
echo "stepLR" > stepLR_valid_acc.txt
grep Eval ${root_dir}/experiment/cmp_plbl_0.3_stepLR.log | grep Valid | awk '{print $5}' >> stepLR_valid_acc.txt
paste -d"," cyclicLR_valid_acc.txt adam_valid_acc.txt stepLR_valid_acc.txt > cmp_valid_acc.csv
rm cyclicLR_valid_acc.txt adam_valid_acc.txt stepLR_valid_acc.txt
# ====================================================

# compare the Test Acc.
# cyclicLR_valid_acc.txt
echo "cyclicLR" > cyclicLR_test_acc.txt
grep Eval ${root_dir}/experiment/baseline_plbl_0.3.log | grep Test | awk '{print $5}' >> cyclicLR_test_acc.txt

# adam_test_acc.txt
echo "adam" > adam_test_acc.txt
grep Eval ${root_dir}/experiment/cmp_plbl_0.3_adam.log | grep Test | awk '{print $5}' >> adam_test_acc.txt
# stepLR_test_acc.txt
echo "stepLR" > stepLR_test_acc.txt
grep Eval ${root_dir}/experiment/cmp_plbl_0.3_stepLR.log | grep Test | awk '{print $5}' >> stepLR_test_acc.txt
paste -d"," cyclicLR_test_acc.txt adam_test_acc.txt stepLR_test_acc.txt > cmp_test_acc.csv
rm cyclicLR_test_acc.txt adam_test_acc.txt stepLR_test_acc.txt
