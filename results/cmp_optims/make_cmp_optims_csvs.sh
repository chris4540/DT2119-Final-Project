#!/bin/bash
#
# The whole experiment is done with Config.part_labeled = 0.3
#
set -e

# mark the project root dir
root_dir=`readlink -f ../..`
part_labeled=0.3
# cyclicLR_loss.txt
for optim in adam stepLR cyclicLR; do
    echo ${optim} > ${optim}_loss.txt
    cp ${optim}_loss.txt ${optim}_train_acc.txt
    cp ${optim}_loss.txt ${optim}_valid_acc.txt
    cp ${optim}_loss.txt ${optim}_test_acc.txt
    file=${root_dir}/experiment/cmp_optim_${optim}_plbl_${part_labeled}.log
    grep Loss ${file} | awk '{print $6}' >> ${optim}_loss.txt
    grep Train ${file} | awk '{print $10}' >> ${optim}_train_acc.txt
    grep Eval ${file} | grep Valid | awk '{print $5}' >> ${optim}_valid_acc.txt
    grep Eval ${file} | grep Test | awk '{print $5}' >> ${optim}_test_acc.txt
done
paste -d"," cyclicLR_loss.txt adam_loss.txt stepLR_loss.txt > cmp_loss.csv
paste -d"," cyclicLR_train_acc.txt adam_train_acc.txt stepLR_train_acc.txt > cmp_train_acc.csv
paste -d"," cyclicLR_test_acc.txt adam_test_acc.txt stepLR_test_acc.txt > cmp_test_acc.csv
paste -d"," cyclicLR_valid_acc.txt adam_valid_acc.txt stepLR_valid_acc.txt > cmp_valid_acc.csv
rm *.txt