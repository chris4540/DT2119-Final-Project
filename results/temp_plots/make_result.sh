#!/bin/bash
# This script is to collect the validation acc. and test acc. from the log files
# and translate them to csv

set -e

# mark the project root dir
root_dir=`readlink -f ../..`
valid_csv="valid_temp.csv"
test_csv="test_temp.csv"
baseline_acc_csv="baseline_acc.csv"
teacher_acc_csv="teacher_acc.csv"


# Create index
echo "" > ${valid_csv}
for part_labeled in 0.01 0.03 0.05 0.1 0.2 0.3 0.5; do
    echo $part_labeled >> $valid_csv
done
# copy index
cp $valid_csv $test_csv
# =====================================================================

for temp in 0.5 1 2 4 6 8 10; do
    echo $temp > valid_acc.txt
    echo $temp > test_acc.txt
    for part_labeled in 0.01 0.03 0.05 0.1 0.2 0.3 0.5; do
        file=${root_dir}/experiment/student_plbl_${part_labeled}_T${temp}.log
        tail -n 1 $file | awk '{print $9}' >> valid_acc.txt
        tail -n 1 $file | awk '{print $12}' >> test_acc.txt
    done
    paste -d"," ${valid_csv} valid_acc.txt > ${valid_csv}.tmp
    paste -d"," ${test_csv} test_acc.txt > ${test_csv}.tmp

    mv -f ${valid_csv}.tmp ${valid_csv}
    mv -f ${test_csv}.tmp ${test_csv}
done

rm valid_acc.txt test_acc.txt
# ============================================================
# Make baseline csv
# index
echo "" > ${baseline_acc_csv}
for part_labeled in 0.01 0.03 0.05 0.1 0.2 0.3 0.5; do
    echo $part_labeled >> ${baseline_acc_csv}
done

echo "valid_acc" > valid_acc.txt
echo "test_acc" > test_acc.txt
for part_labeled in 0.01 0.03 0.05 0.1 0.2 0.3 0.5; do
    file=${root_dir}/experiment/baseline_plbl_${part_labeled}.log
    tail -n 1 $file | awk '{print $9}' >> valid_acc.txt
    tail -n 1 $file | awk '{print $12}' >> test_acc.txt
done
paste -d"," ${baseline_acc_csv} valid_acc.txt test_acc.txt > ${baseline_acc_csv}.tmp
mv -f ${baseline_acc_csv}.tmp ${baseline_acc_csv}

rm valid_acc.txt
rm test_acc.txt
# ============================================================
echo "" > ${teacher_acc_csv}
for part_labeled in 0.01 0.03 0.05 0.1 0.2 0.3 0.5; do
    echo $part_labeled >> ${teacher_acc_csv}
done

echo "valid_acc" > valid_acc.txt
echo "test_acc" > test_acc.txt
for part_labeled in 0.01 0.03 0.05 0.1 0.2 0.3 0.5; do
    file=${root_dir}/experiment/teacher_plbl_${part_labeled}.log
    tail -n 1 $file | awk '{print $9}' >> valid_acc.txt
    tail -n 1 $file | awk '{print $12}' >> test_acc.txt
done
paste -d"," ${teacher_acc_csv} valid_acc.txt test_acc.txt > ${teacher_acc_csv}.tmp
mv -f ${teacher_acc_csv}.tmp ${teacher_acc_csv}

rm valid_acc.txt
rm test_acc.txt
# ============================================================