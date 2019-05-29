#!/bin/bash
set -e

# mark the project root dir
root_dir=`readlink -f ../..`
valid_csv="valid_temp.csv"
test_csv="test_temp.csv"


echo "" > ${valid_csv}
for part_labeled in 0.01 0.03 0.05 0.1 0.2 0.3 0.5; do
    echo $part_labeled >> $valid_csv
done
# copy it
cp $valid_csv $test_csv

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