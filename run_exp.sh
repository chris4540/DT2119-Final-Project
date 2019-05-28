#!/bin/bash
#               Experiement
#   Possible opts for part_labeled:
#       0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5
#   Possible opts for temperature
#       0.5, 1, 2, 4, 6, 8, 10
# Setting of the precentage of the labeled data
export part_labeled=0.3

# train teacher
python -u train_teacher.py | tee teacher_plbl_${part_labeled}.log

# run semi-supervised learning
for temp in 0.5 1 2 4 6 8 10; do
    export temp=${temp}
    python -u train_student.py | tee student_plbl_${part_labeled}_T${temp}.log
done
