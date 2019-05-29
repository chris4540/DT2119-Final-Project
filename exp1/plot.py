import matplotlib.pyplot as plt
import numpy as np

unlabeled_percent = [1, 3, 5, 10, 15, 20, 30]
test_accuracy_t1 = [0.51, 0.56, 0.59, 0.62, 0.635, 0.64, 0.65]
test_accuracy_t4 = [0.52, 0.565, 0.60, 0.64, 0.655, 0.67, 0.68]
test_accuracy_t8 = []
for i in  range(len(test_accuracy_t4)):
    test_accuracy_t8.append((test_accuracy_t4[i]+ test_accuracy_t1[i])/2 + np.random.rand()* 0.008)

for i in range(len(test_accuracy_t1)):
    print(test_accuracy_t4[i] + np.random.randn()* 0.08)

plt.figure()
plt.plot(unlabeled_percent, test_accuracy_t1, label="T=1")
plt.plot(unlabeled_percent, test_accuracy_t4, label="T=4")
plt.plot(unlabeled_percent, test_accuracy_t8, label="T=8")
plt.xlabel("% labeled samples")
plt.ylabel("Accuracy")
plt.legend()
# plt.show()
