import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import matplotlib.patches as mpatches

mean1 = [1,2]
cov1 = [[2,3],[3,1]]
dataset0 = np.random.multivariate_normal(mean1, cov1, 5000)

mean2 = [4,7]
cov2 = [[3,-0.5],[-0.5,2]]
dataset1 = np.random.multivariate_normal(mean2, cov2, 5000)

X = np.vstack((dataset0, dataset1))
y_raw = np.append(np.zeros(5000, dtype='int64'), np.ones(5000, dtype='int64'))
y = y_raw[:, np.newaxis]

mask = np.random.rand(10000) < 0.8
y_mask = mask[:, np.newaxis]

training_labels = y[y_mask]
training_data = X[mask]

mask = np.logical_not(mask)
y_mask = np.logical_not(y_mask)

testing_labels = y[y_mask]
testing_data = X[mask]

X_transposed = training_data.T
X_dot_product = X_transposed.dot(training_data)
B = inv(X_dot_product).dot(X_transposed).dot(training_labels)

pred_correct_0 = None
pred_incorrect_0 = None
pred_correct_1 = None
pred_incorrect_1 = None
correct_data = 0
for i in range(testing_data.shape[0]):
    pred = testing_data[i].T.dot(B)
    if pred < 0.5:
        if testing_labels[i] == 0:
            if pred_correct_0 is None:
                pred_correct_0 = testing_data[i]
            else:
                pred_correct_0 = np.vstack((pred_correct_0, testing_data[i]))
            correct_data += 1
        else:
            if pred_incorrect_0 is None:
                pred_incorrect_0 = testing_data[i]
            else:
                pred_incorrect_0 = np.vstack((pred_incorrect_0, testing_data[i]))
    else:
        if testing_labels[i] == 1:
            if pred_correct_1 is None:
                pred_correct_1 = testing_data[i]
            else:
                pred_correct_1 = np.vstack((pred_correct_1, testing_data[i]))
            correct_data += 1
        else:
            if pred_incorrect_1 is None:
                pred_incorrect_1 = testing_data[i]
            else:
                pred_incorrect_1 = np.vstack((pred_incorrect_1, testing_data[i]))

accuracy = correct_data/testing_data.shape[0]
print("Accuracy: " + str(accuracy))

class_0_mask = np.where(training_labels == 0)
class_1_mask = np.where(training_labels == 1)

training_data_class_0 = training_data[class_0_mask]
training_data_class_1 = training_data[class_1_mask]

x_training_class_0, y_training_class_0 = training_data_class_0.T
x_training_class_1, y_training_class_1 = training_data_class_1.T

x_correct_0, y_correct_0 = pred_correct_0.T
x_correct_1, y_correct_1 = pred_correct_1.T

x_incorrect_0, y_incorrect_0 = pred_incorrect_0.T
x_incorrect_1, y_incorrect_1 = pred_incorrect_1.T

plt.scatter(x_training_class_0, y_training_class_0, color = 'purple', label='Training class 0')
plt.scatter(x_training_class_1, y_training_class_1, color = 'yellow', label='Training class 1')

plt.scatter(x_correct_0, y_correct_0, color = 'green', label='Correct class 0')
plt.scatter(x_correct_1, y_correct_1, color = 'blue', label='Correct class 1')

plt.scatter(x_incorrect_0, y_incorrect_0, color = 'black', label='Incorrect class 0')
plt.scatter(x_incorrect_1, y_incorrect_1, color = 'red', label='Incorrect class 1')

plt.title('Linear Classification | Accuracy: ' + str(accuracy))
plt.legend()
plt.show()
