import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
plt.figure(figsize=(10,9))

# Building the dataset
mean0 = [0,0]
cov0 = [[2,3],[3,1]]
dataset0 = np.random.multivariate_normal(mean0, cov0, 1000)

mean1 = [1,0]
cov1 = [[1,0],[0,5]]
dataset1 = np.random.multivariate_normal(mean1, cov1, 1000)

mean2 = [4,7]
cov2 = [[9,0],[0,1]]
dataset2 = np.random.multivariate_normal(mean2, cov2, 1000)

mean3 = [4,5]
cov3 = [[9,-9],[-9,6]]
dataset3 = np.random.multivariate_normal(mean3, cov3, 1000)

mean4 = [6,6]
cov4 = [[1,0],[0,1]]
dataset4 = np.random.multivariate_normal(mean4, cov4, 1000)


mean5 = [5,5]
cov5 = [[1,1],[1,1]]
dataset5 = np.random.multivariate_normal(mean5, cov5, 1000)

mean6 = [7,8]
cov6 = [[1,0],[0,3]]
dataset6 = np.random.multivariate_normal(mean6, cov6, 1000)

mean7 = [4,9]
cov7 = [[9,-3],[-3,2]]
dataset7 = np.random.multivariate_normal(mean7, cov7, 1000)

mean8 = [5,2]
cov8 = [[5,3],[3,1]]
dataset8 = np.random.multivariate_normal(mean8, cov8, 1000)

mean9 = [8,1]
cov9 = [[5,1],[1,5]]
dataset9 = np.random.multivariate_normal(mean9, cov9, 1000)

X = np.vstack((dataset0, dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9))
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

#
# Linear Classification
#

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

plt.subplot(2, 1, 1)
plt.title('Linear Classification | Accuracy: ' + str(accuracy))
plt.scatter(x_training_class_0, y_training_class_0, color = 'purple', label='Training class 0')
plt.scatter(x_training_class_1, y_training_class_1, color = 'yellow', label='Training class 1')

plt.scatter(x_correct_0, y_correct_0, color = 'green', label='Correct class 0')
plt.scatter(x_correct_1, y_correct_1, color = 'blue', label='Correct class 1')

plt.scatter(x_incorrect_0, y_incorrect_0, color = 'black', label='Incorrect class 0')
plt.scatter(x_incorrect_1, y_incorrect_1, color = 'red', label='Incorrect class 1')

plt.legend()

#
# Nearest Neighbor Classification
#
kd_tree = cKDTree(training_data, 1)

prediction_indices = kd_tree.query(testing_data)
predictions = training_labels[prediction_indices[1]]

pred_correct_0 = None
pred_incorrect_0 = None
pred_correct_1 = None
pred_incorrect_1 = None
correct_data = 0

for i in range(predictions.shape[0]):
    if predictions[i] == 0:
        if predictions[i] == testing_labels[i]:
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
        if predictions[i] == testing_labels[i]:
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

# Plot
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

plt.subplot(2, 1, 2)
plt.title('KDTree NN Classification | Accuracy: ' + str(accuracy))
plt.scatter(x_training_class_0, y_training_class_0, color = 'purple', label='Training class 0')
plt.scatter(x_training_class_1, y_training_class_1, color = 'yellow', label='Training class 1')

plt.scatter(x_correct_0, y_correct_0, color = 'green', label='Correct class 0')
plt.scatter(x_correct_1, y_correct_1, color = 'blue', label='Correct class 1')

plt.scatter(x_incorrect_0, y_incorrect_0, color = 'black', label='Incorrect class 0')
plt.scatter(x_incorrect_1, y_incorrect_1, color = 'red', label='Incorrect class 1')

plt.legend()
plt.show()
