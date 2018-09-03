import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree

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

x1, y1 = dataset0.T
x2, y2 = dataset1.T

plt.scatter(x1, y1, c = 'blue')
plt.scatter(x2, y2, c = 'red')
plt.show()