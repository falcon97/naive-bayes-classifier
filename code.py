import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.stats as st
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Function to predict classes for a univariate distribution


def class_prediction_univariate(data, dataset_name):
    print('Predicting classes for feature', dataset_name)
    data_train, data_test = data[:int(
        data.shape[0] * 0.1), :], data[int(data.shape[0] * 0.1):, :]

    # Calculate mean and standard deviation for each class
    data_mean = data_train.mean(axis=0)
    data_std = data_train.std(axis=0)

    # Calculate probabilities for each class and predict most probable class
    data_predictions = np.zeros(data_test.shape)
    for row in range(0, data_test.shape[0]):
        for col in range(0, data_test.shape[1]):
            data_predictions[row, col] = np.argmax(st.norm.pdf(
                data_test[row, col], data_mean, data_std)) + 1

    true_data = np.tile([1, 2, 3, 4, 5], (900, 1))

    # Calculate classification accuracy and error rate
    accuracy = sum(sum(np.equal(true_data, data_predictions))) / \
        data_predictions.size
    print('Classification accuracy:', accuracy)
    print('Error rate:', 1 - accuracy)
    return accuracy

# Function to predict classes for a multivariate distribution


def class_prediction_multivariate(data1, data2, dataset_name):
    print('Predicting classes for feature', dataset_name)
    data1_train, data1_test = data1[:int(
        data1.shape[0] * 0.1), :], data1[int(data1.shape[0] * 0.1):, :]
    data2_train, data2_test = data2[:int(
        data2.shape[0] * 0.1), :], data2[int(data2.shape[0] * 0.1):, :]

    # Calculate mean and standard deviation for each class
    data1_mean = data1_train.mean(axis=0)
    data1_std = data1_train.std(axis=0)
    data2_mean = data2_train.mean(axis=0)
    data2_std = data2_train.std(axis=0)

    # Calculate probabilities for each class and predict most probable class
    data_predictions = np.zeros(data1_test.shape)
    for row in range(0, data1_test.shape[0]):
        for col in range(0, data1_test.shape[1]):
            data_predictions[row, col] = np.argmax(st.norm.pdf(
                data1_test[row, col], data1_mean, data1_std)*st.norm.pdf(
                data2_test[row, col], data2_mean, data2_std)) + 1

    true_data = np.tile([1, 2, 3, 4, 5], (900, 1))

    # Calculate classification accuracy and error rate
    accuracy = sum(sum(np.equal(true_data, data_predictions))) / \
        data_predictions.size
    print('Classification accuracy:', accuracy)
    print('Error rate:', 1 - accuracy)
    return accuracy


# Load data
data = loadmat('data.mat')
f1 = data['F1']
f2 = data['F2']

f1_accuracy = class_prediction_univariate(f1, 'F1') * 100
f2_accuracy = class_prediction_univariate(f2, 'F2') * 100

# Normalizing F1
z1 = preprocessing.normalize(f1)
z1_accuracy = class_prediction_univariate(z1, 'Z1') * 100

multivariate_accuracy = class_prediction_multivariate(z1, f2, '[Z1 F2]') * 100

# Plotting F1 against F2
plt.scatter(f1[:, 0], f2[:, 0], c='b', marker='.', label='C1')
plt.scatter(f1[:, 1], f2[:, 1], c='g', marker='.', label='C2')
plt.scatter(f1[:, 2], f2[:, 2], c='r', marker='.', label='C3')
plt.scatter(f1[:, 3], f2[:, 3], c='k', marker='.', label='C4')
plt.scatter(f1[:, 4], f2[:, 4], c='c', marker='.', label='C5')
plt.legend(loc='upper left')
plt.xlabel('F1')
plt.ylabel('F2')
plt.suptitle('F1 vs F2')
plt.show()

# Plotting Z1 against F2
plt.scatter(z1[:, 0], f2[:, 0], c='b', marker='.', label='C1')
plt.scatter(z1[:, 1], f2[:, 1], c='g', marker='.', label='C2')
plt.scatter(z1[:, 2], f2[:, 2], c='r', marker='.', label='C3')
plt.scatter(z1[:, 3], f2[:, 3], c='k', marker='.', label='C4')
plt.scatter(z1[:, 4], f2[:, 4], c='c', marker='.', label='C5')
plt.legend(loc='upper left')
plt.xlabel('Z1')
plt.ylabel('F2')
plt.suptitle('Z1 vs F2')
plt.show()

# Comparing all accuracy values
accuracy_values = [f1_accuracy, f2_accuracy,
                   z1_accuracy, multivariate_accuracy]
plt.barh(['F1', 'F2', 'Z1', '[Z1 F2]'], accuracy_values, color='indigo')
plt.xlabel('Accuracy (%)')
plt.ylabel('Feature')
plt.suptitle('Comparision of classification accuracy')
plt.show()
