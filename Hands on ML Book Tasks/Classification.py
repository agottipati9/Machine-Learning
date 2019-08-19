import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import numpy as np
import pandas as pd

mnst = fetch_mldata('MNIST original')
X, y = mnst["data"], mnst["target"]

some_digit = X[1700]
# some_digit_image = some_digit.reshape(28, 28) (784 = 28^2)

# plt.imshow(some_digit_image, cmap=plt.cm.binary,
#           interpolation="nearest")
# plt.axis("off")
# plt.show()

# print(y[1700])

X_train, X_test, y_train, y_test = X[:60_000], X[60_000:], y[:60_000], y[60_000:]
# Mnist is already cleaned and split intro test and training set

# Still need to shuffle data because cross vals will be similar
#  and some algorithms perform worse without variation in data
shuffled_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffled_index], y_train[shuffled_index]

"""Training Binary Classifier"""
# Test whether 5 or not

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, y_train_5)

# print(sgd_clf.predict([some_digit]))

"""Implementing Cross-Validation"""
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)
# Stratified K Fold class perfroms stratified sampling (representative folds of data)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    # print(n_correct / len(y_pred)) prints 0.95995, 0.9669, 0.96665
    # Each iteration code creates a clone of the classifier, trains that clone on the training folds, and
    # makes predictions on the test fold. Then it counts the number of correct predictions
    # and outputs the ratio of correct predictions.

