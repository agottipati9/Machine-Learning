import sklearn, scipy, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from pandas.plotting import scatter_matrix

DATA_PATH = os.path.join("datasets", "pulsarStar")

"""Returns dataframe of the dataset"""
def load_data(path=DATA_PATH):
    csv_path = os.path.join(path, "pulsar_stars.csv")
    return pd.read_csv(csv_path)

pulsar = load_data()

"""Useful information about the data. """
# print(pulsar.info())
# print(pulsar.describe())
# No Null elements exist

"""Visualize the data"""
# pulsar.hist(bins=50, figsize=(20, 15))
# plt.show()

"""Splitting the training and test set via SciKit Learn"""
train_set, test_set = train_test_split(pulsar, test_size=0.2, random_state=42)

"""Check correlation between features. There is very little correlation between the features and target."""
# attributes = ["target_class", " Standard deviation of the integrated profile", " Mean of the integrated profile"]
# scatter_matrix(pulsar[attributes], figsize=(12,8))
# plt.show()

train_set_clean = train_set.drop("target_class", axis=1)
train_set_labels = train_set["target_class"].copy()

"""Building a model."""
X = train_set_clean
y = train_set_labels

# clf = svm.SVC(gamma='scale')
# clf.fit(X, y)

# some_data = train_set_clean[:40]
# some_labels = train_set_labels[:40]
# print("Predictions: ", clf.predict(some_data))
# print("Labels: ", list(some_labels))

"""Measure the error of current model."""
# clf_predictions = clf.predict(train_set_clean)
# clf_mse = mean_squared_error(train_set_labels, clf_predictions)
# clf_rmse = np.sqrt(clf_mse)
# print(clf_rmse) # .1652

"""Try other models"""
forest_reg = RandomForestRegressor()
# forest_reg.fit(X, y)

# forest_predictions = forest_reg.predict(train_set_clean)
# forest_mse = mean_squared_error(train_set_labels, forest_predictions)
# forest_rmse = np.sqrt(forest_mse)
# print(forest_rmse) # Best score of 0.0593

# forest_clf = RandomForestClassifier()
# forest_clf.fit(X, y)

# forest_clf_predictions = forest_clf.predict(train_set_clean)
# forest_clf_mse = mean_squared_error(train_set_labels, forest_clf_predictions)
# forest_clf_rmse = np.sqrt(forest_clf_mse)
# print(forest_clf_rmse) # 0.0596

"""Utilize cross validation"""
# forest_reg_scores = cross_val_score(forest_reg, train_set_clean, train_set_labels,
#                          scoring="neg_mean_squared_error", cv=10) # cv = 10 -> 10 evaluations.
# Set is split 9 : 1 and evaulated for all combinations.
# forest_reg_rmse = np.sqrt(-forest_reg_scores) # Expected a utility function so greater -> negative


def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Std: ", scores.std())


# print_scores(forest_reg_rmse)
"""Scores:  [0.14734617 0.13749048 0.13900585 0.13263025 0.13286697 0.14974373
 0.13391401 0.13034655 0.13401294 0.14440412]
Mean:  0.13817610609014436
Std:  0.006431093105235854 """

"""Compare the scores and decide on which model"""
# forest_clf_scores = cross_val_score(forest_clf, train_set_clean, train_set_labels,
#                          scoring="neg_mean_squared_error", cv=10)
# forest_clf_rmse = np.sqrt(-forest_clf_scores)
# print_scores(forest_clf_rmse)
"""Scores:  [0.15855493 0.15633727 0.14713275 0.13212911 0.1423074  0.15633727
 0.13474578 0.12673379 0.15185782 0.14718415]
Mean:  0.14533202378036963
Std:  0.010525033275103826"""

# clf_scores = cross_val_score(clf, train_set_clean, train_set_labels,
#                          scoring="neg_mean_squared_error", cv=10)
# clf_rmse = np.sqrt(-clf_scores)
# print_scores(clf_rmse)
"""Scores:  [0.17528907 0.1772698  0.15180478 0.16074199 0.15855493 0.17528907
 0.1650292  0.15408769 0.17929125 0.16508685]
Mean:  0.16624446330228604
Std:  0.009507370028500748"""

"""Fine tune model with Grid Search CV"""
param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
              {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
              ]

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')

grid_search.fit(train_set_clean, train_set_labels)

# print(grid_search.best_params_)
# print(grid_search.best_estimator_)

"""Final test of model"""
final_model = grid_search.best_estimator_

# X_test = test_set.drop("target_class", axis=1)
# y_test = test_set["target_class"].copy()

# final_predictions = final_model.predict(X_test)
# final_mse = mean_squared_error(y_test, final_predictions)
# final_rmse = np.sqrt(final_mse)
# print(final_rmse) # 0.12374133530601127

"""Saving model"""
# joblib.dump(final_model, "pulsar_model.pkl")

# If want to load, simply use joblib.load(filename)
