import scipy, sklearn
import os, tarfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from six.moves import urllib
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

"""Retrieves the housing data as a tgz file, then extracts it to retrieve the csv."""
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
       os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


"""Returns Pandas DataFrame object containing all the data."""
def load_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

"""Splits the dataset based on the given test ratio."""
def split_train_set(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

"""One way to ensure we are testing on same data (useful for cross validation)"""
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

"""Ensure we are testing on the same data (useful for corss validation)"""
def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing = load_data()

"""Displays top 5 rows of data"""
# print(housing.head())

"""Provides quick description of data (num rows, types, non-nulls, etc"""
# print(housing.info())

"""How many categories exist/how many districts belong to this category"""
# print(housing["ocean_proximity"].value_counts())

"""Summary of Numerical Attributes (nulls are ignored)"""
# print(housing.describe())

"""Displays data as a histogram. Relies on matplotlib."""
# housing.hist(bins=50, figsize=(20,15))
# plt.show()

"""Splitting data set. This can generate varying sets which is bad. 
To avoid we can seed the np.random so that we obtain the same values everytime. 
We could also try hashing."""
# train_set, test_set = split_train_set(housing, 0.2)
# print(len(train_set), len(test_set))

"""Solution: adding an indexing column"""
# housing_with_id = housing.reset_index() #adds an index column b/c housing data does not have one
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

"""If use row index as identifier, make sure new data is appended, and no row ever gets deleted
If not possible, use most stable feature (latitude and longitude in this case). 
Latitude and Longitude is guaranteed to not vary much over many years."""
# housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

"""Scikit-Learn provides a splitter similar to the one above, except has random_state parameter.
This seeds the np.random to ensure we get the same values every time. 
Use this preferably when crafting training and test sets."""
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

"""Creating stratified testing set, so that it is representative
of the whole group we are trying to predict on. Dividing by 1.5 to limit the number of categories, 
and rounding up to produce discrete categories (creates representative test set of median income."""
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
# housing["income_cat"].hist()
# plt.show()

"""Stratified Sampling based on income category. Avoids the skewing of sampling bias (random sample)."""
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

"""Remove income_cat attribute so data is back in original form"""
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

"""Creates a copy of the training set to ensure 
we do not alter the training set and do not work on the test set."""
housing = strat_train_set.copy()

"""Since using long and lat, use scatter plot to visualize data. 
Alpha allows us to see areas of high density easier. Radius of circle = population (option s).
Color represents price (option c). Predefined color map (cmap) called jet, 
low val = blue, high val = red."""
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10, 7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
# plt.legend()
# plt.show()

"""Standard Correlation Coefficient"""
# corr_matrix = housing.corr()
# print(corr_matrix)

"""Another way to check correlation. 
Plots every numerical attribute against each other i.e. (11 attributes yields 11^2 plots.
Pandas displays histogram instead of straight line 
for the same attribute being compared to itself."""
# from pandas.plotting import scatter_matrix

# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

"""Zoom in on median_income because strong correlation with median_house_value"""
# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()

"""Good idea is to test different combination of features"""
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_household"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

"""Cleaning Data"""
"""Separating predictors and labels because we don't want to apply same transformations"""
housing = strat_train_set.drop("median_house_value", axis=1)  # drop creates a copy of data without attribute
housing_labels = strat_train_set["median_house_value"].copy()

"""Handling data with missing values. 3 options. """
# housing.dropna(subset=["total_bedrooms"]) # Get rid of districts with missing values
# housing.drop("total_bedrooms", axis=1)  # Get rid of entire attribute that has some missing values

"""Option 3: Median. Calculate median from training set and fill values in training set.
Save Median value to replace in test set and for future reference. """
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)  # Compute median and fill missing values with median

"""Scikit_Learn Imputer handles missing values"""
from sklearn.impute import SimpleImputer as ip

imputer = ip(strategy="median")  # Only works with numerical attributes
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num) # Train imputer
# print(imputer.statistics_)  # Values are stored in the statistics_ instance variable
# Even though only total_bedrooms had missing values, we should run imputer only all attributes in case
X = imputer.transform(housing_num)  # Numpy array with transformed features
housing_tr = pd.DataFrame(X, columns=housing_num.columns)  # Insert back into data frame

"""Handling non-numerical data"""
housing_cat = housing["ocean_proximity"]
housing_cat_encoded, housing_categories = housing_cat.factorize() # Maps category to different integer
# print(housing_cat_encoded[:10])
# print(housing_categories[:10]) # See where each category was mapped to
# ML algos will see two close numbers as "close" categories i.e 0 and 2 closer than 0 and 4
# To fix this, use one binary attirbute per category
# EX: one-hot encoding (OneHotEncoder) 0 = far 1 = close to ocean

# from sklearn.preprocessing import OneHotEncoder
# encoder = OneHotEncoder()
# housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
# fit_transform() expects 2D array so we need to reshape our 1D array to 2D
# encoder.fit_transform() returns a sparse SciPy Matrix which is really good for
# representing categorical with thousands of categories. Matrix only stores location of non-zero
# (otherwise a ton of wasted space to store 0s). Can get dense numPy array with .toarray()

"""Can convert directly from categorical to integers to one hot encoding with 
CategoricalEncoder class in SciKit 0.19.0+"""
from sklearn.preprocessing import CategoricalEncoder
# cat_encoder = CategoricalEncoder()
# housing_cat_reshaped = housing.cat.values.reshape(-1, 1)
# housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped) # Out puts sparse but can be set to onehot-dense
# CategoricalEncoder(encoding="onehot-dense") # list categories with categories_ instance variable

"""Small transformer class that adds the combined attributes"""
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

"""Small pipeline for numerical attributes"""
"""Pipeline constuctor takes a list of name/estimator pairs defining a sequence of steps.
All but last estimator must be transformers (must have fit_transform() method). Names can
be anything except cannot contain double underscore."""
num_pipeline = Pipeline([('imputer', ip(strategy="median")),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scalar', StandardScaler()),
                         ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


"""Custom transformer for feeding in non-numerical columns directly into our pipeline 
(instead of manually extracting)"""
from sklearn.base import BaseEstimator, TransformerMixin

"""Transform data by selecting desired attributes, dropping 
the rest, and converting resulting DataFrame to Numpy array. 
With this, can easily make pipeline that handles only numerical values: 
start with DataFrameSelector and the previous preprocessing steps. """


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

"""Can write another pipeline for categorical attributes by simply selecting 
categorical attributes using DataFrameSelector
and then apply CategoricalEncoder"""
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),
                         ('imputer', ip(strategy="median")),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler()),
                        ])
cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),
                         ('cat_encoder', OneHotEncoder())])

# Can join these two pipelines with Scikit's FeatureUnion class
# Give it a list of transformers(can be entire transformer pipelines); when its transform()
# is called, runs each transformer's transform() method in parallel, waits for the output,
# then concatenates them and returns the result.

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),
                                                ("cat_pipeline", cat_pipeline),
                                                ])

housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared)


"""Building the Model: Linear Regression Model"""
from sklearn.linear_model import LinearRegression

# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)

"""Testing the model"""
# some_data = housing.iloc[:5]
# some_labels = housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)
# print("Predictions: ", lin_reg.predict(some_data_prepared))
# print("Labels: ", list(some_labels))

"""Measuring Error with RMSE"""
from sklearn.metrics import mean_squared_error
# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse) # Housing prices range from 120k to 265k, so being off by 68k is not good
# Example of underfitting data or features aren't good enough to make accurate predictions
# To fix this, should either choose a more powerful model, choose better features, or reduce constraints on model
# Model is not regularized so last option does not apply.

"""Trying Decision Trees"""
from sklearn.tree import DecisionTreeRegressor
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_rmse = np.sqrt(tree_mse)
# print(tree_rmse)  # Even though error is 0, this is probbaly case of overfitting

"""Using cross Validation"""
# Could use train_test_split function; however Scikit has cross-validation feature.
"""Scikit's cross validation, splits training set into 10 distinct subsets called folds.
It then trains and evaulates the Decision Tree model 10 times, picking
a different fold for every evaluation and training on the other 9 folds.
The result is an array containing the 10 eval scores"""

from sklearn.model_selection import cross_val_score
# scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                       #   scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-scores) # Expects utility function (greater is better)
# and opp of mse (neg value) so we do -scores

def print_scores(scores):
    print("Scores: ", scores)
    print("Mean ", scores.mean())
    print("Standard Deviation: ", scores.std())


# print_scores(tree_rmse_scores) # Cross validation allows us to get an estimate of performance of model
# and how precise the predictions are (standard deviation)
# Model score is about 71127 +- 2832.305. Would not have this data if used one validation set
# But cross validation makes us train model multiple times

"""Comparing to Lin_Reg Model"""

"""Trying RandomForests"""
from sklearn.ensemble import RandomForestRegressor

# forest = RandomForestRegressor()
# forest.fit(housing_prepared, housing_labels)
# housing_predictions = forest.predict(housing_prepared)

# forest_mse = mean_squared_error(housing_labels, housing_predictions)
# forest_rmse = np.sqrt(forest_mse)
# print(forest_rmse)

# scores = cross_val_score(forest, housing_prepared, housing_labels,
                        # scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-scores) # Expects utility function (greater is better)
# and opp of mse (neg value) so we do -scores
# print_scores(forest_rmse_scores)

"""Saving Model for later"""
from sklearn.externals import joblib

# joblib.dump(forest, "my_model.pkl")


"""Fine Tuning the Model"""
"""We could choose to optimize hyperparameters ourselves, but this is very tedious.
Scikit has built in optimizer called GridSearchCV. 
Need to tell which hyperparameter and which values to test."""
#  This code searches for the best combination for the random forest

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
              {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
              ]
# When do not know what hyper parameter values to try out, should try out consecutive powers of 10
"""param_grid tells scikit to evaluate all 3 X 4 = 12 combinations of
n_estimators and max_features values specified in the first dict, then
it tries all 2 X 3 = 6 combinations in the second dict, with bootstrap set to false.
It is initially set to true. Grid search will evaluate 18 combinations, and will train each 
model 5 times (cv = 5 below). 18 x 5 = 90 rounds of training."""


forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')

# random_search = RandomizedSearchCV(forest_reg, param_distributions=param_grid, cv=5,
#                                   scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)
# random_search.fit(housing_prepared, housing_labels)

"""Best params are stored in an instance var. Should reattempt with diff values, if there is a correlation
(i.e. max values are the most promising)."""
# print(grid_search.best_params_)
# print(random_search.best_params_)

"""Getting best estimator directly. NOTE: LEARN ABOUT GRID SEARCH API"""
# print(grid_search.best_estimator_)
# print(random_search.best_estimator_)
# If GridSearchCV is initialized with refit=True (which is default) once
# best estimator is found using cross validation, retrains that model on the whole training set.
# Good idea because may improve performance.
# print(grid_search.cv_results_) # Evaluation results
# Grid search can help identify whether we want to add a feature based on performance
# May even help better understand how to handle outliers

"""Randomized Grid Search"""
# If "search space" is large, use RandomizedSearchCV instead.
# Instead of trying out all combinations, evaluates a given number of random
# combinations by selecting random value for each hyper-parameter at each iteration.
# PROS: 1. If iterations is set to 1000, will try out 1000 different combinations of hyper-parameters
# 2. Mor control over computing budget by setting number of iterations

"""Ensemble methods"""
# Combining models may yield better performance i.e RandomForest from the individual decision trees

"""Analyze the best models and their error"""
# Inspecting good models gives insight
# EX: RandomForest can indicate the relative importance of each attribute
feature_importances = grid_search.best_estimator_.feature_importances
print(feature_importances)
# Format the importances to make easier to view and understand
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = cat_pipeline.named_steps["cat_encoder"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
# sorted(zip(feature_importances, attributes), reverse=True) # Try dropping less useful features

"""Eval on test set"""
# final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

# final_predictions = final_model.predict(X_test_prepared)

# final_mse = mean_squared_error(y_test, final_predictions)
# print(final_rmse = np.sqrt(final_mse))
# Model may perform worse on test set, since hyperparameters were tuned for test set
# Resist urge to tweak hyperparameters on test set because we won't be generalizing new data well

# After deploying, must monitor model (automated). Retrain (automated) with new data every once in a while


