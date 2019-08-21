from sklearn import datasets
from sklearn import svm
#from joblib import dump, load
import pickle

clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
digits = datasets.load_digits()

X, y = iris.data, iris.target
clf.fit(X, y).predict(X)

svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

# s = pickle.dumps(clf)
# clf2 = pickle.load(s)
print(clf.predict(X[0:8]))

# dump(clf, 'filename.joblib')
# clf = load('filename.joblib')



# Questions:

# What is gamma and C?
# What are the parameters of SVC? What is SVC?
# When to worry about type casting data sets?
# What are hyper parameters used for? What is Kernel (rbf vs linear)

# Explanation on multiclass vs multilabel fitting. What is labelbinarizer?
