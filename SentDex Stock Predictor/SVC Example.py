import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib import style
style.use("ggplot")

x = [1, 5, 3, 8, 9, 10]
y = [2, 8, 4, 11, 2, 9]

plt.scatter(x, y)
#plt.show()

X = np.array([[1,2],[5,8], [3,4], [8,11], [9,2], [10,9]])

y = [0, 0, 1, 1, 0, 1]
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

#print(clf.predict([[10.58, 10.76]]))

"""Coefficient for graphing line"""
w = clf.coef_[0]
#print(w)

a =-w[0]/ w[1]
xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weight div")

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.legend()
plt.show()