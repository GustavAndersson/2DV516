import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.datasets.base import get_data_home
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# Downloading the MNIST dataset
mnist = fetch_openml('mnist_784', data_home='data')

data = np.asarray(mnist.data)
target = np.asarray(mnist.target)

# Training data
X = data[:30000,:]
y = target[:30000]

# Test data
Xtest = data[60000:,:]
yTest = target[60000:]

# Create and train svm
clf = svm.SVC(gamma='scale', kernel='rbf', C=10)
clf = clf.fit(X,y)

# Predict y and calculate accuracy
yPred = clf.predict(Xtest)
accuracy = 100 - np.around((np.mean(yPred !=yTest)*100),2)
print("Accuracy: "+str(accuracy) +"%")

# Grid Search to find the best possible parameters
#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}
#clf_grid = GridSearchCV(svm.SVC(kernel='rbf'), param_grid)
#clf_grid.fit(X,y)
#print("Best Parameters:\n", clf_grid.best_params_)
#print("Best Estimators:\n", clf_grid.best_estimator_)
