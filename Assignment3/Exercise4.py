import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Generated training set
X = np.asarray([[0,0], [0,1], [1,0], [1,1]])
y = np.asarray([0, 1, 1, 0])
print(X)
print(y)

# Feed forward neural network with one hidden layer containing two sigmoid neurons
clf = MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,))
clf.fit(X,y)

# Printing weights and biases
print(clf.coefs_)
print(clf.intercepts_)

# Predict and calculate accuracy
yPred = clf.predict(X)
print('Accuracy: ' + str(clf.score(X,y)*100) + '%')
