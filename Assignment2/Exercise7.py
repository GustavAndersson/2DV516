import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression

def sigmoid(x):
    return (np.e ** x) / ((np.e ** x) + 1)

def cost(n, y, x, beta):
    return (-1/n) * (np.dot(y.T, np.log(sigmoid(np.dot(x,beta))))+np.dot((1-y).T, np.log(1-sigmoid(np.dot(x,beta)))))

def gradientDescent(alpha, iteration, X, y):
    beta = np.array([0,0,0,0,0,0])
    n = len(X)
    gd = []
    for i in range(iteration):
        beta = beta - (np.dot(((alpha/n) * X.T), ((sigmoid(np.dot(X,beta))) - y)))
        new_cost = cost(n, y, X, beta)
        gd.append([i,new_cost])
    return gd, beta

def mapFeature(X1,X2,D, Ones): # Pyton
    one = np.ones([len(X1),1])
    if Ones:
        Xe = np.c_[one,X1,X2] # Start with [1,X1,X2]
    else:
        Xe = np.c_[X1, X2]  # Start with [X1,X2]
    for i in range(2,D+1):
        for j in range(0,i+1):
            Xnew = X1**(i-j)*X2**j # type (N)
            Xnew = Xnew.reshape(-1,1) # type (N,1) required by append
            Xe = np.append(Xe,Xnew,1) # axis = 1 ==> append column
    return Xe

def logreg_plot(X, y, i):
    # Setup mesh grid
    h = .05  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  # Mesh Grid
    x1, x2 = xx.ravel(), yy.ravel()  # Two Nx1 vectors

    #  predict each mesh point
    xy_mesh = mapFeature(x1,x2,i,False)
    classes = logreg.predict(xy_mesh)
    clz_mesh = classes.reshape(xx.shape)

    # Create mesh plot
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    plt.pcolormesh(xx, yy, clz_mesh, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='.', cmap=cmap_bold)

my_data = np.genfromtxt('microchips.csv', delimiter=',')

y = my_data[:, 2]
X = my_data[:, [0, 1]]

logreg = LogisticRegression(solver='lbfgs', penalty='l2', C=1)   # instantiate the model

plt.figure(1)
for i in range(9):
    plt.subplot(3,3,i+1)
    Xe = mapFeature(X[:, 0], X[:, 1], i+1,False)
    logreg.fit(Xe, y)
    y_pred = logreg.predict(Xe)
    errors = np.sum(y_pred != y)
    logreg_plot(X, y, i+1)
    plt.title("D = " + str(i+1) + ", Tra. errors = " + str(errors))
plt.show()

# Which of the models is best? Compare with your results from Exercise 4.
# Model 5 is the best because it has the lowest training errors which is 19.
# If you compare with exercise 4, you can see that each model here has more training errors.