import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

def cost(Xe, y, beta):
    n = len(Xe)
    j = np.dot(Xe,beta)-y
    J = (j.T.dot(j))/n
    return J

my_data = np.genfromtxt('GPUbenchmark.csv', delimiter=',')

y = my_data[:, 6]
X = my_data[:, [0, 1, 2, 3, 4, 5]]

linreg = LinearRegression()

it = [0, 1, 2, 3, 4, 5]; # keeping track of which features to try
best = [];
for i in range(6):
    scores = [];
    for j in it:
        tmpX = X[:, best + [j]]
        linreg.fit(tmpX, y)
        mdl = linreg.predict(tmpX)#train a linear regression model using tmpX and y;
        beta = linreg.coef_
        scores.append(cost(tmpX, mdl, beta))

    m = it[np.argmin(scores)]  # finds the index of the smallest cost
    best.append(m)
    it = np.setdiff1d(it,m) # removes m from it since it is already used
print('Best: ' + str(best))

x_best = np.array([X[:, best[0]], X[:, best[1]], X[:, best[2]], X[:, best[3]], X[:, best[4]], X[:, best[5]]]).T

features = [];
for i in range(6):
    plt.subplot(2,3,i+1)
    tmpX = x_best[:, features + [i]]
    linreg.fit(tmpX, y)
    prd = linreg.predict(tmpX)
    mse = (np.square(prd-y)).mean()
    plt.title("Features: " + str(i+1) + ", MSE: " + str(np.around(mse, 1)))
    plt.plot(y, "ro", markersize=3)
    plt.plot(prd, "black")
    features.append(i)
plt.show()

#   Using the idea of Occam’s razor, which would be the best choice of model?
#   Answer: Feature 4 is not the lowest but isn't far away from feature 5 and 6 in error.
#           According to occam's razor the best choice is the simpler, and in this case
#           that would be the model with 4 features.