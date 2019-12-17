import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from sklearn.linear_model import LinearRegression
import scipy.stats

def normalize(data, x):
    mean = np.mean(x)
    stdev = np.std(x)
    normal = (data - mean)/stdev
    return normal

my_data = np.genfromtxt('data_build_stories.csv', delimiter=',')

y = my_data[:, 1]
X = my_data[:, [0]]

# Plot data
plt.figure(1)
plt.plot(X, y, 'bo', markersize=3)

linreg = LinearRegression()
linreg.fit(X, y)
predY = linreg.predict(X)

# Plot predicted data
plt.plot(X, predY, 'r')
plt.vlines(X,y,predY)
plt.axhline(y=0)

beta = linreg.coef_

# Calculate standard error
se = np.sqrt((((predY - y)**2).sum())/((y.shape[0]-2)*((X-X.mean())**2).sum()))
print("Se: " + str(se))

# Calculate confidence interval
ci = [beta-2*se, beta+2*se]
print("CI: " + str(ci))

plt.show()
