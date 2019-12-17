import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

# Function that plots the chosen polynomial model
def fit(X, beta, degree):
    if degree == 1:
        plt.plot(X, beta[0]+X*beta[1], 'r')
    elif degree == 2:
        plt.plot(X, beta[0]+X*beta[1] + (pow(X,2)*beta[2]), 'r')
    elif degree == 3:
        plt.plot(X, beta[0]+X*beta[1] + (pow(X,2)*beta[2]) + (pow(X,3)*beta[3]) , 'r')
    elif degree == 4:
        plt.plot(X, beta[0]+X*beta[1] + (pow(X,2)*beta[2]) + (pow(X,3)*beta[3]) + (pow(X,4)*beta[4]), 'r')

my_data = np.genfromtxt('housing_price_index.csv', delimiter=',')

# Task 1 - Plot the data in the matrix housing_price_index
plt.figure(1)
plt.plot(my_data[:, 0], my_data[:, 1], 'bo', markersize=3)

# Task 2 -  Which polynomial degree do you think gives the best fit?
#           Answer: I think the fourth degree gives the best fit
#                   because it follows the data smoothly in comparison
#                   to the other degrees.
X = my_data[:, [0]]
y = my_data[:, 1]

Xe = np.c_[np.ones((len(X),1)),X[:, 0],pow(X[:, 0],2),pow(X[:, 0],3),pow(X[:, 0],4)]
beta = np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)

# Degree 1
plt.subplot(2,2,1)
plt.plot(my_data[:, 0], my_data[:, 1], 'bo', markersize=3)
fit(X, beta, 1)

# Degree 2
plt.subplot(2,2,2)
plt.plot(my_data[:, 0], my_data[:, 1], 'bo', markersize=3)
fit(X, beta, 2)

# Degree 3
plt.subplot(2,2,3)
plt.plot(my_data[:, 0], my_data[:, 1], 'bo', markersize=3)
fit(X, beta, 3)

# Degree 4
plt.subplot(2,2,4)
plt.plot(my_data[:, 0], my_data[:, 1], 'bo', markersize=3)
fit(X, beta, 4)

# Task 3
# Answer:   The housing price index 2022 is 799.
#           Therefore he can expect to get 2.3 + 0.8 = 3.1 million SEK
#           for his house in 2022. I think it's realistic because in 2017 the index was
#           668 and it has increased fairly until 2022.
x = 2022 - 1975
print(beta[0] + beta[1]*x + beta[2]*(x**2) + beta[3]*(x**3) + beta[4]*(x**4))

plt.show()