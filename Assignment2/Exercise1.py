import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

def normalize(data, x):
    mean = np.mean(x)
    stdev = np.std(x)
    normal = (data - mean)/stdev
    return normal

def cost(Xe, y, beta):
    j = np.dot(Xe,beta)-y
    J = (j.T.dot(j))/n
    return J

def gradientDescent(Xe, y, alpha):
    beta = np.array([0,0,0,0,0,0,0])
    gd = []
    iteration = 1000
    for i in range(iteration):
        beta = beta - (np.dot((alpha * Xe.T), ((np.dot(Xe,beta)) - y)))
        new_cost = cost(Xe, y, beta)
        gd.append([i,new_cost])
    return gd, beta

my_data = np.genfromtxt('GPUbenchmark.csv', delimiter=',')

y = my_data[:, 6]
X = my_data[:, [0, 1, 2, 3, 4, 5]]

cudaCores = np.array(X[:, 0])
baseClock = np.array(X[:, 1])
boostClock = np.array(X[:, 2])
memorySpeed = np.array(X[:, 3])
memoryConfig = np.array(X[:, 4])
memoryBandwidth = np.array(X[:, 5])

# Task 1 - Normalizing X
n_cudaCores = normalize(cudaCores, cudaCores)
n_baseClock = normalize(baseClock, baseClock)
n_boostClock = normalize(boostClock, boostClock)
n_memorySpeed = normalize(memorySpeed, memorySpeed)
n_memoryConfig = normalize(memoryConfig, memoryConfig)
n_memoryBandwidth = normalize(memoryBandwidth, memoryBandwidth)

Xn = np.c_[n_cudaCores, n_baseClock, n_boostClock, n_memorySpeed, n_memoryConfig, n_memoryBandwidth]

# Task 2 - Plot Xi vs y for each feature
plt.figure(1)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.plot(Xn[:, i], y, 'bo', markersize=3)

# Task 3 - Normal equation
n = len(my_data)
Xe = np.c_[np.ones((n,1)), Xn]
beta = np.linalg.inv(Xe.T.dot(Xe)).dot(Xe.T).dot(y)

# What is the predicted benchmark result for a graphic card
# with the following (non-normalized) feature values?
# Answer: 110.80403513783246
testData = [2432, 1607, 1683, 8, 8, 256]
test_cudaCores = normalize(testData[0], cudaCores)
test_baseClock = normalize(testData[1], baseClock)
test_boostClock = normalize(testData[2], boostClock)
test_memorySpeed = normalize(testData[3], memorySpeed)
test_memoryConfig = normalize(testData[4], memoryConfig)
test_memoryBandwidth = normalize(testData[5], memoryBandwidth)

n_testData = [1, test_cudaCores, test_baseClock, test_boostClock, test_memorySpeed, test_memoryConfig, test_memoryBandwidth]

predicted = np.dot(beta, n_testData)
print(predicted)

# Task 4 -  What is the cost J(β) when using the β computed by the normal equation above?
#           Answer: 12.396444360915686
cost_ = cost(Xe, y, beta)
print(cost_)

# Gradient descent - task 5
# (a)   What hyperparameters α, N are needed to get within 1% of the final cost for the
#       normal equation?
#       Answer: α = 0.01, N = 1000
gd, beta = gradientDescent(Xe, y, 0.01)
gd_arr = np.asarray(gd)
plt.figure(2)
plt.plot(gd_arr[:, 0], gd_arr[:, 1])

# (b)   What is the predicted benchmark result for the example graphic card presented above?
#       Answer: 111.71170956389668
predicted = np.dot(beta, n_testData)
print(predicted)

plt.show()