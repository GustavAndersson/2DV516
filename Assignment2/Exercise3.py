import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

def normalize(x):
    mean = np.mean(x)
    stdev = np.std(x)
    normal = (x - mean)/stdev
    return normal

def sigmoid(x):
    return (np.e ** x) / ((np.e ** x) + 1)

def cost(n, y, x, beta):
    return (-1/n) * (np.dot(y.T, np.log(sigmoid(np.dot(x,beta))))+np.dot((1-y).T, np.log(1-sigmoid(np.dot(x,beta)))))

def gradientDescent(alpha, iteration, X, y):
    beta = np.array([0,0,0,0,0,0,0,0,0,0])
    n = len(X)
    gd = []
    for i in range(iteration):
        beta = beta - (np.dot(((alpha/n) * X.T), ((sigmoid(np.dot(X,beta))) - y)))
        new_cost = cost(n, y, X, beta)
        gd.append([i,new_cost])
    return gd, beta

def main(dataSet, type):
    # Task 3
    X0 = normalize(dataSet[:, 0])
    X1 = normalize(dataSet[:, 1])
    X2 = normalize(dataSet[:, 2])
    X3 = normalize(dataSet[:, 3])
    X4 = normalize(dataSet[:, 4])
    X5 = normalize(dataSet[:, 5])
    X6 = normalize(dataSet[:, 6])
    X7 = normalize(dataSet[:, 7])
    X8 = normalize(dataSet[:, 8])

    n = len(dataSet)
    dataSetEx = np.c_[np.ones((n, 1)), X0, X1, X2, X3, X4, X5, X6, X7, X8]

    alpha = 0.01
    iterations = 5000
    print('α = ' + str(alpha) + ', N = ' + str(iterations))
    gd, beta = gradientDescent(alpha, iterations, dataSetEx, dataSet[:, 9])
    gd_arr = np.asarray(gd)

    plt.figure(1)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')

    if type == 1:
        plt.plot(gd_arr[:, 0], gd_arr[:, 1], label='Training data')
    elif type == 2:
        plt.plot(gd_arr[:, 0], gd_arr[:, 1], label='Test data')
    plt.legend()

    # Task 4
    p = sigmoid(np.dot(dataSetEx, beta))
    pred = np.round(p)
    dataError = np.round((np.mean(pred != dataSet[:, 9]) * 100), 2)
    dataAccuracy = 100 - dataError

    print("Number of non-correct classifications: " + str(
        int(np.around(len(dataSet[:, 9]) * np.mean(pred != dataSet[:, 9])))))
    print("Training accuracy: " + str(dataAccuracy) + "%")

# Task 1 - Read data and shuffle the rows in the raw data matrix
my_data = np.genfromtxt('breast_cancer.csv', delimiter=',')
np.random.shuffle(my_data)

# Task 2 -  How many observations did you allocated for testing, and why this number?
#           Answer: I chose to allocate 20% (137) of the observations for testing.
#           This because the training data needs to have a lot of samples to get a more accurate model.
#           So 80% (546) for training set and 20% (137) for test set
for i in my_data:
    if i[9] == 2:
        i[9] = 0
    elif i[9] == 4:
        i[9] = 1

trainingSet = np.array(my_data[:546])
testSet = np.array(my_data[546:])

print("Training data:")
main(trainingSet, 1)

# Task 5 - test error and test accuracy can be seen in the printout
print("\nTest data:")
main(testSet, 2)

plt.show()

# Task 6
# Repeated runs will result in a very small difference. The training set will always be within 96-98% in accuracy.
# The test set will always be within 97-100% in accuracy.
# Depending on how you divide it, the amount of errors will change but the accuracy will be approximately the same.