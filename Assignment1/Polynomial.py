import numpy as np
import matplotlib.pyplot as plt
import operator
from decimal import Decimal

# Euclidean distance
def euclideanDistance(x1, x2):
    distance = (pow((x1 - x2), 2))
    return np.sqrt(distance)

# K nearest neighbor function
def knn(dataSet, x_val, k):

    # Calculate distance for every x in the training set
    distances = {}
    for x in range(len(dataSet)):
        distances[x] = euclideanDistance(x_val, dataSet[x][0])

    # Sort distances in ascending order
    asc_distances = sorted(distances.items(), key=operator.itemgetter(1))

    # Fetch the knn
    neighbors = []
    for x in range(k):
        neighbors.append(asc_distances[x][0])

    # Calculate and return the average y-value of the knn
    y_sum = 0
    for x in range(len(neighbors)):
        y_sum = y_sum + dataSet[neighbors[x]][1]
    return y_sum/k


# Load data from the file
my_data = np.genfromtxt('polynomial200.csv', delimiter=',')

# Divide data into training- and test data
trainingData = np.array(my_data[:100])
testData = np.array(my_data[100:])

# Plotting the training- and test data
plt.figure(1)
plt.subplot(1, 2, 1)
plt.title("Training set")
plt.plot(trainingData[:, 0], trainingData[:, 1], 'bo', markersize=3)
plt.subplot(1, 2, 2)
plt.title("Test set")
plt.plot(testData[:, 0], testData[:, 1], 'ro', markersize=3)


# Start of k-NN regression
plt.figure(2)
figNr = 0
for k in [1,3,5,7]:

    figNr = figNr + 1

    # Dividing x-range into 100 intervals and find kNN for each one
    xy = []
    for x in np.arange(0, 25, 0.25):
        y = knn(trainingData, x, k)
        xy.append([x, y])
    regression = np.asarray(xy)

    # Calculate prediction for the training data
    predicted = []
    for x in trainingData:
        result = knn(trainingData, x[0], k)
        predicted.append([x[0], result])
    trainingPredicted = np.asarray(predicted)

    # Calculate prediction of the test data
    predicted2 = []
    for x in testData:
        result = knn(testData, x[0], k)
        predicted2.append([x[0], result])
    testPredicted = np.asarray(predicted2)

    # Calculate MSE for training- and test data
    mse_training = np.mean((trainingPredicted[:, 1]-trainingData[:, 1])**2)
    mse_test = np.mean((testPredicted[:, 1] - testData[:, 1]) ** 2)

    # Plot k-NN regression result and the MSE training error
    plt.subplot(2,2,figNr)
    plt.title("k=" + str(k) + ", MSE=" + str(round(mse_training, 2)))
    plt.plot(trainingData[:, 0], trainingData[:, 1], 'ro', markersize=3)
    plt.plot(regression[:, 0], regression[:, 1])

    # Presentation of the MSE test error
    print("MSE test error:")
    print("k=" + str(k) + ", MSE=" + str(round(mse_test, 2)))

plt.show()

