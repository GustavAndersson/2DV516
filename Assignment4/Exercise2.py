import numpy as np
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

def sammon(X, iter, error, alpha):

    # Number of samples
    n_samples = len(X)

    # 1. Random two-dimensional layout Y
    y = np.random.normal(1, 5, [n_samples, 2])

    sum_delta_ij = 0
    sum_distance = 0
    c = 0

    # Calculates all the distances of the input space
    # Then calculates the sum of the input distances
    # c = the sum of all input distances
    distX = pdist(X, 'euclidean')
    sum_delta_ij = np.sum(distX)
    c = sum_delta_ij

    # Loop iter times
    for x in range(iter):

        # Calculates all the distances of the output space
        distY = pdist(y, 'euclidean')

        # Calculates the "second sum part" of the Sammons stress equation
        sum_distance = np.sum((((distX - distY)** 2) / distX))

        # 2. Compute the stress E of Y
        E = (1 / sum_delta_ij) * sum_distance

        # 3. If E < e --> stop
        if (E < error):
            return y

        # 4. For each yi of Y, find the next vector yi(t+1)
        else:
            partial1 = np.array([0,0])
            partial2 = np.array([0,0])
            y_next = np.zeros((n_samples, 2))

            for i in range(n_samples):
                for j in range(n_samples):
                    if (j != i):

                        # Differences needed further
                        X_diff = X[i] - X[j]
                        y_diff = y[i] - y[j]

                        delta_ij = np.sqrt(np.sum(np.square(X_diff)))
                        d_ij = np.sqrt(np.sum(np.square(y_diff)))
                        divergence = delta_ij - d_ij
                        denominator = d_ij * delta_ij

                        # Limits how small the denominator can be
                        if (denominator < 0.000001):
                            denominator = 0.1

                        # Calculates the partial equations
                        partial1 = partial1 + (divergence / denominator) * y_diff
                        partial2 = partial2 + (1 / denominator) * (divergence - (((y_diff ** 2) / d_ij) * (1 + (divergence / d_ij))))

                deltai_t = (((-2 / c) * partial1) / np.abs(((-2 / c) * partial2)))
                y_next[i] = y[i] - alpha * deltai_t

            y = np.copy(y_next)
    return y

X = make_blobs(n_samples=20, n_features=3)
y = X[1]
test = sammon(X[0], 100, 0.0, 0.3)

plt.figure(1)
plt.scatter(test[:, 0],test[:, 1], c=y, cmap='rainbow')
plt.show()