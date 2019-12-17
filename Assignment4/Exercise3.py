# Please read the attached PDF document
# to read my conclusions regarding the result
#
# I would also like to recommend you to test my functions
# in the files 'Exercise1.py' and 'Exercise2.py'
# instead of here because it's easer to get an overall view there.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import datasets
from scipy.spatial.distance import pdist

# Function which returns the largest cluster to divide
def largest_cluster(indices):
    # Basically finds the most common cluster in the array and returns its corresponding number
    counts = np.bincount(indices)
    return np.argmax(counts)

# Bisecting k-Means function
def bkmeans(X,k,iter):
    minSSE = 0
    countClusters = 0
    largestCluster = 0
    finalCluster = np.zeros(X.shape[0], dtype=int)
    clsLabels = np.zeros(X.shape[0])

    # Run this until we have k clusters
    while (countClusters < k-1):

        # Divide the largest cluster into two smaller sub-clusters iter times
        for i in range(iter):
            kmeans = KMeans(n_clusters=2).fit(X[clsLabels == largestCluster])

            # Choose the best solution according to SSE
            tempSSE = kmeans.inertia_
            if (i == 0):
                minSSE = tempSSE
                subcluster_indices = kmeans.labels_

            if (minSSE > tempSSE):
                minSSE = tempSSE
                subcluster_indices = kmeans.labels_

        # Assign new colors to the sub-clusters because it cant be 0 and 1 since it already exists
        subcluster_indices = np.where(subcluster_indices == 1, countClusters + 1, subcluster_indices)
        subcluster_indices = np.where(subcluster_indices == 0, largestCluster, subcluster_indices)

        # Merge the two new sub-clusters into the main array
        index = 0
        for n, i in enumerate(finalCluster):
            if i == largestCluster:
                number = subcluster_indices[index]
                finalCluster[n] = number
                index += 1

        # Prepare values for next round
        countClusters += 1
        largestCluster = largest_cluster(finalCluster)
        clsLabels = finalCluster

    return finalCluster

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

# Dataset 1
plt.figure(1)
dataset1 = np.genfromtxt('seeds_dataset.csv', delimiter=',')

X1 = dataset1[1:100, :7]
y1 = dataset1[1:100, 7]

sammon1 = sammon(X1, 100, 0.0, 0.3)
plt.subplot(3,3,1)
plt.title('Dataset 1 - Sammon')
plt.scatter(sammon1[:, 0],sammon1[:, 1], c=y1, cmap='rainbow')

pca1 = PCA(n_components=2)
pca1.fit(X1)
result_pca1 = pca1.transform(X1)
plt.subplot(3,3,2)
plt.title('Dataset 1 - PCA')
plt.scatter(result_pca1[:, 0],result_pca1[:, 1], c=y1, cmap='rainbow')

tsne1 = TSNE(n_components=2)
result_tsne1 = tsne1.fit_transform(X1)
plt.subplot(3,3,3)
plt.title('Dataset 1 - t-SNE')
plt.scatter(result_tsne1[:, 0],result_tsne1[:, 1], c=y1, cmap='rainbow')

##############################################################################################################
# Dataset 2
dataset2 = datasets.load_wine(True)

X2 = dataset2[0]
y2 = dataset2[1]

X2 = X2[1:100]
y2 = y2[1:100]

sammon2 = sammon(X2, 100, 0.0, 0.3)
plt.subplot(3,3,4)
plt.title('Dataset 2 - Sammon')
plt.scatter(sammon2[:, 0],sammon2[:, 1], c=y2, cmap='rainbow')

pca2 = PCA(n_components=2)
pca2.fit(X2)
result_pca2 = pca2.transform(X2)
plt.subplot(3,3,5)
plt.title('Dataset 2 - PCA')
plt.scatter(result_pca2[:, 0],result_pca2[:, 1], c=y2, cmap='rainbow')

tsne2 = TSNE(n_components=2)
result_tsne2 = tsne2.fit_transform(X2)
plt.subplot(3,3,6)
plt.title('Dataset 2 - t-SNE')
plt.scatter(result_tsne2[:, 0],result_tsne2[:, 1], c=y2, cmap='rainbow')

######################################################################################
# Dataset 3
dataset3 = datasets.load_breast_cancer(True)

X3 = dataset3[0]
y3 = dataset3[1]

X3 = X3[1:100]
y3 = y3[1:100]

sammon3 = sammon(X3, 100, 0.0, 0.3)
plt.subplot(3,3,7)
plt.title('Dataset 3 - Sammon')
plt.scatter(sammon3[:, 0],sammon3[:, 1], c=y3, cmap='rainbow')

pca3 = PCA(n_components=2)
pca3.fit(X3)
result_pca3 = pca3.transform(X3)
plt.subplot(3,3,8)
plt.title('Dataset 3 - PCA')
plt.scatter(result_pca3[:, 0],result_pca3[:, 1], c=y3, cmap='rainbow')

tsne3 = TSNE(n_components=2)
result_tsne3 = tsne3.fit_transform(X3)
plt.subplot(3,3,9)
plt.title('Dataset 3 - t-SNE')
plt.scatter(result_tsne3[:, 0],result_tsne3[:, 1], c=y3, cmap='rainbow')

######################################################################################
# Cluster techniques

plt.figure(2)

###### Dataset 1
bkmeans1 = bkmeans(result_tsne1,4,100)
plt.subplot(3,3,1)
plt.axis('off')
plt.title('Dataset 1 - BKMeans')
plt.scatter(result_tsne1[:, 0],result_tsne1[:, 1], c=bkmeans1, cmap='rainbow')

plt.subplot(3,3,2)
plt.axis('off')
plt.title('Dataset 1 - Classic KMeans')
kmeans1 = KMeans(n_clusters=4).fit(result_tsne1)
plt.scatter(result_tsne1[:, 0],result_tsne1[:, 1], c=kmeans1.labels_, cmap='rainbow')

plt.subplot(3,3,3)
plt.axis('off')
plt.title('Dataset 1 - Hierarchical')
agglo1 = AgglomerativeClustering(n_clusters=4).fit(result_tsne1)
plt.scatter(result_tsne1[:, 0],result_tsne1[:, 1], c=agglo1.labels_, cmap='rainbow')

###### Dataset 2
bkmeans2 = bkmeans(result_tsne2,4,100)
plt.subplot(3,3,4)
plt.axis('off')
plt.title('Dataset 2 - BKMeans')
plt.scatter(result_tsne2[:, 0],result_tsne2[:, 1], c=bkmeans2, cmap='rainbow')

plt.subplot(3,3,5)
plt.axis('off')
plt.title('Dataset 2 - Classic KMeans')
kmeans2 = KMeans(n_clusters=4).fit(result_tsne2)
plt.scatter(result_tsne2[:, 0],result_tsne2[:, 1], c=kmeans2.labels_, cmap='rainbow')

plt.subplot(3,3,6)
plt.axis('off')
plt.title('Dataset 2 - Hierarchical')
agglo2 = AgglomerativeClustering(n_clusters=4).fit(result_tsne2)
plt.scatter(result_tsne2[:, 0],result_tsne2[:, 1], c=agglo2.labels_, cmap='rainbow')

###### Dataset 3
bkmeans3 = bkmeans(result_tsne3,4,100)
plt.subplot(3,3,7)
plt.axis('off')
plt.title('Dataset 3 - BKMeans')
plt.scatter(result_tsne3[:, 0],result_tsne3[:, 1], c=bkmeans3, cmap='rainbow')

plt.subplot(3,3,8)
plt.axis('off')
plt.title('Dataset 3 - Classic KMeans')
kmeans3 = KMeans(n_clusters=4).fit(result_tsne3)
plt.scatter(result_tsne3[:, 0],result_tsne3[:, 1], c=kmeans3.labels_, cmap='rainbow')

plt.subplot(3,3,9)
plt.axis('off')
plt.title('Dataset 3 - Hierarchical')
agglo3 = AgglomerativeClustering(n_clusters=4).fit(result_tsne3)
plt.scatter(result_tsne3[:, 0],result_tsne3[:, 1], c=agglo3.labels_, cmap='rainbow')

plt.show()
