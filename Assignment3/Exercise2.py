import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

def normalize(x):
    mean = np.mean(x)
    stdev = np.std(x)
    normal = (x - mean)/stdev
    return normal

# Load the data from the csv file
my_data = np.genfromtxt('dataBM.csv', delimiter=',')

y = my_data[:, 2]
X = my_data[:, [0, 1]]

# Normalize the data
X1 = normalize(X[:, 0])
X2 = normalize(X[:, 1])
Xn = np.c_[X1, X2]

# Create and train svm
clf = svm.SVC(gamma='scale', kernel='rbf', C=10000)
clf = clf.fit(Xn,y)

# Print the accuracy
score = clf.score(Xn, y)
print(str(score*100) + '%')

# Calculate and plot the decision boundary
x_min, x_max = Xn[:, 0].min() - 1, Xn[:, 0].max() + 1
y_min, y_max = Xn[:, 1].min() - 1, Xn[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

# Plot the training points
for i, color in zip(range(2), 'rb'):
    id = np.where(y == i)
    plt.scatter(Xn[id, 0], Xn[id, 1], c=color, cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.show()