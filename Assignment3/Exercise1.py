import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

# Function that takes an array of test points and returns the classification error
def testing(Xtest):

    # Label encoder to replace strings into integers
    le.fit(Xtest[:, 1])
    testX1 = le.transform(Xtest[:, 1])

    le.fit(Xtest[:, 2])
    testX2 = le.transform(Xtest[:, 2])

    le.fit(Xtest[:, 3])
    testX3 = le.transform(Xtest[:, 3])

    Xtest[:, 1] = testX1
    Xtest[:, 2] = testX2
    Xtest[:, 3] = testX3

    # Predict and calculate accuracy
    yPred = clf.predict(Xtest)
    accuracy = 100 - np.around((np.mean(yPred != yTest) * 100), 2)
    print("Accuracy: " + str(accuracy) + "%")

# Load the data from the csv file
df = pd.read_csv('trainingDecisionTree.csv')
data = df.values

# 400 000 for training set
X = data[:400000,0:41]
y = data[:400000,41]

# 100 000 for test set
Xtest = data[400000:,0:41]
yTest = data[400000:,41]

# Label encoder to replace strings to integers
le = preprocessing.LabelEncoder()
le.fit(X[:,1])
X1 = le.transform(X[:,1])

le.fit(X[:,2])
X2 = le.transform(X[:,2])

le.fit(X[:,3])
X3 = le.transform(X[:,3])

X[:,1] = X1
X[:,2] = X2
X[:,3] = X3

# Creating and training the decision tree classifier
clf = DecisionTreeClassifier(max_depth=5,max_features=5)
clf.fit(X,y)

# Grid Search to find the best possible parameters
#param_grid = {'max_depth': range(1,5,1), 'max_features': range(1,5,1)}
#clf_tree = DecisionTreeClassifier()
#clf_grid = GridSearchCV(clf_tree, param_grid)
#clf_grid.fit(X,y)
#print("Best Parameters:\n", clf_grid.best_params_)
#print("Best Estimators:\n", clf_grid.best_estimator_)

# Calling test function to test the test set
testing(Xtest)

