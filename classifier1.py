from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import random
from scipy.spatial import distance

# function t calculate the euclidean distance
def euc(a, b):
    return distance.euclidean(a, b)

# classifier form scratch
class TrainKnn():
    # fit function
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    # predict function
    def predict(self, X_test):
        # return list
        prediction = []
        for row in X_test:
            label = self.closest(row)
            prediction.append(label)
        return prediction

    # closest function to calculate the nearest neighbour
    def closest(self, row):
        best_distance = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            new_dist = euc(row, self.X_train[i])
            if best_distance > new_dist:
                best_distance = new_dist
                best_index = i
        return self.Y_train[best_index]



# iris dataset offered by the sklearn
iris = datasets.load_iris()

# x and y
X = iris.data
Y = iris.target

# splits the train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)

# Choice of classifier
classifier = TrainKnn()

# fit the training data to produce the features
classifier.fit(X_train, Y_train)

# predict using the features
predictions  = classifier.predict(X_test)

# test the accuracy
print(accuracy_score(Y_test, predictions))