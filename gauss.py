# Using python 3

def classify(features_train, labels_train):
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB
    ### create classifier
    clf = GaussianNB()
    ### fit the classifier on the training features and labels
    clf.fit(features_train, labels_train)
    ### return the fit classifier
    return clf

    ### your code goes here!

from udacity.choose_your_own.class_vis import prettyPicture
from udacity.choose_your_own.prep_terrain_data import makeTerrainData
#from classify import NBAccuracy

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

import pprint
pp = pprint.PrettyPrinter(indent=4)

features_train, labels_train, features_test, labels_test = makeTerrainData()

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)


print(pred)

print(clf.score(features_test, labels_test))

#from sklearn.metrics import accuracy_score

accuracy = accuracy_score(labels_test, list(pred))

accuracy = accuracy_score(pred, features_test)

classified_correctly / all_points