#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
from udacity.choose_your_own.class_vis import prettyPicture, output_image
from udacity.choose_your_own.prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl

from sklearn import tree
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()

def classify(features_train, labels_train):
    clf = tree.DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    return clf

clf = classify(features_train, labels_train)
pred = clf.predict(features_test)

acc = accuracy_score(labels_test, pred)

#prettyPicture(clf, features_test, labels_test)
#output_image("test.png", "png", open("test.png", "rb").read())

