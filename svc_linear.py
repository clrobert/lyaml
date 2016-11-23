import sys
from udacity.choose_your_own.class_vis import prettyPicture
from udacity.choose_your_own.prep_terrain_data import makeTerrainData
import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = SVC(kernel="linear")
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)

def submitAccuracy():
    return acc
