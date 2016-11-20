import sys
from udacity.choose_your_own.class_vis import prettyPicture
from udacity.choose_your_own.prep_terrain_data import makeTerrainData
import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl
from sklearn.svm import SVC

features_train, labels_train, features_test, labels_test = makeTerrainData()

########################## SVM #################################
### we handle the import statement and SVC creation for you here
clf = SVC(kernel="linear")


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data

#### store your predictions in a list named pred

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

def submitAccuracy():
    return acc
