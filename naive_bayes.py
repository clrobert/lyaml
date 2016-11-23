from udacity.choose_your_own.class_vis import prettyPicture
from udacity.choose_your_own.prep_terrain_data import makeTerrainData
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pprint
pp = pprint.PrettyPrinter(indent=4)

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "predicting time time:", round(time()-t1, 3), "s"

accuracy = accuracy_score(labels_test, pred)
print "accuracy: ", accuracy

print(clf.score(features_test, labels_test))
