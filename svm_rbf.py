#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""
from udacity import *
import sys
from time import time
sys.path.append("/udacity/tools/")
from udacity.tools.email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

clf = SVC(C=10000.0, kernel="rbf")

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "predicting time time:", round(time()-t1, 3), "s"

print "Elements: ", pred[10], pred[26], pred[50]

print len(pred)

expected_chris = [ x for x in pred if x == 1 ]

print len(expected_chris)

acc = accuracy_score(pred, labels_test)
print acc
