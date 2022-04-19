#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
# clf = SVC(kernel='linear')
clf = SVC(kernel = 'rbf',C=10000)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print( "accuracy:", accuracy_score(labels_test, pred))

#########################################################

print ("Predictions:")
print ("10:", pred[10])
print ("26:", pred[26])
print ("50:", pred[50])

c = Counter(pred)
print ("No of predictions for Chris(1):", c[1])

#########################################################
