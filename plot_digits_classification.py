"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump,load
import argparse



digits = datasets.load_digits()
data=digits.images
n_samples = len(data)  
data = data.reshape((n_samples, -1))




 
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.4, shuffle=False)
X_validation, X_test, y_validation, y_test = train_test_split(
  X_test, y_test, test_size=0.5, shuffle=False)


def train_with_gamma(gamma,X_train,y_train):
  clf = svm.SVC(gamma=gamma)
  clf.fit(X_train, y_train)
  return clf

def evaluate(clf,X,y):
  predicted = clf.predict(X)
  accuracy = metrics.accuracy_score(y, predicted)
  f1_score = metrics.f1_score(y, predicted,average="macro")
  return accuracy,f1_score




gammas=[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2,0.5,0.8,1,2,5,10]
accuracies=[]
f1_scores = []
optimum_accuracy = 0
optimal_gamma = 0
#optimal_classifier = None
parser = argparse.ArgumentParser(description='Perser')
parser.add_argument('--file_path', default='/content/digit_recog_model', type=str)
the_args = parser.parse_args()
for g in gammas:
  classifier = train_with_gamma(g,X_train,y_train)
  accuracy,f1_score = evaluate(classifier,X_validation ,y_validation)
  accuracies.append(accuracy)
  f1_scores.append(f1_score)
  print(f"For gamma {g} accuracy is {accuracy:.4f} and F1 score is {f1_score:.4f}",)
  if accuracy > optimum_accuracy:
    optimum_accuracy = accuracy
    optimal_gamma = g
    dump(classifier,the_args.file_path)
    #optimal_classifier = classifier


print(f"Optimal Gamma is {optimal_gamma}")
optimal_classifier = load(the_args.file_path)
accuracy,f1_score = evaluate(optimal_classifier,X_test ,y_test)
print(f"Accuracy and F1 score on test set with optimal Gamma are {accuracy} and {f1_score}")