import os
import numpy as np

from joblib import dump, load
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from numpy.lib.function_base import average


def preprocess_split_data(digits,train_split,val_split,test_split):
  data=digits["data"]
  n_samples = len(data)
  data = data.reshape((n_samples, -1))
  split_train = np.round(1 - train_split/100,2)
  split_val = test_split/(val_split + test_split)
  X_train, X_test, y_train, y_test = train_test_split(
      data, digits.target, test_size=split_train, shuffle=False)
  X_validation, X_test, y_validation, y_test = train_test_split(
      X_test, y_test, test_size=split_val, shuffle=False)
  return X_train, y_train, X_validation, y_validation, X_test, y_test


def eval_model_from_path(best_model_path,x,y):
    clf=load(best_model_path)
    metrics=test(clf,x,y)
    return metrics

def eval(classifier,x,y):
    predicted = classifier.predict(x)
    acc = metrics.accuracy_score(y_pred=predicted,y_true=y)
    f1 = metrics.f1_score(y_pred=predicted,y_true=y,average="macro")
    return {"accuracy": acc, "f1": f1}

def get_random_acc(y):
    return max(np.bincount(y))/len(y)

def run_classification_experiement(x_train,y_train,x_valid,y_valid,gamma,output_model_file,skip_random=True):
    random_val_acc=get_random_acc(y_valid)
    clf = svm.SVC(gamma=gamma)
    clf.fit(x_train,y_train)
    metrics_valid = eval(clf,x_valid,y_valid)
    if skip_random and metrics_valid["accuracy"] < random_val_acc:
        return None
    output_folder=os.path.dirname(output_model_file)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    dump(clf,output_model_file)
    return metrics_valid
def run_classification_experiement(x_train,y_train,x_valid,y_valid,gamma,output_model_file,skip_random=True):
    random_val_acc=get_random_acc(y_valid)
    clf = svm.SVC(gamma=gamma)
    clf.fit(x_train,y_train)
    metrics_valid = eval(clf,x_valid,y_valid)
    if skip_random and metrics_valid["accuracy"] < random_val_acc:
        return None
    output_folder=os.path.dirname(output_model_file)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    dump(clf,output_model_file)
    return metrics_valid
