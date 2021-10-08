import os
import numpy as np

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump,load
import argparse
from utils import *

def test_split_creation_case1():
    digits = datasets.load_digits()
    num_samples = 100
    digits["data"] = digits["data"][0:num_samples]
    digits["target"] = digits["target"][0:num_samples]
    train_split = 70
    val_split = 20
    test_split = 10
    X_train, y_train, X_validation, y_validation, X_test, y_test = preprocess_split_data(digits,train_split,val_split,test_split)
    assert len(y_train) + len(y_validation) + len(y_test) == num_samples
    assert len(y_train) == np.round(num_samples*train_split/100,0)
    assert len(y_validation) == np.round(num_samples * val_split/100, 0)
    assert len(y_test) == np.round(num_samples * test_split/100, 0)


def test_split_creation_case2():
    digits = datasets.load_digits()
    num_samples = 9
    digits["data"] = digits["data"][0:num_samples]
    digits["target"] = digits["target"][0:num_samples]
    train_split = 70
    val_split = 20
    test_split = 10
    X_train, y_train, X_validation, y_validation, X_test, y_test = preprocess_split_data(digits,train_split,val_split,test_split)
    assert len(y_train) + len(y_validation) + len(y_test) == num_samples
    assert len(y_train) == np.round(num_samples*train_split/100,0)
    assert len(y_validation) == np.round(num_samples * val_split/100, 0)
    assert len(y_test) == np.round(num_samples * test_split/100, 0)



