import os
import numpy as np

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump,load
import argparse
from utils import *

def test_model_writing():
    digits = datasets.load_digits()
    split_train = 0.6
    split_val = 0.5
    X_train, y_train, X_validation, y_validation, X_test, y_test = preprocess_split_data(digits,split_train,split_val)
    expected_model_file_path = "C:/Users/user/PycharmProjects/mlops/model_stored.joblib"
    gamma = 0.01
    metrics_valid = run_classification_experiement(X_train, y_train, X_validation, y_validation, gamma,
                                                   expected_model_file_path, skip_dummy=True)
    assert os.path.isfile(expected_model_file_path)

def test_small_data_overfit_checking():
    digits = datasets.load_digits()
    split_train = 0.4
    split_val = 0.5
    X_train, y_train, X_validation, y_validation, X_test, y_test = preprocess_split_data(digits, split_train, split_val)
    expected_model_file_path = "E:/Gits/recog_mnist/mlops2/model_stored.joblib"
    gamma = 0.01
    metrics_train = run_classification_experiement(X_train, y_train, X_train, y_train, gamma,
                                                   expected_model_file_path, skip_dummy=True)
    assert metrics_train['accuracy'] > 0.8
    assert metrics_train['f1'] > 0.8
