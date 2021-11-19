import os
import numpy as np

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump,load
import argparse
#from utils2 import *
digits = datasets.load_digits()
data = digits.images
n_samples = len(data)
data = data.reshape((n_samples, -1))
targets = digits.target
print(targets)
testing_data = [0]*10
for i in range(10):
    for j in range(len(targets)):
        if targets[j] == i:
            testing_data[i] = data[j]

svm_clf = load('model_stored_dtree1.joblib')
tree_clf = load('model_stored_svm1.joblib')


def test_digit_correct_0():
    assert svm_clf.predict([testing_data[0]])[0] == 0
def test_digit_correct_1():
    assert svm_clf.predict([testing_data[1]])[0] == 1

def test_digit_correct_2():
    assert svm_clf.predict([testing_data[2]])[0] == 2

def test_digit_correct_3():
    assert svm_clf.predict([testing_data[3]])[0] == 3

def test_digit_correct_4():
    assert svm_clf.predict([testing_data[4]])[0] == 4

def test_digit_correct_5():
    assert svm_clf.predict([testing_data[5]])[0] == 5

def test_digit_correct_6():
    assert svm_clf.predict([testing_data[6]])[0] == 6

def test_digit_correct_7():
    assert svm_clf.predict([testing_data[7]])[0] == 7

def test_digit_correct_8():
    assert svm_clf.predict([testing_data[8]])[0] == 8

def test_digit_correct_9():
    assert svm_clf.predict([testing_data[9]])[0] == 9


def test_digit_correct_0_tree():
    assert tree_clf.predict([testing_data[0]])[0] == 0


def test_digit_correct_1_tree():
    assert tree_clf.predict([testing_data[1]])[0] == 1


def test_digit_correct_2_tree():
    assert tree_clf.predict([testing_data[2]])[0] == 2


def test_digit_correct_3_tree():
    assert tree_clf.predict([testing_data[3]])[0] == 3


def test_digit_correct_4_tree():
    assert tree_clf.predict([testing_data[4]])[0] == 4


def test_digit_correct_5_tree():
    assert tree_clf.predict([testing_data[5]])[0] == 5


def test_digit_correct_6_tree():
    assert tree_clf.predict([testing_data[6]])[0] == 6


def test_digit_correct_7_tree():
    assert tree_clf.predict([testing_data[7]])[0] == 7


def test_digit_correct_8_tree():
    assert tree_clf.predict([testing_data[8]])[0] == 8


def test_digit_correct_9_tree():
    assert tree_clf.predict([testing_data[9]])[0] == 9


def test_svm_model_bised():
    assert svm_clf.predict([testing_data[0]])[0] != svm_clf.predict([testing_data[1]])[0]

def test_tree_model_bised():
    assert tree_clf.predict([testing_data[0]])[0] != tree_clf.predict([testing_data[1]])[0]






