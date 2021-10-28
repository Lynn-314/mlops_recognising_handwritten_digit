

print(__doc__)


import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from joblib import dump,load
import argparse
import numpy as np









def preprocess_split_data(digits,split_train,split_val):
  data=digits.images
  n_samples = len(data)  
  data = data.reshape((n_samples, -1))
  X_train, X_test, y_train, y_test = train_test_split(
      data, digits.target, test_size=split_train, shuffle=True)
  X_validation, X_test, y_validation, y_test = train_test_split(
      X_test, y_test, test_size=split_val, shuffle=True)
  return X_train, y_train, X_validation, y_validation, X_test, y_test

def train_with_gamma(gamma,X_train,y_train):
  clf = svm.SVC(gamma=gamma)
  clf.fit(X_train, y_train)
  return clf

def train_decision_tree(d,X_train,y_train):
  clf = DecisionTreeClassifier(random_state=0, max_depth=d)
  clf.fit(X_train, y_train)
  return clf

def evaluate(clf,X,y):
  predicted = clf.predict(X)
  accuracy = metrics.accuracy_score(y, predicted)
  f1_score = metrics.f1_score(y, predicted,average="macro")
  return accuracy,f1_score



def store_classifier(X_train,y_train,X_validation ,y_validation,file_path,type_model):
  if type_model == "svm":
    gammas=[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2,0.5,0.8,1,2,5,10]
    accuracies=[]
    f1_scores = []
    optimum_accuracy = 0.3 ### minimium accuracy is kept as 30% to throw away the random-like behaviour
    optimal_gamma = 0
    for g in gammas:
      classifier = train_with_gamma(g,X_train,y_train)
      accuracy,f1_score = evaluate(classifier,X_validation ,y_validation)
      accuracies.append(accuracy)
      f1_scores.append(f1_score)
      if accuracy > optimum_accuracy:
        optimum_accuracy = accuracy
        optimal_gamma = g
        dump(classifier,file_path) ###model saved in hard disk, only trained once for even optimal gamma
    return optimal_gamma
  elif type_model == "tree":
    max_deapth=[2,5,10,15,20,30,50,100]
    accuracies=[]
    f1_scores = []
    optimum_accuracy = 0.3 ### minimium accuracy is kept as 30% to throw away the random-like behaviour
    optimal_depth = 0
    for d in max_deapth:
      classifier = train_decision_tree(d,X_train,y_train)
      accuracy,f1_score = evaluate(classifier,X_validation ,y_validation)
      accuracies.append(accuracy)
      f1_scores.append(f1_score)
      if accuracy > optimum_accuracy:
        optimum_accuracy = accuracy
        optimal_depth = d
        dump(classifier,file_path) ###model saved in hard disk, only trained once for even optimal gamma
    return optimal_depth

def main():
  digits = datasets.load_digits()
  parser = argparse.ArgumentParser(description='Perser') ###arguments taken from input instead of heardcoded values
  parser.add_argument('--file_path', default='/content/digit_recog_model', type=str)
  parser.add_argument('--train_size', default=0.6, type=float)
  parser.add_argument('--test_size', default=0.2, type=float)
  arguments = parser.parse_args()
  split_train = (1-arguments.train_size)
  split_val = arguments.test_size/split_train
  tree_accuracies = []
  tree_F1 = []
  svm_accuracies = []
  svm_F1 = []
  print("Split    Tree accuracy   Tree F1 Value   SVM accuracy   SVM F1 Value ")
  for i in range(5):
    X_train, y_train, X_validation, y_validation, X_test, y_test = preprocess_split_data(digits,split_train,split_val)
    optimal_depth = store_classifier(X_train,y_train,X_validation ,y_validation,arguments.file_path,"tree")
    optimal_classifier = load(arguments.file_path)
    accuracy_tree,f1_score_tree = evaluate(optimal_classifier,X_test ,y_test)
    tree_accuracies.append(accuracy_tree)
    tree_F1.append(f1_score_tree)
    

    optimal_gamma = store_classifier(X_train,y_train,X_validation ,y_validation,arguments.file_path,"svm")
    optimal_classifier = load(arguments.file_path)
    accuracy_svm,f1_score_svm = evaluate(optimal_classifier,X_test ,y_test)
    svm_accuracies.append(accuracy_svm)
    svm_F1.append(f1_score_svm)
    print("{}          {:.4f}           {:.4f}          {:.4f}          {:.4f}".format(i,accuracy_tree,f1_score_tree,accuracy_svm,f1_score_svm))

  metrices = [tree_accuracies,tree_F1,svm_accuracies,svm_F1]
  mean_of_metrices = [sum(el)/len(el) for el in metrices]
  std_dev_of_metrices = [np.std(el) for el in metrices]
  print("\n\n")
  print("Tree accuracy : mean = {:.4f}, standard deviation = {:.4f}".format(mean_of_metrices[0],std_dev_of_metrices[0]))
  print("Tree F1 score : mean = {:.4f}, standard deviation = {:.4f}".format(mean_of_metrices[1],std_dev_of_metrices[1]))
  print("SVM accuracy : mean = {:.4f}, standard deviation = {:.4f}".format(mean_of_metrices[2],std_dev_of_metrices[2]))
  print("SVM accuracy : mean = {:.4f}, standard deviation = {:.4f}".format(mean_of_metrices[3],std_dev_of_metrices[3]))


if __name__ == "__main__":
  main()