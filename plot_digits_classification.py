

print(__doc__)


import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump,load
import argparse








def preprocess_split_data(digits,split_train,split_val):
  data=digits.images
  n_samples = len(data)  
  data = data.reshape((n_samples, -1))
  X_train, X_test, y_train, y_test = train_test_split(
      data, digits.target, test_size=split_train, shuffle=False)
  X_validation, X_test, y_validation, y_test = train_test_split(
      X_test, y_test, test_size=split_val, shuffle=False)
  return X_train, y_train, X_validation, y_validation, X_test, y_test



def train_with_gamma(gamma,X_train,y_train):
  clf = svm.SVC(gamma=gamma)
  clf.fit(X_train, y_train)
  return clf

def evaluate(clf,X,y):
  predicted = clf.predict(X)
  accuracy = metrics.accuracy_score(y, predicted)
  f1_score = metrics.f1_score(y, predicted,average="macro")
  return accuracy,f1_score



def store_classifier_with_optimal_gamma(X_train,y_train,X_validation ,y_validation,file_path):
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
    print(f"For gamma {g} accuracy is {accuracy:.4f} and F1 score is {f1_score:.4f}")
    if accuracy > optimum_accuracy:
      optimum_accuracy = accuracy
      optimal_gamma = g
      dump(classifier,file_path) ###model saved in hard disk, only trained once for even optimal gamma
  return optimal_gamma

def main():
  digits = datasets.load_digits()
  parser = argparse.ArgumentParser(description='Perser') ###arguments taken from input instead of heardcoded values
  parser.add_argument('--file_path', default='/content/digit_recog_model', type=str)
  parser.add_argument('--train_size', default=0.6, type=float)
  parser.add_argument('--test_size', default=0.2, type=float)
  arguments = parser.parse_args()
  split_train = (1-arguments.train_size)
  split_val = arguments.test_size/split_train
  X_train, y_train, X_validation, y_validation, X_test, y_test = preprocess_split_data(digits,split_train,split_val)
  optimal_gamma = store_classifier_with_optimal_gamma(X_train,y_train,X_validation ,y_validation,arguments.file_path)
  print(f"Optimal Gamma is {optimal_gamma}")
  optimal_classifier = load(arguments.file_path)
  accuracy,f1_score = evaluate(optimal_classifier,X_test ,y_test)
  print(f"Accuracy and F1 score on test set with optimal Gamma are {accuracy} and {f1_score}")

if __name__ == "__main__":
  main()