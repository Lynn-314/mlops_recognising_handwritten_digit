import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tabulate import tabulate


def preprocess_split_data(digits,split_train,split_val):
  data=digits.images
  n_samples = len(data)  
  data = data.reshape((n_samples, -1))
  X_train, X_test, y_train, y_test = train_test_split(
      data, digits.target, test_size=split_train, shuffle=True)
  X_validation, X_test, y_validation, y_test = train_test_split(
      X_test, y_test, test_size=split_val, shuffle=True)
  return X_train, y_train, X_validation, y_validation, X_test, y_test


def train_decision_tree(max_dep,min_sam,X_train,y_train):
  clf = DecisionTreeClassifier(random_state=0, max_depth=max_dep, min_samples_split=min_sam)
  clf.fit(X_train, y_train)
  return clf

def evaluate(clf,X,y):
  predicted = clf.predict(X)
  accuracy = metrics.accuracy_score(y, predicted)
  return accuracy

def store_classifier(X_train,y_train,X_validation ,y_validation,file_path,type_model):
  max_deapth=[5,10,20,50,100]
  min_split= [2,4,6,8]
  optimum_accuracy = 0.3 ### minimium accuracy is kept as 30% to throw away the random-like behaviour
  optimal_depth = 0
  for d in max_deapth:
    for x in min_split:
      classifier = train_decision_tree(d,x,X_train,y_train)
      accuracy = evaluate(classifier,X_validation ,y_validation)
      accuracies.append(accuracy)
      if accuracy > optimum_accuracy:
        optimum_accuracy = accuracy
        optimal_depth = d
        dump(classifier,file_path) ###model saved in hard disk, only trained once for even optimal gamma
  return None

def main():
  digits = datasets.load_digits()
  split_train = 0.3
  split_val = 0.5
  max_depth=[5,10,20,50,100]
  min_split= [2,4,6,8]
  observation_table = {"max_depth":[],"min_split":[],"run":[],"train_acc":[],"test_acc":[],"val_acc":[]}
  for d in max_depth:
    for x in min_split:
      for i in range(3):
        X_train, y_train, X_validation, y_validation, X_test, y_test = preprocess_split_data(digits,split_train,split_val)
        classifier = train_decision_tree(d,x,X_train,y_train)
        accuracy_train = evaluate(classifier,X_train,y_train)
        accuracy_val = evaluate(classifier,X_validation,y_validation)
        accuracy_test = evaluate(classifier,X_test,y_test)
        observation_table["max_depth"].append(d)
        observation_table["min_split"].append(x)
        observation_table["run"].append(i)
        observation_table["train_acc"].append(round(accuracy_train,2))
        observation_table["test_acc"].append(round(accuracy_test,2))
        observation_table["val_acc"].append(round(accuracy_val,2))

  data_table = pd.DataFrame(data =observation_table)
  new_clm = ["max_depth","min_split","train_acc_1","test_acc_1","val_acc_1","train_acc_2","test_acc_2","val_acc_2","train_acc_3","test_acc_3","val_acc_3","train_acc_mean","test_acc_mean","val_acc_mean"]
  data_table2 = pd.DataFrame(columns= new_clm)
  for index, row in data_table.iterrows():
    if row["run"] == 0:
      entry = []
      entry.append(row["max_depth"])
      entry.append(row["min_split"])
      entry.append(row["train_acc"])
      entry.append(row["test_acc"])
      entry.append(row["val_acc"])
    elif row["run"] == 1:
      entry.append(row["train_acc"])
      entry.append(row["test_acc"])
      entry.append(row["val_acc"])
    elif row["run"] == 2:
      entry.append(row["train_acc"])
      entry.append(row["test_acc"])
      entry.append(row["val_acc"])
      mean_train = round(np.mean([entry[2],entry[5],entry[8]]),2)
      mean_test = round(np.mean([entry[3],entry[6],entry[9]]),2)
      mean_val = round(np.mean([entry[4],entry[7],entry[10]]),2)
      entry.append(mean_train)
      entry.append(mean_test)
      entry.append(mean_val)
      data_table2.loc[len(data_table2.index)] = entry 
  print(tabulate(data_table2, headers='keys', tablefmt='psql',))

if __name__ == "__main__":
  main()



        

