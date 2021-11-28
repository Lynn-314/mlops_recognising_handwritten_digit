import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from joblib import dump,load


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

def compare_models(model1,model2,X,y,model1label,model2label):
  predicted1 = model1.predict(X)
  predicted2 = model2.predict(X)
  c_m1 = metrics.confusion_matrix(y,predicted1)
  c_m2 = metrics.confusion_matrix(y,predicted2)
  print(model1label,"\n",c_m1)
  print(model2label,"\n",c_m2)

def store_classifier_with_optimal_gamma(X_train,y_train,X_validation ,y_validation,file_path):
  gammas=[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5]
  accuracies=[]
  f1_scores = []
  optimum_accuracy = 0 
  optimal_gamma = 0
  for g in gammas:
    classifier = train_with_gamma(g,X_train,y_train)
    accuracy,f1_score = evaluate(classifier,X_validation ,y_validation)
    accuracies.append(accuracy)
    f1_scores.append(f1_score)
    #print(f"For gamma {g} accuracy is {accuracy:.4f} and F1 score is {f1_score:.4f}")
    if accuracy > optimum_accuracy:
      optimum_accuracy = accuracy
      optimal_gamma = g
      dump(classifier,file_path) ###model saved in hard disk, only trained once for even optimal gamma
  return None

def main():
  digits = datasets.load_digits()
  file_path ='/content/digit_recog_model'
  model_paths = []
  test_accuracies = []
  split_1 = 0.2
  split_2 = 0.5
  X_train, y_train, X_validation, y_validation, X_test, y_test = preprocess_split_data(digits,split_1,split_2)

  for i in range(10):
    path = file_path + str(i+1) + "0%trainingdata"
    model_paths.append(path)
    training_len = int((i+1)*len(X_train)/10)
    X_train_new = X_train[:training_len]
    y_train_new = y_train[:training_len]
    store_classifier_with_optimal_gamma(X_train_new,y_train_new,X_validation ,y_validation,path)
    optimal_classifier = load(path)
    accuracy,f1_score = evaluate(optimal_classifier,X_test ,y_test)
    test_accuracies.append(accuracy)

  plt.figure(figsize=(10, 5))
  plt.plot(np.arange(1,11)*10,test_accuracies)
  plt.xlabel("percentage of training data used")
  plt.ylabel("test accuracy")
  plt.xticks(np.arange(1,11)*10)
  plt.ylim(0.6,1)
  plt.show()


  for i in range(9):
    print(f"comparing models with {(i+1)*10:d}% training data vs  with {(i+2)*10:d}% training data")
    model1label = str((i+1)*10) + "% data used"
    model2label = str((i+2)*10) + "% data used"
    model1 = load(model_paths[i])
    model2 = load(model_paths[i+1])
    compare_models(model1,model2,X_test ,y_test,model1label,model2label)




if __name__ == "__main__":
  main()
