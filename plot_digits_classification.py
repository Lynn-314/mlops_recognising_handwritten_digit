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
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean


###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()
"""
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)
"""

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

def classify_given_res_and_split(test_size,res,data):
  n_samples = len(digits.images)
  resized_data = resize(data, (n_samples,res, res),anti_aliasing=True)
  
  data = resized_data.reshape((n_samples, -1))

  # Create a classifier: a support vector classifier
  clf = svm.SVC(gamma=0.001)

  # Split data 
  X_train, X_test, y_train, y_test = train_test_split(
      data, digits.target, test_size=test_size, shuffle=False)

  # Learn the digits on the train subset
  clf.fit(X_train, y_train)

  # Predict the value of the digit on the test subset
  predicted = clf.predict(X_test)


  ###############################################################################
  # :func:`~sklearn.metrics.classification_report` builds a text report showing
  # the main classification metrics.

  #print(f"Classification report for classifier {clf}:\n"f"{metrics.classification_report(y_test, predicted)}\n")
  accuracy = metrics.accuracy_score(y_test, predicted)
  f1_score = metrics.f1_score(y_test, predicted,average="macro")
  return accuracy,f1_score




resolutions = [64,32,16]
test_sizes = [0.1,0.2,0.3,0.4]


for res in resolutions:
  for test_size in test_sizes:
    accuracy,f1_score = classify_given_res_and_split(test_size=test_size,res=res,data=digits.images)
    print(f"For resolotion {res}X{res}, and train-test split {int((1-test_size)*100)}-{int((test_size)*100)}\n accuracy is {accuracy:.4f} and F1 score is {f1_score:.4f}",)