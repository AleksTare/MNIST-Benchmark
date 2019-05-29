from mlxtend.data import loadlocal_mnist
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import *
import numpy as np
import time
import preproc
import pandas as pd

print("\bLoading dataset..")
train_images, train_labels = loadlocal_mnist(
        images_path='MNIST_data/train-images-idx3-ubyte/train-images.idx3-ubyte',
        labels_path='MNIST_data/train-labels-idx1-ubyte/train-labels.idx1-ubyte')
test_images, test_labels = loadlocal_mnist(
        images_path='MNIST_data/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte',
        labels_path='MNIST_data/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte')

new_df = pd.DataFrame(data=np.int_(train_images[0:, 0:]))
# train_changing_pixels_df, dropped_pixels = preproc.remove_constant_pixels(new_df)
train_images_pd, dropped_pixels = preproc.remove_constant_pixels(new_df)
train_images = np.array(train_images_pd)
##
# train = preproc.reshape_to_img(train_images)
# bounds_train = preproc.get_data_to_box(train) * 1. / 28
##
# Determine right number of Dimensions d
pca = PCA()
pca.fit(train_images)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95)+1

pca = PCA(n_components = d)
train_images_reduced = pca.fit_transform(train_images)
test_images_reduced = pca.fit_transform(test_images)

clf = LinearSVC()
start = time.time()
print("\bTraining model..")
clf.fit(train_images_reduced,train_labels)
duration = time.time()-start
print("\bTrain duration: ", duration, 's')
print("\b----------------------Results----------------------")

##
# Determine Cross Validation Score & Prediction
crossVScore = cross_val_score(clf,train_images_reduced, train_labels, cv=3, scoring="accuracy")
print('\bCross Validation Score = ' + str(crossVScore))

y_train_pred = cross_val_predict(clf,train_images_reduced, train_labels, cv=3)
print('\bCross Validation Prediction = ' + str(y_train_pred))

##
# Determine Confusion Matrix
confusionMatrix = confusion_matrix(train_labels,y_train_pred)
print("\bConfusion Matrix = \n", confusionMatrix)

##
# Determine Precision and Recall and F1
precision = precision_score(train_labels, y_train_pred,average='micro')
recall = recall_score(train_labels, y_train_pred, average='micro')
f1 = f1_score(train_labels, y_train_pred, average='micro')

print('\bPrecision: ', precision)
print('Recall: ', recall)
print('F1 Score: ', f1)