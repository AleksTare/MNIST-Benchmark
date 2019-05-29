import numpy as np
import time
import preproc
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import *
import pandas as pd
import matplotlib.pyplot as plt
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)
from mlxtend.data import loadlocal_mnist
import spinner

with spinner.Spinner():
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

    X,y = train_images,train_labels

    print('\b---------------------------------------------------')
    print('------------------Random Forest--------------------')
    print('--------------Multiclass Classifier----------------')

    ##
    # Determine right number of Dimensions d
    pca = PCA()
    pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d = np.argmax(cumsum >= 0.95)+1

    plt.plot(cumsum)
    plt.xlabel("Dimensions")
    plt.ylabel("Explained Variance")
    plt.show()

    pca = PCA(n_components = d)
    X_reduced = pca.fit_transform(X)


    ##
    # Split the Datasets into training and testing
    #X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    ##
    # With and Without PCA
    X_train_pca, X_test_pca = X_reduced[:60000], X_reduced[60000:]

    ##
    # Shuffle Datasets of training
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = train_images[shuffle_index], train_labels[shuffle_index]
    X_train_pca = X_train_pca[shuffle_index]

    ##
    # Train without PCA
    #start = time.time()
    ##
    # Initialize Random Forest Classifier and train it
    # rf_clf = RandomForestClassifier()
    # rf_clf.fit(X_train,y_train)
    # duration = time.time()-start
    # print("\bDuration without PCA: ", duration, 's')

    ##
    # Train with PCA
    start = time.time()
    ##
    # Initialize Random Forest Classifier and train it
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train_pca,y_train)
    duration = time.time()-start
    print("\bDuration with PCA: ", duration, 's')
    print("----------------------Results----------------------")

    ##
    # Determine Cross Validation Score & Prediction
    crossVScore = cross_val_score(rf_clf, X_train_pca, y_train, cv=3, scoring="accuracy")
    print('\bCross Validation Score = ' + str(crossVScore))

    y_train_pred = cross_val_predict(rf_clf, X_train_pca, y_train, cv=3)
    print('\bCross Validation Prediction = ' + str(y_train_pred))

    ##
    # Determine Confusion Matrix
    confusionMatrix = confusion_matrix(y_train,y_train_pred)
    print("\bConfusion Matrix = \n", confusionMatrix)

    ##
    # Determine Precision and Recall and F1
    precision = precision_score(y_train, y_train_pred, average='weighted')
    recall = recall_score(y_train, y_train_pred, average='weighted')
    f1 = f1_score(y_train, y_train_pred, average='weighted')

    print('\bPrecision: ', precision)
    print('Recall: ', recall)
    print('F1 Score: ', f1)

