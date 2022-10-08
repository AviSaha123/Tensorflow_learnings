import itertools
from turtle import title
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix
import os 



def plot_decision_boundary(model, X, y):
        """Plots the decision boundary created by a model  predicting on X

        Args:
            model (_type_): model
            X (_type_): Training data
            y (_type_): Testing Data
        """
        # Define the axis boundaries of the plot and create meshgrid
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max()+0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max()+0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        # Create X values to predict

        x_in = np.c_[xx.ravel(), yy.ravel()]

        # Make Predictions
        y_pred = model.predict(x_in)

        # Check for multi class classification

        if len(y_pred[0]) > 1:
            print('Doing multicalss classification')
            y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
        else:
            print('Binary Classification')
            y_pred = np.round(y_pred).reshape(xx.shape)

        # Plot the decision boundary
        # plt.figure(figsize=(10,6))
        plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())


def plot_model_training(history):
    # try:
    #     plt.plot(history.history['mae'])
    #     plt.plot(history.history['val_mae'])
    #     plt.title('model MAE')
    #     plt.ylabel('accuracy')
    #     plt.xlabel('mae')
    #     plt.legend(['train', 'test'], loc='upper right')
    #     plt.show()
    # except:
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model MAE')
    plt.ylabel('accuracy')
    plt.xlabel('mae')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()


def save_in_tensorboard(model_name):
    """ Functions checks if model with the given name already exists in TensorBoard. 
    Returns true if the model already exists in TensorBoard and false otherwise then proceeding to add the model to tensorboard.
    Use: 'tensorboard --logdir=./' in terminal to run tensorboard
    Args:
        model_name (_type_): _description_
    """
    path = r'logs\{}'.format(model_name)
    if os.path.exists(path) == False:
        NAME = f'{model_name}'
        tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
        return tensorboard
    # else:
    #     return tf.keras.callbacks.Callback()


def plot_confusion_matrix(y_test, y_pred):
    """ Plots the confusion matrix for a given model.

    Args:
        y_test (_type_): Testing Data
        y_pred (_type_): Predicted Data
    """
    cm = confusion_matrix(y_test, tf.round(y_pred))
    cm_norm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    classes = False
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    ax.set(title='Confusion matrix',
           xlabel='Predicted Model',
           ylabel='True Label',
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels,
           )

    threshold = (cm.max() + cm.min()) / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i,j]} ({cm_norm[i,j]* 100: .1f} %)',
                 horizontalalignment='center',
                 color= 'white' if cm[i, j] > threshold else 'black',
                 size = 15)
