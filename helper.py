import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix
import matplotlib.image as mpimg
import datetime
import tensorflow_hub as hub
from tensorflow.keras import layers
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
    Checks true if the model already exists in TensorBoard and false otherwise then proceeding to add the model to tensorboard.
    Returns TensorBoard object.
    Use: 'tensorboard --logdir=./' in terminal to run tensorboard
    Args:
        model_name (string): _description_
    """
    path = r'logs\{}'.format(model_name)
    if os.path.exists(path) == False:
        NAME = f'{model_name}'
        tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
        return tensorboard
    # else:
    #     return tf.keras.callbacks.Callback()


def plot_confusion_matrix(y_true, y_pred, classes = None,figsize = (15,15),text_size =15):
    """ Plots the confusion matrix for a given model.

    Args:
        y_test (_type_): Testing Data
        y_pred (_type_): Predicted Data
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0] # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
    
    # Label the axes
    ax.set(title="Confusion Matrix",
            xlabel="Predicted label",
            ylabel="True label",
            xticks=np.arange(n_classes), # create enough axis slots for each class
            yticks=np.arange(n_classes), 
            xticklabels=labels, # axes will labeled with class names (if they exist) or ints
            yticklabels=labels)
    
    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                size=text_size)
        
        
def plot_random_images(model,images,true_labels,classes):
    """ Plots random images with labels and shows the true labels and the predicted labels along with the confidnece scores.

    Args
    model: a trained model (trained on data similar to what's in images).
    images: a set of random images (in tensor form).
    true_labels: array of ground truth labels for images.
    classes: array of class names for images.
    
    
  Returns:
    A plot of a random image from `images` with a predicted class label from `model`
    as well as the truth class label from `true_labels`.
    
    """
    i = random.randint(0,len(images))
    
    target_image = images[i]
    pred_probs = model.predict(target_image.reshape(1, 28, 28)) # have to reshape to get into right size for model
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]
    
    plt.imshow(target_image, cmap = plt.cm.binary)
    
    if pred_label == true_label:
        color = 'green'
    else:
        color = 'red'
        
    plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                     100*tf.reduce_max(pred_probs),
                                                     true_label),
                                                    color=color)


def view_random_image(target_dir, target_class):
  target_folder = target_dir+target_class
  random_image = random.sample(os.listdir(target_folder), 1)
  img = mpimg.imread(target_folder + '/' + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis('off')
  print(f'Image shape: {img.shape}')
  return img


def plot_loss_curves(history):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  plt.plot(epochs, loss, label='training loss')
  plt.plot(epochs, val_loss, label='validation loss')
  plt.title('loss')
  plt.xlabel('epochs')
  plt.legend()

  plt.figure()
  plt.plot(epochs, accuracy, label='training accuracy')
  plt.plot(epochs, val_accuracy, label='val accuracy')
  plt.title('accuracy')
  plt.xlabel('epochs')
  plt.legend()
  
  


def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + \
      datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback


IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 32

def create_model(model_url, num_classes=10):
  """Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.
  
  Args:
    model_url (str): A TensorFlow Hub feature extraction URL.
    num_classes (int): Number of output neurons in output layer,
      should be equal to number of target classes, default 10.

  Returns:
    An uncompiled Keras Sequential model with model_url as feature
    extractor layer and Dense output layer with num_classes outputs.
  """
  # Download the pretrained model and save it as a Keras layer
  feature_extractor_layer = hub.KerasLayer(model_url,
                                           trainable=False,  # freeze the underlying patterns
                                           name='feature_extraction_layer',
                                           input_shape=IMAGE_SHAPE+(3,))  # define the input image shape

  # Create our own model
  model = tf.keras.Sequential([
      feature_extractor_layer,  # use the feature extraction layer as the base
      layers.Dense(num_classes, activation='softmax',
                   name='output_layer')  # create our own output layer
  ])

  return model


def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  """
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend()
