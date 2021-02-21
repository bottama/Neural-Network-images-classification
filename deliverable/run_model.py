""" Run model """

# import modules
from tensorflow.keras import utils as tf_utils
from src import utils
import numpy as np

if __name__ == '__main__':

    # Load the test CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = utils.load_cifar10()
    # ...

    # Pre-processing
    # Normalize each pixel of each channel so that the range is [0, 1];
    # each pixel is represented by an integer value in the 0-255 range.
    x_train, x_test = x_train / 255., x_test / 255.

    # Create one-hot encoding of the labels;
    # pre-process targets in order to perform multi-class classification.
    n_classes = 3
    y_train = tf_utils.to_categorical(y_train, n_classes)
    y_test = tf_utils.to_categorical(y_test, n_classes)
    # ...

    # Load the trained models
    model_task1 = utils.load_keras_model('../deliverable/nn_task1.h5')
    model_task2 = utils.load_keras_model('../deliverable/nn_task2.h5')

    # Predict on the given samples
    y_pred_task1 = model_task1.predict(x_test)
    y_pred_task2 = model_task2.predict(x_test)

    # Evaluate the miss-classification error on the test set
    assert y_test.shape == y_pred_task1.shape
    assert y_test.shape == y_pred_task2.shape
    # model accuracy
    acc1 = (np.argmax(y_test, axis=1) == np.argmax(y_pred_task1, axis=1)).mean()
    acc2 = (np.argmax(y_test, axis=1) == np.argmax(y_pred_task2, axis=1)).mean()
    # showing model accuracy results
    print("Accuracy model task 1:", acc1)
    print("Accuracy model task 2:", acc2)
