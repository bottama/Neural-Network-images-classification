""" T2. Hyper-parameter tuning """

# import modules
import utils
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import (utils as tf_utils,
                              regularizers,
                              optimizers)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Dense,
                                     Conv2D,
                                     MaxPooling2D,
                                     AveragePooling2D,
                                     Flatten,
                                     Dropout,
                                     LeakyReLU)
import numpy as np

if __name__ == '__main__':

    # 1. load CIFAR-10 data set
    (x_train, y_train), (x_test, y_test) = utils.load_cifar10()

    # 2. Pre-process the data
    # 2.1. Normalize each pixel of each channel so that the range is [0,1];
    #      each pixel is represented by an integer value in the 0-255 range.
    x_train, x_test = x_train / 255., x_test / 255.

    # 2.2. Create one-hot encoding of the labels;
    #      pre-process targets in order to perform multi-class classification.
    n_classes = 3
    y_train = tf_utils.to_categorical(y_train, n_classes)
    y_test = tf_utils.to_categorical(y_test, n_classes)

    # 3. train test split between train and validation
    #    20% of the training data as validation set
    val_size = .2
    x_atr, x_val, y_atr, y_val = train_test_split(x_train, y_train, test_size=val_size, shuffle=True, random_state=42)

    # data check
    print('Training, Validation, Test samples: {}, {}, {}'.format(len(x_atr), len(x_val), len(x_test)))

    # 4. Model selection

    # 4.1. Define the model
    def create_ffnn(neurons, learning_rate):
        """

        :param neurons: number of neurons in the last dense hidden layer
        :param learning_rate: RMSprop optimizer learning rate
        :return: compiled model not trained
        """

        """ bonus variables """
        # Bonus 1: dropout layer with dropout probability of 0.3 before each Dense layer;
        # Bonus 2: L2 regularization with factor 0.005 on the weights to every Dense layer;
        # Bonus 3: the slope of the Leaky ReLU for x < 0 to 0.15;
        dropout_rate = .3
        regularizer = regularizers.l2(.005)
        slope = .15

        """ model architecture """
        # - Sequential model;
        # - Convolutional layer, with 8 filters of size 5 by 5, stride of 1 by 1, and Leaky ReLU activation function;
        # - Max pooling layer, with pooling size of 2 by 2;
        # - Convolutional layer, with 16 filters of size 3 by 3, stride of 2 by 2, and Leaky ReLU activation function;
        # - Average pooling layer, with pooling size of 2 by 2;
        # - Layer to convert the 2D feature maps to flat vectors;
        # - Dropout layer with dropout probability of 0.3
        # - Dense layer with 'neurons' and the Hyperbolic Tangent activation function;
        # - Dropout layer with dropout probability of 0.3;
        # - L2 regularization with factor 0.005 on the weights to every Dense layer;
        # - Dense output layer with softmax activation function;

        model = Sequential()
        model.add(Conv2D(filters=8, kernel_size=(5, 5), strides=(1, 1), input_shape=x_train.shape[1:]))
        model.add(LeakyReLU(alpha=slope))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2)))
        model.add(LeakyReLU(alpha=slope))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(rate=dropout_rate))
        model.add(Dense(units=neurons, activation='tanh', kernel_regularizer=regularizer))
        model.add(Dropout(rate=dropout_rate))
        model.add(Dense(units=3, activation='softmax', kernel_regularizer=regularizer))

        """ model compile """
        # RMSprop optimization algorithm with 'learning_rate';
        # categorical cross-entropy as a loss function;

        optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model


    # 4.2. hyper-parameters to tune:
    #      neurons - learning rate
    model_parameters = [(8, .01),
                        (8, .001),
                        (64, .01),
                        (64, .001)]

    # 4.3. Train models:
    # - train for 500 epochs;
    # - Batch size of 128;
    # - Implement early stopping, monitoring the validation accuracy of the model with a patience of 10 epochs;
    # - store best weights found
    epochs = 500
    batch_size = 128
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    # grid search
    acc_list = []
    # models iteration
    for (neurons, learning_rate) in model_parameters:
        print("Training NN with {} neurons and {} learning rate".format(neurons, learning_rate))

        # defining the feed-forward neural network
        model = create_ffnn(neurons=neurons, learning_rate=learning_rate)

        # model fitting on training set
        model.fit(x_atr, y_atr,
                  shuffle=True, batch_size=batch_size, epochs=epochs,
                  validation_data=(x_val, y_val),
                  verbose=1, callbacks=[early_stopping])

        # Evaluate the model and append the result
        _, acc = model.evaluate(x_val, y_val)
        acc_list.append(acc)

    # 4.4. Identify the most promising hyper-parameter setup
    imax = np.argmax(acc_list)
    print("Best model parameters:", model_parameters[imax])

    # Performance of best model
    (neurons, learning_rate) = model_parameters[imax]
    best_model = create_ffnn(neurons=neurons, learning_rate=learning_rate)
    history = best_model.fit(x_atr, y_atr,
                             shuffle=True, batch_size=batch_size, epochs=epochs,
                             validation_data=(x_val, y_val),
                             verbose=1, callbacks=[early_stopping])
    scores = best_model.evaluate(x_test, y_test)
    print('Test loss: {} - Accuracy: {}'.format(*scores))

    # save model in deliverable
    utils.save_keras_model(best_model, '../deliverable/nn_task2.h5')
