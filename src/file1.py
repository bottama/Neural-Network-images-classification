""" T1. Follow our recipe """

# import modules
import utils
from tensorflow.keras import utils as tf_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (Dense,
                                     Conv2D,
                                     MaxPooling2D,
                                     AveragePooling2D,
                                     Flatten,
                                     Dropout,
                                     LeakyReLU)
from tensorflow.python.keras import Sequential

if __name__ == '__main__':
    # 1. Download and load CIFAR-10 data set.
    (x_train, y_train), (x_test, y_test) = utils.load_cifar10()

    # 2. Pre-process the data:
    # 2.1. Normalize each pixel of each channel so that the range is [0, 1];
    #      each pixel is represented by an integer value in the 0-255 range.
    x_train, x_test = x_train / 255., x_test / 255.

    # 2.2. Create one-hot encoding of the labels;
    #      pre-process targets in order to perform multi-class classification.
    n_classes = 3
    y_train = tf_utils.to_categorical(y_train, n_classes)
    y_test = tf_utils.to_categorical(y_test, n_classes)

    # 3. Define the feed-forward neural network
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
        # - Convolutional layer, with 8 filters of size 5 by 5, stride of 1 by 1, and Leaky ReLU activation;
        # - Max pooling layer, with pooling size of 2 by 2;
        # - Convolutional layer, with 16 filters of size 3 by 3, stride of 2 by 2, and Leaky ReLU activation;
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
        # accuracy as metric to be monitored;

        optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model


    # 4. Train model
    # - train for 500 epochs;
    # - Batch size of 128;
    # - Implement early stopping, monitoring the validation accuracy of the model with a patience of 10 epochs;
    # - Number of neurons in the last hidden layer: 8;
    # - learning rate RMSprop algorithm: 0.003;
    # - 20% of training is validation set.

    epochs = 500
    batch_size = 128
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    neurons = 8
    learning_rate = .003
    validation_split = .2

    print("Training NN with {} neurons and {} learning rate".format(neurons, learning_rate))

    # model fitting
    model = create_ffnn(neurons=neurons, learning_rate=learning_rate)
    history = model.fit(x_train, y_train,
                        shuffle=True, batch_size=batch_size, epochs=epochs,
                        validation_split=validation_split,
                        verbose=1, callbacks=[early_stopping])

    # 5. Draw a plot with epochs on the x-axis and with two graphs:
    #    the train accuracy and validation accuracy;

    def plot_history(history):
        """

        :param history: trained model history
        :return: plot regarding the model accuracy both on the train and on the validation set
        """
        plt.figure(figsize=(5, 5))
        plt.plot(history.history['accuracy'], label='train accuracy')
        plt.plot(history.history['val_accuracy'], label='validation accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('Model accuracy')
        plt.legend()
        return plt.show()


    # plot
    plot_history(history=history)

    # 6. Assess the performance of the network on the test set loaded in point 1:
    # - provide an estimate of the classification accuracy expected on new and unseen images.

    # Evaluate the model
    scores = model.evaluate(x_test, y_test)
    print('Test loss: {} - Accuracy: {}'.format(*scores))

    # save model in deliverable
    utils.save_keras_model(model, '../deliverable/nn_task1.h5')
