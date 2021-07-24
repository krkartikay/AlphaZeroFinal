# Keras model for Alpha Zero Neural network

# Imported and used by:
#  - Selfplay.py [prediction]
#  - Evaluate.py [prediction]
#  - Train.py [training]

# Also allows saving and loading data from .h5 file.

import config

import tensorflow as tf
import tensorflow.keras as keras

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Model():
    """
    Custom Neural Network with two-headed architecture
    built on top of keras functional API model.
    """
    def __init__(self):
        inp_layer = keras.Input(shape=(9,))
        l1 = keras.layers.Dense(50, activation="relu")(inp_layer)
        l2 = keras.layers.Dense(20, activation="relu")(l1)
        prob_logits = keras.layers.Dense(9, activation="relu")(l2)
        prob_head = keras.layers.Activation("softmax")(prob_logits)
        value_head = keras.layers.Dense(1, activation="tanh")(l2)
        self.model = keras.Model(inputs=inp_layer, outputs=[prob_head, value_head])
        # self.pred_model = keras.Model(inputs=inp_layer, outputs=[prob_head, value_head])
        self.model.compile(optimizer=keras.optimizers.SGD(learning_rate=config.learning_rate, decay=config.decay), loss=[
            keras.losses.KLDivergence(),
            keras.losses.MeanSquaredError()
        ])

    def predict(self, gamestate):
        return self.model(gamestate.to_image())

    def train(self, data, epochs = 100, verbose = False):
        xs, probs, values = data
        # TODO: specify some callback and log the losses
        history = self.model.fit(xs, [probs, values], batch_size=config.batch_size, epochs=epochs, verbose=verbose, shuffle=True)
        return history

    def load(self, filename="latest_weights.h5"):
        try:
            self.model.load_weights(filename)
        except OSError:
            print(f"No such file: {filename}, initialising from scratch")

    def store(self, filename="latest_weights.h5"):
        self.model.save_weights(filename)