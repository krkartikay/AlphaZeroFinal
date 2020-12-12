# Keras model for Alpha Zero Neural network

# Imported and used by:
#  - Client.py [prediction]
#  - Evaluate.py [prediction]
#  - Train.py [training]

# Also allows saving and loading data from .h5 file.

import config

import tensorflow as tf
import tensorflow.keras as keras

class Model():
    """
    Custom Neural Network with two-headed architecture
    built on top of keras functional API model.
    """
    def __init__(self):
        inp_layer = keras.Input(shape=(9,))
        l1 = keras.layers.Dense(20, activation="relu")(inp_layer)
        l2 = keras.layers.Dense(20, activation="relu")(l1)
        prob_logits = keras.layers.Dense(9)(l2)
        prob = keras.layers.Activation("softmax")(prob_logits)
        value_head = keras.layers.Dense(1, activation="sigmoid")(l2)
        self.model = keras.Model(inputs=inp_layer, outputs=[prob_logits, value_head])
        self.pred_model = keras.Model(inputs=inp_layer, outputs=[prob, value_head])
        self.model.compile(optimizer=keras.optimizers.SGD(config.learning_rate), loss=[
            keras.losses.CategoricalCrossentropy(from_logits=True),
            keras.losses.MeanSquaredError()
        ])

    def predict(self, gamestate):
        return self.pred_model(gamestate.to_image())

    def train(self, data, epochs = 100):
        xs, probs, values = data
        # TODO: specify some callback and log the losses
        history = self.model.fit(xs, [probs, values], batch_size=2048, epochs=epochs, verbose=False)
        return history

    def load(self):
        self.model.load_weights("latest_weights.h5")

    def store(self):
        self.model.save_weights("latest_weights.h5")