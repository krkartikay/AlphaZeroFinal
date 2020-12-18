# Neural network training process.

# Starts by loading 'latest_network.h5'.
# Reads training data from 'training_data.log' and trains the network on it.
# Writes the trained network back to 'latest_network.h5'. Keeps doing this until stopped.

import config
import model

import numpy as np

net = model.Model()
net.load()

lines = open("training_data.log").readlines()
data = [[eval(x) for x in line.strip().split('\t')]  for line in lines]

xs = [l[0] for l in data]
values = [l[1] for l in data]
probs = [l[2] for l in data]

xs = np.array(xs)
values = np.array(values)
probs = np.array(probs)

hist = net.train([xs, probs, values], config.train_epochs, True)

from matplotlib import pyplot as plt
plt.plot(hist.history['loss'])
plt.show()

net.store()
