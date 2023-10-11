# Neural network training process.

# Starts by loading 'latest_network.h5'.
# Reads training data from 'training_data.log' and trains the network on it.
# Writes the trained network back to 'latest_network.h5'. Keeps doing this until stopped.

import config
import model

import numpy as np
import torch

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.Model(device)
    net.load()

    print("loading training data")

    lines = open("training_data.log").readlines()
    lines = lines[-config.last_N_games:]
    data = [[eval(x) for x in line.strip().split('\t')]  for line in lines]

    print("done loading")

    xs = [l[0][0] for l in data]
    values = [l[1] for l in data]
    probs = [l[2] for l in data]

    xs = np.array(xs)
    values = np.array(values)
    probs = np.array(probs)

    print(xs.shape)
    print(values.shape)
    print(probs.shape)

    hist = net.train([xs, probs, values], config.train_epochs)
    net.store()

    return hist

def main():
    losses = train()
    # print(losses)

if __name__ == "__main__":
    main()