# Neural network training process.

# Starts by loading 'latest_network.h5'.
# Reads training data from 'training_data.log' and trains the network on it.
# Writes the trained network back to 'latest_network.h5'. Keeps doing this until stopped.

import config
import model

import numpy as np
import torch
import json
import time

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.Model(device)
    try:
        net.load()
    except FileNotFoundError:
        net.store()
        print("Initialised new model")

    print("loading training data")

    try:
        with open("training_data.log", "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("No training data")
        return []

    lines = lines[-config.last_N_games:]
    data = [[json.loads(x) for x in line.strip().split('\t')]  for line in lines]

    print("done loading")

    xs = [l[0][0] for l in data]
    values = [l[1] for l in data]
    probs = [l[2] for l in data]

    xs = torch.Tensor(xs)
    values = torch.Tensor(values)
    probs = torch.Tensor(probs)

    print(xs.shape)
    print(values.shape)
    print(probs.shape)

    hist = net.train_model([xs, probs, values], config.train_epochs)
    net.store()

    return hist

import gc

def main():
    try:
        log = open('loss.tsv', 'r')
        log.close()
    except FileNotFoundError:
        with open("loss.tsv", "w") as log:
            log.write("total\tprob\tvalue\n")
    #while True:
    losses = train()
    with open("loss.tsv", "a") as log:
        for total, prob, value in losses:
            log.write(f"{total}\t{prob}\t{value}\n")
    # gc.collect()
    # torch.cuda.empty_cache()
    # print(losses)

if __name__ == "__main__":
    main()