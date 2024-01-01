import random
import torch
import evaluate
import pickle

random.seed(42)
torch.random.manual_seed(42)

#######################################################

import model

print("Running neural net training!")
net = model.Model('cuda')

print(net)

with open('gen_data.pkl', 'rb') as file:
    all_inps, all_outs, all_vals = pickle.load(file)

# net.load()

for i in range(30):
    print(f"{i+1}:")
    net.train([all_inps, all_outs, all_vals], epochs=10)
    net.store()
    win, draw, loss, illegal, moves, _, all_moves = evaluate.evaluate_model(net)
    print(f"\n\t\tAvg moves: {moves}, Completed games: {win+draw+loss}, All games: {all_moves}\n")