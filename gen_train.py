import random
import torch
import evaluate
import pickle
import csv

random.seed(42)
torch.random.manual_seed(42)

#######################################################

import model

print("Running neural net training!")
net = model.Model('cuda')

print(net)

with open('gen_data.pkl', 'rb') as file:
    all_inps, all_outs, all_vals = pickle.load(file)

net.load()

with open("all_moves.csv","a") as results_file:
    results_writer = csv.writer(results_file)
    win, draw, loss, illegal, moves, _, all_moves = evaluate.evaluate_model(net, verbose=True)
    results_writer.writerow(all_moves)
    results_file.flush()

    for i in range(100):
        print(f"{i+1}:")
        net.train([all_inps, all_outs, all_vals], epochs=10)
        net.store()
        win, draw, loss, illegal, moves, _, all_moves = evaluate.evaluate_model(net, verbose=True)
        results_writer.writerow(all_moves)    
        print(f"\n\t\tAvg moves: {moves}, Completed games: {win+draw+loss}, All games: {all_moves}\n")
        results_file.flush()