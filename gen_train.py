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

results_file = open("all_moves.csv","a")
results_writer = csv.writer(results_file)
summary_file = open("summary.csv","a")
summary_writer = csv.writer(summary_file)
# win, draw, loss, illegal, moves, _, all_moves = evaluate.evaluate_model(net, verbose=True)
# summary_writer.writerow([win,draw,loss,illegal,0])
# results_writer.writerow(all_moves)
# results_file.flush()
# summary_file.flush()

for i in range(100):
    print(f"{i+1}:")
    losses = net.train_model([all_inps, all_outs, all_vals], epochs=5)
    net.store()
    win, draw, loss, illegal, moves, _, all_moves = evaluate.evaluate_model(net, verbose=True)
    summary_writer.writerow([win,draw,loss,illegal,sum(all_moves)/len(all_moves),losses[-1][0]])
    results_writer.writerow(all_moves)
    print(f"\n\t\tAvg moves: {moves}, Completed games: {win+draw+loss}, All games: {all_moves}\n")
    results_file.flush()
    summary_file.flush()