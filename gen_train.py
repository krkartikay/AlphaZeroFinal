import os
import csv
import pickle

import torch

import config
import evaluate
import model

from dataset import ChessDataset

print("Running neural net training!")
net = model.Model('cuda')

print(net)

# with open('gen_data.pkl', 'rb') as file:
#     all_inps, all_outs, all_vals = pickle.load(file)

net.load()

results_file = open("all_moves.csv","a")
results_writer = csv.writer(results_file)
summary_file = open("summary.csv","a")
summary_writer = csv.writer(summary_file)
win, draw, loss, illegal, moves, _, all_moves = evaluate.evaluate_model(net, verbose=True)
summary_writer.writerow([win,draw,loss,illegal,0])
results_writer.writerow(all_moves)
results_file.flush()
summary_file.flush()

data_stats_file = 'training_data/data_stats.pkl'

if os.path.exists(data_stats_file):
    with open(data_stats_file, 'rb') as file:
        data_stats = pickle.load(file)
else:
    print("No training data!")
    os.abort()

chess_dataset = ChessDataset(num_files=data_stats['next_file_num'] - 1,
                             file_lengths=data_stats['file_lengths'])

for i in range(500):
    print(f"{i+1}:")
    losses = net.train_model(dataset=chess_dataset, epochs=5)
    net.store()
    win, draw, loss, illegal, moves, _, all_moves = evaluate.evaluate_model(net, verbose=True)
    summary_writer.writerow([win,draw,loss,illegal,sum(all_moves)/len(all_moves),losses[-1]])
    results_writer.writerow(all_moves)
    print(f"\n\t\tAvg moves: {moves}, Completed games: {win+draw+loss}, All games: {all_moves}\n")
    results_file.flush()
    summary_file.flush()