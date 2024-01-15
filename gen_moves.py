import os
import game
import numpy
import random
import torch
import pickle

# random.seed(42)
# torch.random.manual_seed(42)

NUM_GAMES = 100

print("Generating training data.")

all_inps = []
all_outs = []
all_vals = []

for i in range(NUM_GAMES):
    g = game.GameState()
    while not g.terminated():
        inp = g.to_image()
        legal_actions = [i for i, x in enumerate(g.legal_actions()) if x]
        out = [int(x) for x in g.legal_actions()]
        out = numpy.array(out)
        # num_legal_actions = out.sum()
        # norm_out = out/num_legal_actions
        action = random.choice(legal_actions)
        g = g.next_state(action)
        all_inps.append(inp[0])
        all_outs.append(out)
        all_vals.append(0)
    print(f"Game {i+1} done!")

all_inps = torch.Tensor(numpy.array(all_inps))
all_outs = torch.Tensor(numpy.array(all_outs))
all_vals = torch.Tensor(numpy.array(all_vals))

print("Training data ready. Shape:")
print(all_inps.shape)
print(all_outs.shape)
print(all_vals.shape)

# Training data stats file
data_stats_file = 'training_data/data_stats.pkl'

# Check if the file exists and load the dictionary, otherwise initialize a new one
if os.path.exists(data_stats_file):
    with open(data_stats_file, 'rb') as file:
        data_stats = pickle.load(file)
else:
    data_stats = { 'next_file_num': 1, 'file_lengths': []}

next_file_num = data_stats['next_file_num']

data_stats['file_lengths'].append(len(all_inps))
data_stats['next_file_num'] = next_file_num + 1

# Store the updated dictionary
with open(data_stats_file, 'wb') as file:
    pickle.dump(data_stats, file)

# Update the dictionary and saved it first to minimise race conditions
# Writing the file will take time so we do it later
data = {'all_inps': all_inps, 'all_outs': all_outs, 'all_vals': all_vals}
with open(f'training_data/data_{next_file_num}.pt', 'wb') as data_file:
    torch.save(data, data_file)

print(data_stats)
