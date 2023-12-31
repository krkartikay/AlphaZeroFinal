import game
import numpy
import random
import torch
import pickle

random.seed(42)
torch.random.manual_seed(42)

NUM_GAMES = 1000

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
        num_legal_actions = out.sum()
        norm_out = out/num_legal_actions
        action = random.choice(legal_actions)
        g = g.next_state(action)
        all_inps.append(inp[0])
        all_outs.append(norm_out)
        all_vals.append(0)
    print(f"Game {i+1} done!")

print("Training data ready. Shape:")
all_inps = torch.Tensor(numpy.array(all_inps))
all_outs = torch.Tensor(numpy.array(all_outs))
# print(len(all_inps), len(all_inps[0]))
# print(len(all_outs), len(all_outs[0]))
print(all_inps.shape)
print(all_outs.shape)

with open('gen_data.pkl', 'wb') as file:
    pickle.dump((all_inps, all_outs, all_vals), file)

