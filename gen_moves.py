import game
import numpy
import random
import torch

random.seed(42)
torch.random.manual_seed(42)

print("Generating training data.")

all_inps = []
all_outs = []
all_vals = []

for i in range(500):
    g = game.GameState()
    while not g.terminated():
        inp = g.to_image()
        legal_actions = [i for i, x in enumerate(g.legal_actions()) if x]
        out = [int(x) for x in g.legal_actions()]
        num_legal_actions = sum(out)
        norm_out = [x/num_legal_actions for x in out]
        action = random.choice(legal_actions)
        g = g.next_state(action)
        all_inps.append(inp[0])
        all_outs.append(norm_out)
        all_vals.append(0)
    print(f"Game {i+1} done!")

print("Training data ready. Shape:")
print(len(all_inps), len(all_inps[0]))
print(len(all_outs), len(all_outs[0]))

#######################################################

import model

print("Running neural net training!")
net = model.Model('cuda')

print(net)

# net.load()

for i in range(30):
    net.train([all_inps, all_outs, all_vals], epochs=10)
    net.store()