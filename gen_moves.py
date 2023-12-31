import game
import numpy
import random

random.seed(42)

print("Generating training data.")

all_inps = []
all_outs = []
all_vals = []

for i in range(20):
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


inp_dim = 7 * 8 * 8
out_dim = 64 * 64
batch_dim = 100

import model

print("Running neural net training!")
net = model.Model('cuda')
net.load()

net.train([all_inps, all_outs, all_vals])

net.store()