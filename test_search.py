import random
import time
import model
import selfplay
import game
import chess
import config

selfplay.debug = True
config.num_simulate = 100

net = model.Model()
net.load()

mcts = selfplay.MCTS(net)

g = game.GameState()
# g.board = chess.Board("3K4/8/8/8/4Q3/2B5/8/2k5 w - - 9 1")  # mate in 2
g.board = chess.Board("3K4/8/8/8/8/2B5/6Q1/2k5 b - - 10 1")  # mate in 1

root = selfplay.Node(g)
for i in range(config.num_simulate):
    print("simulating from root")
    mcts.simulate(root)
    mcts.print_tree(root, all=False)
    time.sleep(1)

print(g.board)


p = [
    root.children[i].visit / (root.visit - 1) if i in root.children else 0.0
    for i in range(config.num_actions)
]

for i, px in enumerate(p):
    if px:
        print(g.get_move(i), px)
