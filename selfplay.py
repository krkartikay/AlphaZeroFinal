# MCTS algorithm for self play

import game
import model
import config

import random
from math import sqrt, log, exp
import numpy as np
import torch


class Node(game.GameState):
    def __init__(self, *args, **kwargs):
        super(Node, self).__init__(*args, **kwargs)
        self.prob = 0.0  # P
        self.value = 0.0  # V
        self.visit = 0  # N
        self.value_sum = 0.0  # Q * N = sum(value of leaf nodes)
        self.children = {}
        self.parent = None
        self.is_expanded = False

    def expand(self, net: model.Model):
        self.is_expanded = True
        if self.terminated():
            self.value = float(self.leaf_value())
        else:
            l = self.legal_actions()
            probs, value = net.predict(self)
            l = torch.Tensor(l).to(probs.device)
            result_sum = (probs[0] * l).sum()
            nf = 1 / result_sum
            self.value = value[0][0].item()
            for i in range(config.num_actions):
                if l[i]:
                    self.children[i] = Node(self.next_state(i))
                    self.children[i].prob = probs[0][i] * nf
                    self.children[i].parent = self
        self.value_sum += self.value
        self.visit = 1

    def avg_value(self):
        if self.visit != 0:
            return self.value_sum / self.visit
        else:
            return 0

    def ucb_score(self):
        avg_val = self.avg_value()
        exp_score = (
            self.prob * sqrt(self.parent.visit / (1 + self.visit)) * config.pb_c_init
        )
        return avg_val + exp_score
        # * (config.pb_c_init + log((self.parent.visit + config.pb_c_base + 1) / config.pb_c_base))


class MCTS:
    """
    MCTS Algorithm

    Initialise with network and use selfplay()
    """

    def __init__(self, net: model.Model):
        self.net = net

    def selfplay(self) -> str:
        """
        Returns Game History
            -> string containing the following in each line:
                    position: Image,
                    value: Int,
                    prob: Float[...]
            this will go in the log and be used by train.py
        """
        g = game.GameState()
        history = []
        move_history = []
        print("New game")
        print(g.board, "\n")
        while not g.terminated():
            probs = self.get_probs(g)
            p = np.array(probs)
            p = p ** (1 / config.temperature)
            history.append([g, probs, 0])
            # choose action acc to probs
            action = random.choices(list(range(config.num_actions)), p)[0]
            move_history.append(action)
            g = g.next_state(action)
            print(g.board)
            print()
            # print(g.board.fen(), action)
        # history.append([g, [0.0]*9, 0])
        # log game outcome
        winner = g.winner()
        # print(move_history, winner)
        # fill in the value function
        if winner != 0:
            for state in history:
                g = state[0]
                if g.player() == winner:
                    state[2] = -1
                else:
                    state[2] = 1
        return history

    def get_probs(self, g: game.GameState):
        "runs N simulations and returns visit probabilities at root"
        root = Node(g)
        for i in range(config.num_simulate):
            self.simulate(root)
        # self.print_tree(root)
        return [
            root.children[i].visit / (root.visit - 1) if i in root.children else 0.0
            for i in range(config.num_actions)
        ]

    def simulate(self, n: Node):
        "goes upto one leaf state and evaluates it and backpropagates the value"
        if n.is_expanded:
            # choose best child node and recurse
            n.visit += 1
            if n.terminated():  # if we reach terminal state
                n.value_sum += n.value
                return n.value
            m = self.choose_child(n)
            val_child = -self.simulate(m)
            n.value_sum += val_child
            return val_child
        else:
            # expand this node and return value
            n.expand(self.net)
            return n.value

    def choose_child(self, n: Node):
        # Get the scores in a list
        score, node = max(
            ((m.ucb_score().item(), m) for m in n.children.values()), key=lambda x: x[0]
        )
        return node

    def encode_history(self, history) -> str:
        lines = []
        for g, probs, val in history:
            lines.append(f"{g.to_image().tolist()}\t{val}\t{probs}")
        return "\n".join(lines) + "\n"

    def print_tree(self, n: Node, max_depth=3, depth=0, all=True, prefix=""):
        if n.visit:
            a = ">>"
        else:
            a = "--"
        if (all or n.visit) and depth < max_depth:
            print(
                depth * "\t"
                + f"{a} {prefix} position: {n.board.fen()}, prob: {n.prob:0.2}, value: {n.value:0.2},"
                f" visit: {n.visit}, avg_value: {n.avg_value():0.2}"
            )
            for k, ch in n.children.items():
                self.print_tree(
                    ch, max_depth, depth + 1, all, prefix + " " + str(n.get_move(k))
                )


if __name__ == "__main__":
    net = model.Model()
    m = MCTS(net)
    net.load("tmodel.h5")
    print("=" * 60)
    while True:
        print(m.selfplay())
