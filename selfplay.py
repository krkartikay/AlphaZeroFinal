# MCTS algorithm for self play

import game
import model
import config

import random
from math import sqrt, log

class Node(game.GameState):
    def __init__(self, *args, **kwargs):
        super(Node, self).__init__(*args, **kwargs)
        self.prob = 0.0 # P
        self.value = 0.0 # V
        self.visit = 0 # N
        self.value_sum = 0.0 # Q * N = sum(value of leaf nodes)
        self.children = {}
        self.parent = None
    
    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, net: model.Model):
        if self.terminated():
            self.value = self.leaf_value()
        else:
            l = self.legal_actions()
            probs, value = net.predict(self)
            self.value = value[0][0]
            for i in range(config.num_actions):
                if l[i]:
                    self.children[i] = Node(self.next_state(i))
                    self.children[i].prob = probs[0][i]
                    self.children[i].parent = self

    def ucb_score(self):
        return self.value_sum / (self.visit + 1) + self.prob * sqrt(self.parent.visit / (1 + self.visit)) * config.pb_c_init
        # * (config.pb_c_init + log((self.parent.visit + config.pb_c_base + 1) / config.pb_c_base))

class MCTS():
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
        while not g.terminated():
            probs = self.get_probs(g)
            history.append((g, probs))
            # choose action acc to probs
            action = random.choices(list(range(config.num_actions)), probs)[0]
            move_history.append(action)
            g = g.next_state(action)
        # log game outcome?
        print(move_history, g.winner())
        return self.encode_history(history)

    def get_probs(self, g: game.GameState):
        "runs N simulations and returns visit probabilities at root"
        root = Node(g)
        for i in range(config.num_simulate):
            self.simulate(root)
        self.print_tree(root)
        return [
            root.children[i].visit/(root.visit-1) if i in root.children else 0
            for i in range(config.num_actions)
        ]

    def simulate(self, n: Node):
        "goes upto one leaf state and evaluates it and backpropagates the value"
        n.visit += 1
        if n.is_expanded():
            # choose best child node and recurse
            m = self.choose_child(n)
            val_child = - self.simulate(m)
            n.value_sum += val_child
            return val_child
        else:
            # expand this node and return value
            n.expand(self.net)
            return n.value

    def choose_child(self, n: Node):
        score, node = max(((m.ucb_score(), m) for m in n.children.values()), key=lambda x: x[0])
        return node

    def encode_history(self, history) -> str:
        return f"{history}"

    def print_tree(self, n: Node, depth=0):
        if n.visit:
            print(depth*"\t" + f">> position: {n.state}, prob: {n.prob:0.2}, value: {n.value:0.2},"
                            f"visit: {n.visit}, value_sum: {n.value_sum:0.2}")
            for ch in n.children.values():
                self.print_tree(ch, depth+1)

if __name__ == "__main__":
    net = model.Model()
    net.load("tmodel.h5")
    m = MCTS(net)
    g = game.GameState()
    print("Final prbs: ", m.get_probs(g))
