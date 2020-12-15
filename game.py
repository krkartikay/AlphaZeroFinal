# Game Specific rules

# Imported and used by:
#  - Selfplay.py [for self play]
#  - Evaluate.py [for evaluating network]

import numpy as np
from typing import List

win_patterns = [
    [0, 1, 2],  [3, 4, 5],  [6, 7, 8],  [0, 3, 6],
    [1, 4, 7],  [2, 5, 8],  [0, 4, 8],  [2, 4, 6],
]

class GameState:

    def __init__(self, other=None):
        if other is not None:
            self.state = other.state[:]  # Clone data
        else:
            self.state = [0,0,0, 0,0,0, 0,0,0]

    def player(self) -> int:
        if sum(x != 0 for x in self.state) % 2 == 0:
            return 1
        else:
            return -1

    def winner(self):
        for l in win_patterns:
            a, b, c = l
            if (self.state[a] == self.state[b] == self.state[c] and self.state[a] != 0):
                return self.state[a]  # some player wins
        if sum(x != 0 for x in self.state) == 9:
            return 0  # DRAW
        # no one wins
        return None

    def terminated(self) -> bool:
        if self.winner() is not None:
            return True
        else:
            return False

    def legal_actions(self) -> List[bool]:
        if self.terminated():
            return [False]*9
        else:
            return [x==0 for x in self.state]
    
    def next_state(self, action: int):
        if self.state[action] == 0:
            g = GameState(self)
            g.state[action] = self.player()
            return g
        else:
            raise RuntimeError("Illegal action")
    
    def to_image(self):
        "Returns representation of state suitable for input to neural network"
        return np.array([self.state])

    # for minimax, optimal value is:
    # optimal_value(s) = max(optimal_value(ch) for ch in children_states(s))
    #  ... i.e. maximising my action's value
    def leaf_value(self) -> int:
        """
        Returns the value of the current state from the current player's perspective.
        +1 : This is a "won" state
        -1 : This is a "lost" state
        """
        winner = self.winner()
        if winner == 0:
            return 0
        else:
            # in tic tac toe I don't need to check who won, because the player
            # who made the last move must have won
            return +1