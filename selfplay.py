# MCTS algorithm for self play

import game
import model

class Node(game.GameState):
    pass

def selfplay(net: model.Model) -> str:
    """
    Returns Game History
        -> string containing the following in each line:
                position: Image,
                value: Int,
                prob: Float[...]
        this will go in the log and be used by train.py
    """
    g = game.GameState()
