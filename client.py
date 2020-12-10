# MCTS Client process

# Repeatedly does the following:
# Gets latest neural network from server, plays many games of self play 
# with the MCTS algorithm, thereby generating training data, which it
# reports to the server upon termination of the game.

import requests

import game
import model
import config

