# Client process

# Repeatedly does the following:
# Gets latest neural network from server, plays many games of self play 
# with the MCTS algorithm, thereby generating training data, which it
# reports to the server upon termination of the game.

import game
import model
import config
import selfplay

import requests

net = model.model()

def load_model_weights():
    # load weights from server
    pass

def play_game():
    # use MCTS to play a game, and return training data
    # maybe transfer this stuff to mcts.py or something
    pass

def upload_data():
    # send back training data to server
    pass

while True:
    load_model_weights()
    data = play_game()
    upload_data(data)