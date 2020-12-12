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
import tempfile

net = model.Model()

def load_model_weights():
    data = requests.get(config.server_address + "/weights")
    tfile, tname = tempfile.mkstemp(".h5")
    open(tname, "wb").write(data.content)
    net.load(tname)
    g = game.GameState()
    g.state = [0,0,1,0,1,0,0,0,-1]
    print(net.predict(g))

def play_game():
    # use MCTS to play a game, and return training data
    # maybe transfer this stuff to mcts.py or something
    pass

def upload_data():
    # send back training data to server
    resp = requests.post(config.server_address + "/train", data=b"working\n")
    assert(resp.content == b'"OK"\n')

while True:
    load_model_weights()
    play_game()
    upload_data()
    break