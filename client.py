# Client process

# Repeatedly does the following:
# Gets latest neural network from server, plays many games of self play 
# with the MCTS algorithm, thereby generating training data, which it
# reports to the server upon termination of the game.

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

def play_game():
    # use MCTS to play a game, and return training data
    # encode the data in a string format
    mcts = selfplay.MCTS(net)
    game_history = mcts.selfplay()
    ############# TODO #############
    print(game_history)
    import sys; sys.exit(0)
    ################################
    return game_history

def upload_data(string):
    # send back training data to server
    resp = requests.post(config.server_address + "/train", data=string.encode())
    assert(resp.content == b'"OK"\n')

while True:
    load_model_weights()
    data = play_game()
    upload_data(data)