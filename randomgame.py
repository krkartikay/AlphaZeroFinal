import game
import config

import time
import random
import requests
import tempfile

def randomGame(exclude_illegal=False):
    g = game.GameState()
    print('New Game')
    print(g.board, '\n')
    i = 0
    while not g.terminated():
        actions = g.legal_actions()
        s = sum(actions)
        probs = [x/s for x in actions]
        action = random.choices(list(range(len(actions))), probs)[0]
        g = g.next_state(action)
        i += 1
        print (f"After Move {i}")
        print(g.board, '\n')
    winner = g.winner()
    print(winner)

if __name__ == '__main__':
    randomGame()