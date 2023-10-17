# define visualisation of board

import game
import chess
import model
import torch

def vis(fen):
    g = game.GameState()
    if fen != "":
        g.board = chess.Board(fen)
    print(g.board)
    net = model.Model()
    net.load()
    probs, value = net.predict(g)
    probs = probs[0]
    value = value[0]
    print(f"Value: {value.item():.2f}")
    print(f"Top Probs:")
    probs, indices = torch.topk(probs, 10)
    for p, i in zip(probs, indices):
        print(f"Move: {g.get_move(i)}\tProb: {p:.3f}")
        g1 = g.next_state(i)
        print(g1.board)

if __name__ == '__main__':
    fen = input("FEN: ")
    vis(fen)
