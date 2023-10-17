# define visualisation of board

import game
import chess
import model
import torch
import sys

def vis(fen, k):
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
    probs, indices = torch.topk(probs, k)
    for p, i in zip(probs, indices):
        print(f"Move: {g.get_move(i)}\tProb: {p:.3f}")
        g1 = g.next_state(i)
        print(g1.board)

if __name__ == '__main__':
    # fen = input("FEN: ")
    # fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
    if len(sys.argv) > 0:
        fen = ' '.join(sys.argv[1:])
    else:
        fen = ''
    vis(fen, 5)
