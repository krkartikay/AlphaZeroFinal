# Game Specific rules

# Imported and used by:
#  - Selfplay.py [for self play]
#  - Evaluate.py [for evaluating network]

import numpy as np
from typing import List
import chess

''' 
    Represents state of the game (at any given time) : todo (edit this)
'''
class GameState:

    '''
        start the game fresh or copy other state
    '''
    def __init__(self, other=None):
        if other is None:
            self.board = chess.Board("2k5/5q2/8/8/8/8/3Q4/1K6 w - - 0 1")
        else:
            self.board = chess.Board(other.board.fen())
            
    def player(self) -> int:
        if self.board.turn:
            return 1
        else:
            return -1

    def winner(self):
        outcome = self.board.outcome()
        if outcome is None:
            return None
        
        if outcome.termination == chess.Termination.CHECKMATE:
            return (1 if outcome.winner else -1)

        return 0

    def terminated(self) -> bool:
        if self.winner() is not None:
            return True
        else:
            return False

    def legal_actions(self) -> List[bool]:

        moves = [False] * (64 * 64)

        for move in self.board.legal_moves:
            a = move.from_square
            b = move.to_square
            idx = (a * 64) + b
            moves[idx] = True

        return moves
    
    def get_move(self, action: int):
        a = action // 64
        b = action % 64

        move = chess.Move(a,b)
        if chess.square_rank(b) == (7 if self.player() == 1 else 0) and self.board.piece_type_at(a) == chess.PAWN:
            move = chess.Move(a,b,chess.QUEEN)
        
        return move
    
    def next_state(self, action: int):
        legal_actions = self.legal_actions()
        
        if not legal_actions[action]:
            raise RuntimeError("Illegal action")

        g = GameState(self)
        
        g.board.push(self.get_move(action))

        return g     

    "Returns representation of state suitable for input to neural network"
    def to_image(self):

        planes = []

        for i in range(1,7):
            plane = [  
                
                [
                    (1 if ((a * 8) + b) in self.board.pieces(chess.PieceType(i), True) else 0)
                    -(1 if ((a * 8) + b) in self.board.pieces(chess.PieceType(i), False) else 0)
                    for b in range(8)
                ] 
                
                for a in range(8)
            ] 

            planes.append(plane)

        player_plane = [[self.player() for b in range(8)] for a in range(8)] 
        planes.append(player_plane)
        return np.array([planes])

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