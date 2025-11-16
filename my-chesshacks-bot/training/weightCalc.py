import torch
from torch.utils.data import Dataset

import json
import chess
from stockfish import Stockfish
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import os

import ijson

# cp = get_best_centipawn(fen, sf)
# # optional: normalize / clamp
# cp = np.clip(cp, -1000, 1000) / 1000.0

def load_stockfish():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stockfish_path = os.path.join(project_root, "engines", "stockfish")
    if not os.path.exists(stockfish_path):
        raise FileNotFoundError(f"Stockfish not found at {stockfish_path}")
    sf = Stockfish(stockfish_path)
    sf.update_engine_parameters({"MultiPV": 1})
    return sf

# Weight formula/algorithm for best move
def weight(fen: str, gameStatus: str) -> int:
    board = chess.Board(fen) # To store moves and interact with
    sf = load_stockfish()
    sf.set_fen_position(fen)

    bestMoveInfo = sf.get_top_moves(1)[0] # Dict of info about best move according to Stockfish
    bestMove = bestMoveInfo["Move"]
    cpInitial = bestMoveInfo["Centipawn"]

    board.push(bestMove) # Update board with best move
    sf.set_fen_position(board.fen()) # Update sf's board
    cpFinal = sf.get_top_moves(1)["Centipawn"] # New centipawn
    
    playerToMove = fen.split(" ")[1] # "w" or "b" from original board state
    if playerToMove == "w": # Difference in centipawn depends on who the move was made for
        cp = cpFinal - cpInitial
    else:
        cp = cpInitial - cpFinal

    # Tuning values based on game status (EXPERIMENTAL)
    if gameStatus == "opening":
        cp *= 1.3
    elif gameStatus == "endgame":
        cp *= 1.15

    cp = np.clip(cp, -1000, 1000) / 1000.0 # Normalize centipawn value to dodge outliers

    return cp # Weight should be given based on quality of best move (i.e., centipawn difference)

# if __name__ == "__main__":
#     print("foo")