from .utils import chess_manager, GameContext
from chess import Move
import random
import time
import numpy as np
from stockfish import Stockfish
import platform
import os
# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

from datasets import load_dataset

# Load the Lichess dataset in streaming mode
dataset = load_dataset("jrahn/yolochess_lichess-elite_2211", split="train", streaming=True)
print(dataset)

def fen_to_tensor(fen_string):

    parts = fen_string.split(' ') # Denotes the various parts of FEN notation
    piece_placement = parts[0] 
    active_color = parts[1] # Which colour to move
    castling_rights = parts[2] # Whether or not each side can castle kingside/queenside
    en_passant_target = parts[3] # En passant target square

    # Initialize 17 planes (example: 12 pieces, 1 active color, 4 castling)
    tensor = np.zeros((17, 8, 8), dtype=np.float32)

    # --- Piece Placement Encoding ---
    row, col = 0, 0
    piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}

    for char in piece_placement:
        if char == '/':
            row += 1
            col = 0
        elif char.isdigit():
            col += int(char)
        else:
            plane_idx = piece_map[char]
            tensor[plane_idx, row, col] = 1
            col += 1

    # --- Active Color Encoding ---
    if active_color == 'w':
        tensor[12, :, :] = 1 # Plane 12 for white to move
    # else: black to move (plane remains 0)

    # --- Castling Rights Encoding ---
    if 'K' in castling_rights:
        tensor[13, :, :] = 1 # White Kingside
    if 'Q' in castling_rights:
        tensor[14, :, :] = 1 # White Queenside
    if 'k' in castling_rights:
        tensor[15, :, :] = 1 # Black Kingside
    if 'q' in castling_rights:
        tensor[16, :, :] = 1 # Black Queenside

    # --- En Passant Target (more complex, requires mapping square to row/col) ---
    # ... (implementation for en passant)

    return tensor

# Example usage:
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
board_tensor = fen_to_tensor(fen)
print(board_tensor.shape)

def load_stockfish():
    system = platform.system()

    if system == "Windows":
        path = os.path.join("engines", "stockfish.exe")
    else:
        path = os.path.join("engines", "stockfish")  # macOS/Linux

    return Stockfish(
        path,
        depth=20,
        parameters={"Threads": 4, "Minimum Thinking Time": 30}
    )

def get_top_moves(fen, n=3):
    stockfish = load_stockfish()
    
    stockfish.update_engine_parameters({"MultiPV": n})

    stockfish.set_fen_position(fen)

    # Returns list of dicts:
    # [{'Move': ..., 'Centipawn': ...}, ...]
    info = stockfish.get_top_moves(n)

    return info

first_row = next(iter(dataset))
print(get_top_moves(first_row["fen"]))

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    # This gets called every time the model needs to make a move
    # Return a python-chess Move object that is a legal move for the current position

    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.1)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    move_weights = [random.random() for _ in legal_moves]
    total_weight = sum(move_weights)
    # Normalize so probabilities sum to 1
    move_probs = {
        move: weight / total_weight
        for move, weight in zip(legal_moves, move_weights)
    }
    ctx.logProbabilities(move_probs)

    return random.choices(legal_moves, weights=move_weights, k=1)[0]



@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
