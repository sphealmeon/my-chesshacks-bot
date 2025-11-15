from .utils import chess_manager, GameContext
from chess import Move
import torch
import numpy as np
from stockfish import Stockfish
import os
from training import ChessValueNet, fen_to_tensor

# -----------------------------
# Load neural network once
# -----------------------------
model = ChessValueNet()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# -----------------------------
# Load Stockfish once
# -----------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
stockfish_path = os.path.join(project_root, "engines", "stockfish")
if not os.path.exists(stockfish_path):
    raise FileNotFoundError(f"Stockfish not found at {stockfish_path}")
sf = Stockfish(stockfish_path)
sf.update_engine_parameters({"MultiPV": 1})

# -----------------------------
# Helper: board -> tensor
# -----------------------------
def board_to_tensor(board):
    fen = board.fen()
    x = fen_to_tensor(fen)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # add batch dim
    return x

# -----------------------------
# Entrypoint: choose move
# -----------------------------
@chess_manager.entrypoint
def choose_move(ctx: GameContext):
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    move_scores = []
    # Score moves using neural network
    for move in legal_moves:
        ctx.board.push(move)
        x = board_to_tensor(ctx.board)
        with torch.no_grad():
            score = model(x).item()
        move_scores.append(score)
        ctx.board.pop()

    # Convert scores to probabilities
    exp_scores = np.exp(move_scores - np.max(move_scores))
    move_probs = {move: s / exp_scores.sum() for move, s in zip(legal_moves, exp_scores)}
    ctx.logProbabilities(move_probs)

    # Pick move probabilistically based on NN evaluation
    best_move = np.random.choice(legal_moves, p=list(move_probs.values()))
    return best_move

# -----------------------------
# Reset function
# -----------------------------
@chess_manager.reset
def reset_func(ctx: GameContext):
    # Called at the start of a new game
    # Clear any caches or reset internal state if needed
    pass
