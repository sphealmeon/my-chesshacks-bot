from .utils import chess_manager, GameContext
from chess import Move
import torch
import numpy as np
from training.training import ChessValueNet, fen_to_tensor

# -----------------------------
# Load neural network
# -----------------------------
model = ChessValueNet()
model.load_state_dict(torch.load("training/model.pth", map_location="cpu"))
model.eval()

def board_to_tensor(board):
    """Convert python-chess board → model input tensor."""
    fen = board.fen()
    x = fen_to_tensor(fen)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    return x


# =============================
# Entrypoint: choose move
# =============================
@chess_manager.entrypoint
def choose_move(ctx: GameContext):

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available")

    move_scores = []

    # Evaluate all legal moves using the NN
    for move in legal_moves:
        ctx.board.push(move)

        x = board_to_tensor(ctx.board)
        with torch.no_grad():
            value = model(x).item()       # NN returns scalar [-1, 1] or similar

        ctx.board.pop()
        move_scores.append(value)

    # ---------------------------------
    # Convert NN evaluations to choice
    # ---------------------------------
    # Temperature softmax (prevents collapse)
    TEMP = 1.0
    scores = np.array(move_scores)
    scores = scores / TEMP

    # Numerically stable softmax
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / np.sum(exp_scores)

    # Convert to python-chess compatible mapping
    move_probs = {move: float(prob) for move, prob in zip(legal_moves, probs)}
    ctx.logProbabilities(move_probs)

    # Sample move using the probability distribution
    chosen_move = np.random.choice(legal_moves, p=probs)
    return chosen_move


# =============================
# Reset function
# =============================
@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Called at the start of every game.
    Some models (especially RNNs) carry hidden state.
    Your model is feed-forward, but we still ensure consistency.
    """
    
    # No hidden state to clear, but good practice:
    model.eval()      # Make sure model stays in eval mode
    torch.set_grad_enabled(False)

    # You can also log or set temperature here per game, if needed.
    return
