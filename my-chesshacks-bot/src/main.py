from .utils import chess_manager, GameContext
from chess import Move
import random
import time

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

from datasets import load_dataset
import json

# Load the Lichess dataset in streaming mode
dataset = load_dataset("Lichess/standard-chess-games", split="train", streaming=True)

MAX_ROWS = 20_000_000  # Maximum number of games to process
count = 0
output_path = "filtered_1400plus.jsonl"  # Output file to store filtered games

columns_to_remove = ["UTCDate", "UTCTime"]  # Columns to prune

# Open the file for writing
with open(output_path, "w") as f:
    for row in dataset:
        if count >= MAX_ROWS:
            break  # Stop when reaching the row limit

        # Filter games where both players have Elo >= 1400
        white_elo = row.get("WhiteElo")
        black_elo = row.get("BlackElo")

        # Only keep rows where Elo is not None and >= 1400
        if white_elo is not None and black_elo is not None:
            if white_elo >= 1400 and black_elo >= 1400:
                pruned_row = {k: v for k, v in row.items() if k not in columns_to_remove}
                f.write(json.dumps(pruned_row) + "\n")

        count += 1

# Load the filtered dataset into memory for further use
new_dataset = load_dataset("json", data_files="filtered_1400plus.jsonl", split="train")

# Print first 5 entries to verify everything worked
for i in range(5):
    print(new_dataset[i])
    
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
