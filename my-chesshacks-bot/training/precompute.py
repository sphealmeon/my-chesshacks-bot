import json
from datasets import load_dataset
from stockfish import Stockfish
import os
import numpy as np

import modal
image = image = modal.Image.debian_slim().pip_install(open("../requirements.txt", "r").read().split('\n'))
app = modal.App("chesshacks-precompute", image=image)

@app.function() # Outsource to Modal
def load_stockfish():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stockfish_path = os.path.join(project_root, "engines", "stockfish")
    if not os.path.exists(stockfish_path):
        raise FileNotFoundError(f"Stockfish not found at {stockfish_path}")
    sf = Stockfish(stockfish_path)
    sf.update_engine_parameters({"MultiPV": 1})
    return sf

@app.function() # Outsource to Modal
def get_best_centipawn(fen, sf):
    sf.set_fen_position(fen)
    result = sf.get_top_moves(1)
    if not result or result[0]["Centipawn"] is None:
        return 0
    cp = result[0]["Centipawn"]
    if abs(cp) > 10000:
        cp = 10000 * np.sign(cp)
    return cp

# Precompute and save to reduce stockfish calls
@app.function() # Outsource to Modal
def precompute(output_file="precomputed.jsonl", max_rows=5000):
    dataset = load_dataset("jrahn/yolochess_lichess-elite_2211", split="train", streaming=True)
    sf = load_stockfish()
    
    count = 0
    with open(output_file, "w") as f:
        for row in dataset:
            fen = row["fen"]
            cp = get_best_centipawn(fen, sf)
            # optional: normalize / clamp
            cp = np.clip(cp, -1000, 1000) / 1000.0
            json_line = json.dumps({"fen": fen, "cp": cp})
            f.write(json_line + "\n")
            
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} positions")
            if count >= max_rows:
                break
    print(f"Finished {count} positions, saved to {output_file}")

@app.local_entrypoint() # Modal entrypoint
def main():
    precompute(max_rows=1000)  # adjust max_rows for your testing