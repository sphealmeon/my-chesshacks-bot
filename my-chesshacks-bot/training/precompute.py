import json
import subprocess
from datasets import load_dataset
from stockfish import Stockfish
import os
import numpy as np
import chess

from gameAnalysis import identifyGame
import shlex
import multiprocessing as mp



STOCKFISH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "engines",
    "stockfish"
)

# Depth per Stockfish worker evaluation
DEPTH = 8   # or whatever depth you want

# Number of FENs to precompute 
FEN_LIMIT = 5000000   # adjust as needed

# Output file for the results
OUT_PATH = "precomputed_stockfish.jsonl"

# Size of chunks passed to pool.imap
CHUNKSIZE = 50

# Global Stockfish handle for each worker
SF = None



# Weight formula/algorithm for best move
def weight(fen: str, gameStatus: str) -> int:

    global SF

    board = chess.Board(fen) # To store moves and interact with
    SF.set_fen_position(fen)

    bestMoveInfo = SF.get_top_moves(1)[0] # Dict of info about best move according to Stockfish
    bestMove = chess.Move.from_uci(bestMoveInfo["Move"])
    cpInitial = bestMoveInfo["Centipawn"]
    if bestMoveInfo["Mate"]:
        return 1 # Maximize weight if move proceeds to checkmate

    board.push(bestMove) # Update board with best move
    SF.set_fen_position(board.fen()) # Update SF's board
    cpFinal = SF.get_top_moves(1)[0]["Centipawn"] # New centipawn
    if cpFinal == None:
        cpFinal = 0 # Must be stalemate

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

def precompute(output_file="precomputed.jsonl", max_rows=5000):
    dataset = load_dataset("jrahn/yolochess_lichess-elite_2211", split="train", streaming=True)
    
    count = 0
    with open(output_file, "w") as f:
        for row in dataset:
            fen = row["fen"]
            gameStatus = identifyGame(fen)
            cp = weight(fen, gameStatus)
            json_line = json.dumps({"fen": fen, "cp": cp})
            f.write(json_line + "\n")
            
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} positions")
            if count >= max_rows:
                break
    print(f"Finished {count} positions, saved to {output_file}")

def init_worker():
    """
    Runs once inside each worker process. Starts ONE persistent Stockfish subprocess
    and performs the minimal UCI handshake. We force Threads=1 to avoid macOS
    concurrency issues.
    """
    global SF
    SF = subprocess.Popen(
        shlex.split(STOCKFISH_PATH),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    # UCI handshake
    SF.stdin.write("uci\n")
    SF.stdin.flush()
    # wait for uciok
    while True:
        line = SF.stdout.readline()
        if not line:
            raise RuntimeError("Stockfish did not respond during UCI handshake")
        if "uciok" in line:
            break

    # Force single thread inside Stockfish (important on macOS)
    SF.stdin.write("setoption name Threads value 1\n")
    SF.stdin.flush()


def worker_task(fen: str):
    """
    Called for each fen. Reuses the per-worker SF subprocess.
    Returns a simple JSON-serializable dict.
    """
    global SF
    if SF is None:
        # defensive: should not happen if initializer runs
        init_worker()

    # send position + fast eval command
    SF.stdin.write(f"position fen {fen}\n")
    SF.stdin.write(f"go depth {DEPTH}\n")
    SF.stdin.flush()

    # parse output lines until bestmove (and grab latest CP when available)
    last_cp = None
    while True:
        line = SF.stdout.readline()
        if not line:
            # stockfish died or stdout closed unexpectedly
            raise RuntimeError("Stockfish stdout closed unexpectedly in worker")
        line = line.strip()
        # example info line contains: "info depth 1 score cp 23"
        if "score cp" in line:
            try:
                parts = line.split()
                idx = parts.index("cp")
                last_cp = int(parts[idx + 1])
            except Exception:
                pass
        if line.startswith("bestmove"):
            # if last_cp is None, fall back to 0 (safe default)
            cp = last_cp if last_cp is not None else 0
            return {"fen": fen, "cp": cp}


def fen_generator(limit: int):
    """
    Stream exactly one FEN string per dataset row.
    Explicitly cleans up HF iterator on exit to avoid 'Bad file descriptor' warnings.
    """
    ds = load_dataset("jrahn/yolochess_lichess-elite_2211", split="train", streaming=True)
    iterator = iter(ds)
    count = 0
    try:
        for row in iterator:
            # ensure we only take a single FEN string per row
            fen = row.get("fen")
            if not fen:
                continue

            # If the dataset ever contains a list unexpectedly, take the first entry
            if isinstance(fen, (list, tuple)):
                if not fen:
                    continue
                fen = fen[0]

            yield fen
            count += 1
            if count % 10_000 == 0:
                print(f"Streamed {count} FENs...")
            if count >= limit:
                print(f"Streamed {count} FENs — stopping.")
                return
    finally:
        # explicit cleanup to help HF close sockets/threads gracefully
        try:
            del iterator
            del ds
        except Exception:
            pass

def main():
    # set spawn (required on macOS)
    mp.set_start_method("spawn", force=True)

    num_workers = max(1, mp.cpu_count())
    print("Using spawn start method.")
    print(f"Using {num_workers} workers. Writing to {OUT_PATH}")

    # Open output file in main process and stream results as they arrive
    with mp.Pool(processes=num_workers, initializer=init_worker) as pool, open(OUT_PATH, "w") as out:
        idx = 0
        for result in pool.imap(worker_task, fen_generator(FEN_LIMIT), chunksize=CHUNKSIZE):
            out.write(json.dumps(result) + "\n")
            idx += 1
            if idx % 10000 == 0:
                print(f"Processed {idx} FENs")
    print("Done.")


if __name__ == "__main__":
    main()
