import json
import subprocess
from datasets import load_dataset
import os
import numpy as np
import chess
import shlex
import multiprocessing as mp


STOCKFISH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "engines",
    "stockfish"
)

DEPTH = 8
FEN_LIMIT = 500_000
OUT_PATH = "precomputed_stockfish.jsonl"
CHUNKSIZE = 50

SF = None


def init_worker():
    """
    Each worker launches a persistent Stockfish instance.
    Single-threaded Stockfish avoids macOS race conditions.
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

    # UCI init
    SF.stdin.write("uci\n")
    SF.stdin.flush()
    while True:
        line = SF.stdout.readline()
        if "uciok" in line:
            break

    # Force single thread
    SF.stdin.write("setoption name Threads value 1\n")
    SF.stdin.flush()


def worker_task(fen: str):
    """
    Evaluates one FEN using the worker's Stockfish engine.
    Returns evaluation in [-1, 1] from POV of side to move.
    """
    global SF
    if SF is None:
        init_worker()

    # Send position + eval command
    SF.stdin.write(f"position fen {fen}\n")
    SF.stdin.write(f"go depth {DEPTH}\n")
    SF.stdin.flush()

    last_cp = None

    while True:
        line = SF.stdout.readline()
        if not line:
            raise RuntimeError("Stockfish stdout closed unexpectedly")
        line = line.strip()

        # Extract centipawn score from "info" lines
        if "score cp" in line:
            try:
                parts = line.split()
                idx = parts.index("cp")
                last_cp = int(parts[idx + 1])
            except:
                pass

        # Evaluation ends at "bestmove"
        if line.startswith("bestmove"):
            cp = last_cp if last_cp is not None else 0

            # Clip extreme evaluations & normalize to [-1, 1]
            cp = max(-1000, min(1000, cp)) / 1000.0

            return {"fen": fen, "cp": cp}


def fen_generator(limit: int):
    """
    Streams FENs from the HuggingFace dataset.
    """
    ds = load_dataset("jrahn/yolochess_lichess-elite_2211", split="train", streaming=True)
    iterator = iter(ds)

    count = 0
    try:
        for row in iterator:
            fen = row.get("fen")
            if not fen:
                continue

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
        del iterator
        del ds


def main():
    mp.set_start_method("spawn", force=True)
    num_workers = max(1, mp.cpu_count())

    print("Using spawn start method.")
    print(f"Using {num_workers} workers. Writing to {OUT_PATH}")

    with mp.Pool(processes=num_workers, initializer=init_worker) as pool, open(OUT_PATH, "w") as out:
        idx = 0
        for result in pool.imap(worker_task, fen_generator(FEN_LIMIT), chunksize=CHUNKSIZE):
            out.write(json.dumps(result) + "\n")
            idx += 1

            if idx % 10_000 == 0:
                print(f"Processed {idx} FENs")

    print("Done.")


if __name__ == "__main__":
    main()
