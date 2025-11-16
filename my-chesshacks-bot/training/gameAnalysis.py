import torch
from torch.utils.data import Dataset

import json
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import os

import ijson

def identifyGame(fen: str) -> str:
    OPENING_LOWER_BOUND = 28
    ENDGAME_UPPER_BOUND = 14
    pieces = 0
    board = (fen.split(" ")[0]).split("/")
    for row in board:
        pieces += sum([c.isalpha() for c in row])
    if pieces >= 28:
        return "opening"
    elif pieces <= 14:
        return "endgame"
    else:
        return "middlegame"

# if __name__ == "__main__":
#     test1 = "8/5pk1/2r5/3R2KP/8/8/8/8 b - - 4 58"
#     print(identifyGame(test1))
#     test2 = "r3rb1k/1ppq1ppb/2n4p/p6N/P4p2/1BPP1N1P/1P1Q1PP1/3RR1K1 b - - 1 22"
#     print(identifyGame(test2))
#     test3 = "3r1rk1/pq2b1p1/1p2pP2/n5BP/P2p2Q1/2P4K/3N1P2/R5R1 b - - 0 27"
#     print(identifyGame(test3))