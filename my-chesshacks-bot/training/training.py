import torch
from torch.utils.data import Dataset

import json
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import os

import ijson

# import modal

# image = image = modal.Image.debian_slim().pip_install(open("../requirements.txt", "r").read().split('\n'))
# app = modal.App("chesshacks-training", image=image)

# @app.function() # Outsource to Modal
def fen_to_tensor(fen_string):
    parts = fen_string.split(' ')
    piece_placement, active_color, castling_rights = parts[0], parts[1], parts[2]

    tensor = np.zeros((17, 8, 8), dtype=np.float32)
    row, col = 0, 0
    piece_map = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,'p':6,'n':7,'b':8,'r':9,'q':10,'k':11}
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
    if active_color == 'w':
        tensor[12, :, :] = 1
    if 'K' in castling_rights:
        tensor[13, :, :] = 1
    if 'Q' in castling_rights:
        tensor[14, :, :] = 1
    if 'k' in castling_rights:
        tensor[15, :, :] = 1
    if 'q' in castling_rights:
        tensor[16, :, :] = 1
    return tensor

class PrecomputedChessDataset(Dataset):
    # @app.function() # Outsource to Modal
    def __init__(self, file_path="precomputed.jsonl"):
        self.data = []
        with open(file_path, "r") as f:
            parser = ijson.items(f, 'item')
            for line in parser:
                self.data.append(json.loads(line))

    # @app.function() # Outsource to Modal
    def __len__(self):
        return len(self.data)

    # @app.function() # Outsource to Modal
    def __getitem__(self, idx):
        row = self.data[idx]
        x = fen_to_tensor(row["fen"])
        y = row["cp"]
        gameStatus = row["gameStatus"]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        return x, y
    
class ChessValueNet(nn.Module):
    # @app.function() # Outsource to Modal
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(17, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(64*8*8, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    # @app.function() # Outsource to Modal
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# @app.function() # Outsource to Modal
def train(file_path="precomputed.jsonl", batch_size=32, epochs=1):
    ds = PrecomputedChessDataset(file_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = ChessValueNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    # MODIFY OPT HERE
    loss_fn = nn.MarginRankingLoss() # Changed to margin ranking to rank moves

    for epoch in range(epochs):
        for step, (x, y) in enumerate(dl):
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss {loss.item():.4f}")

    # ------------------------------
    # SAVE MODEL AT END OF TRAINING
    # ------------------------------
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_root, "model.pth")

    torch.save(model.state_dict(), model_path)

# @app.local_entrypoint() # Modal entrypoint
def main():
    train(batch_size=32)
