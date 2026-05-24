"""
GRU модель для генерации барабанов по мелодии (оптимизирована для CPU).

Вход:  (B, 1, 128, T)
Выход: (B, 1, 128, T)
"""

import torch
import torch.nn as nn


class LSTMAccompanimentModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, _, P, T = x.shape
        x = x.squeeze(1).permute(0, 2, 1)   # (B, T, 128)
        x = self.input_proj(x)               # (B, T, hidden)
        out, _ = self.gru(x)                 # (B, T, hidden*2)
        out = self.output_proj(out)          # (B, T, 128)
        return out.permute(0, 2, 1).unsqueeze(1)  # (B, 1, 128, T)


if __name__ == "__main__":
    model = LSTMAccompanimentModel()
    x = torch.randn(4, 1, 128, 128)
    y = model(x)
    print(f"Вход: {x.shape} -> Выход: {y.shape}")
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")