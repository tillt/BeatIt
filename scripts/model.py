import torch
import torch.nn as nn


class BeatNet(nn.Module):
    def __init__(self, mel_bins: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
        )
        reduced_bins = mel_bins // 4
        self.head = nn.Conv1d(64 * reduced_bins, 1, kernel_size=1)

    def forward(self, x):
        # x: (batch, frames, mel)
        x = x.unsqueeze(1).transpose(2, 3)  # (batch, 1, mel, frames)
        x = self.conv(x)
        b, c, m, t = x.shape
        x = x.reshape(b, c * m, t)
        beats = self.head(x).squeeze(1)
        beats = torch.sigmoid(beats)
        downbeats = torch.zeros_like(beats)
        return {"beat": beats, "downbeat": downbeats}
