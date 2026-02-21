"""
DuelQNet architecture — matches the ViZDoom learning_pytorch.py example.
Used when loading state_dict-only checkpoints with TorchPolicy.
"""
import torch
import torch.nn as nn


class DuelQNet(nn.Module):
    """Dueling DQN (see https://arxiv.org/abs/1511.06581)."""

    def __init__(self, available_actions_count: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.state_fc = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage_fc = nn.Sequential(
            nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, available_actions_count)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 192)
        x1, x2 = x[:, :96], x[:, 96:]
        sv = self.state_fc(x1).reshape(-1, 1)
        av = self.advantage_fc(x2)
        return sv + (av - av.mean(dim=1, keepdim=True))
