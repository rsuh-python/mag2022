import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.model = nn.Sequential(
            nn.Conv2d(3, 10, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 3),
            nn.ReLU(),
            nn.MaxPool2d(5),
            nn.Flatten(),
            nn.Linear(8820, 1024),
            nn.ReLU(),
            nn.Linear(1024, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax())
        
    def forward(self, x):
        return self.model(x)

