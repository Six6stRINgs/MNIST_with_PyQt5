from torch import nn


class MLP(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # (batch_size, 1, 28, 28) -> (1, 784) (flatten)
        x = x.view(-1, 28 * 28)
        x = self.model(x)
        return x

    def __str__(self):
        return 'MLP'
