"""The three exact architectures specified on page 5 of the Lab 3 manual."""
import torch
from torch import nn


class MLP(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                         nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))


class CNN(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(49, 64)
        # Independent initializations; standard PyTorch defaults made explicit.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=128, dropout=0.1,
                activation="relu", batch_first=True, norm_first=False)
            for _ in range(2)])
        self.classifier = nn.Linear(64, 10)

    @staticmethod
    def patches(x):
        # Row-major patch grid: [B, channel, patch_row, patch_col, row, col].
        return x.unfold(2, 7, 7).unfold(3, 7, 7).permute(
            0, 2, 3, 1, 4, 5).reshape(x.shape[0], 16, 49)

    def forward(self, x):
        x = self.projection(self.patches(x))
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x.mean(dim=1))


MODELS = {"MLP": MLP, "CNN": CNN, "Attention": Attention}
EXPECTED_PARAMETERS = {"MLP": 235146, "CNN": 20490, "Attention": 70794}


def hand_costs():
    """Multiply/add FLOPs per image; excludes biases and elementwise operations."""
    return {
        "MLP": [("Linear 784->256", 2*784*256),
                ("Linear 256->128", 2*256*128),
                ("Linear 128->10", 2*128*10)],
        "CNN": [("Conv 1->16, 28x28", 2*3*3*1*16*28*28),
                ("Conv 16->32, 14x14", 2*3*3*16*32*14*14),
                ("Linear 1568->10", 2*1568*10)],
        "Attention": [("Patch projection 49->64, 16 tokens", 2*16*49*64)]
        + [(f"Encoder {layer}: {name}", value)
           for layer in (1, 2)
           for name, value in [
               ("QKV projections", 2*16*64*3*64),
               ("QK transpose", 2*16*16*64),
               ("Attention times V", 2*16*16*64),
               ("Output projection", 2*16*64*64),
               ("Feedforward 64->128", 2*16*64*128),
               ("Feedforward 128->64", 2*16*128*64)]]
        + [("Classifier 64->10", 2*64*10)]
    }
