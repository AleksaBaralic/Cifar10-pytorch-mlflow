"""Arhitekture neuronskih mreza za CIFAR-10 (ulaz: 3x32x32, izlaz: 10 klasa)."""
import torch
import torch.nn as nn

NUM_CLASSES = 10


class MLP(nn.Module):
    """Baseline bez konvolucija: slika se spljosti u vektor."""

    def __init__(self, hidden1: int = 512, hidden2: int = 256,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """Mala CNN: 3 konvoluciona bloka (conv, ReLU, maxpool) + 2 linearna sloja."""

    def __init__(self, channels=(32, 64, 128), fc: int = 256,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        layers, in_ch = [], 3
        for ch in channels:
            layers += [nn.Conv2d(in_ch, ch, 3, padding=1), nn.ReLU(),
                       nn.MaxPool2d(2)]
            in_ch = ch
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 4 * 4, fc), nn.ReLU(),
            nn.Linear(fc, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNNBatchNorm(nn.Module):
    """Ista ideja kao SimpleCNN, ali sa BatchNorm i Dropout (regularizacija)."""

    def __init__(self, channels=(32, 64, 128), fc: int = 256,
                 dropout: float = 0.3, num_classes: int = NUM_CLASSES):
        super().__init__()
        layers, in_ch = [], 3
        for ch in channels:
            layers += [nn.Conv2d(in_ch, ch, 3, padding=1, bias=False),
                       nn.BatchNorm2d(ch), nn.ReLU(),
                       nn.MaxPool2d(2)]
            in_ch = ch
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch * 4 * 4, fc), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    """Rezidualni blok: dva conv sloja + skip veza."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class SmallResNet(nn.Module):
    """Mali ResNet: stem + 3 stupnja po n_blocks rezidualnih blokova."""

    def __init__(self, base: int = 32, n_blocks: int = 1,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base), nn.ReLU(),
        )
        stages, in_ch = [], base
        for i, mult in enumerate((1, 2, 4)):
            out_ch = base * mult
            stride = 1 if i == 0 else 2
            for b in range(n_blocks):
                stages.append(BasicBlock(in_ch, out_ch, stride if b == 0 else 1))
                in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.stages(self.stem(x))
        return self.fc(torch.flatten(self.pool(x), 1))


MODELS = {
    "mlp": MLP,
    "cnn": SimpleCNN,
    "cnn_bn": CNNBatchNorm,
    "resnet": SmallResNet,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Nepoznat model '{name}'. Dostupni: {list(MODELS)}")
    return MODELS[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """Velicina parametara i buffer-a u MB."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / (1024 ** 2)