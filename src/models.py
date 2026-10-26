import torch.nn as nn


# Activation functions we compare in the experiments
ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class BaselineCNN(nn.Module):
    ''' a small network (two conv layers followed by fully connected layers)'''

    def __init__(self, num_classes=78, dropout=0.0):
        super().__init__()

        # input 3x224x224 -> conv 5x5 -> 16x220x220 -> pool 4 -> 16x55x55
        self.conv1 = nn.Conv2d(3, 16, 5)

        # 16x55x55 -> conv 5x5 -> 32x51x51 -> pool 4 -> 32x12x12
        self.conv2 = nn.Conv2d(16, 32, 5)

        self.pool = nn.MaxPool2d(4, 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # flatten: channels x height x width = 32 x 12 x 12
        self.fc1 = nn.Linear(32 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    """ a deeper CNN that we can change through its arguments

    Each block is: Conv 3x3 -> (BatchNorm) -> Activation -> Pool
    Every block halves the image size, so 5 blocks: 224 -> 7
    """

    def __init__(self, num_classes=78, channels=(32, 64, 128, 256, 256),
                 activation="relu", batch_norm=False, pooling="max",
                 dropout=0.0):
        super().__init__()

        layers = []
        in_channels = 3

        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))

            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))

            layers.append(ACTIVATIONS[activation]())

            if pooling == "max":
                layers.append(nn.MaxPool2d(2))
            else:
                layers.append(nn.AvgPool2d(2))

            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Global average pooling: one number per feature map,
        # so the classifier needs far fewer weights than flattening
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Two conv layers plus a shortcut connection (the idea behind ResNet)
    The block learns F(x) and outputs activation(F(x) + x)"""

    def __init__(self, in_channels, out_channels, activation="relu"):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            ACTIVATIONS[activation](),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # 1x1 conv so the shortcut has the same shape as the main path
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2),
            nn.BatchNorm2d(out_channels)
        )

        self.activation = ACTIVATIONS[activation]()

    def forward(self, x):
        return self.activation(self.main(x) + self.shortcut(x))


class SmallResNet(nn.Module):
    """an advanced architecture built from residual blocks"""

    def __init__(self, num_classes=78, channels=(64, 128, 256),
                 activation="relu", dropout=0.0):
        super().__init__()

        # Stem: 3x224x224 -> 32x112x112 -> 32x56x56
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            ACTIVATIONS[activation](),
            nn.MaxPool2d(2)
        )

        # Each residual block halves the size: 56 -> 28 -> 14 -> 7
        blocks = []
        in_channels = 32
        for out_channels in channels:
            blocks.append(ResidualBlock(in_channels, out_channels, activation))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x


MODELS = {
    "baseline": BaselineCNN,
    "cnn": CNN,
    "resnet": SmallResNet,
}


def build_model(name, num_classes, dropout=0.0, **model_kwargs):
    return MODELS[name](num_classes=num_classes, dropout=dropout, **model_kwargs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
