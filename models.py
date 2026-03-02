import torch
from torchvision.models import resnet18, ResNet18_Weights

def build_model(num_classes: int):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    model.conv1 = torch.nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )

    torch.nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')


    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)

    return model
