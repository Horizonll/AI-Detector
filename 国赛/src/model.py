import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class AIModel(nn.Module):
    def __init__(self, efficientnet_type="efficientnet-b0"):
        super(AIModel, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(
            efficientnet_type,
            num_classes=1,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.sigmoid(x)
        return x


class BaggingModel(nn.Module):
    def __init__(self, models):
        super(BaggingModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        output = torch.mean(outputs, dim=0)
        return output
