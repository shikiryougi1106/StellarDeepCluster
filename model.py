import torch.nn as nn
from torchvision import models

class DeepCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeepCNN, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        return self.base_model(x)
