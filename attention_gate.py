import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, channels):
        super(AttentionGate, self).__init__()
        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_layer1 = nn.Linear(channels, channels)
        self.linear_layer2 = nn.Linear(channels, channels)

    def forward(self, x):
        a, b, c, d = x.shape
        pooled = self.globalpool(x).view(a, b)
        weights = F.relu(self.linear_layer1(pooled))
        weights = torch.sigmoid(self.linear_layer2(weights))
        weights = weights.view(a, b, 1, 1)
        return x * weights# Element multiplication so weights are only multiplied to corresponding elements
# in this case channel information. (a,b) are multiplied ONLY to corresponding weights.
# this means spactial information (c,d) aren't affected and are preserved. (1*x=x) because Element multiplication
# Multiplies elements only with corresponding elements so 1*spactial_info=spactial_info preserved.