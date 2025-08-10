import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassClassifier_(nn.Module):
    def __init__(
        self,
        classes_num: int,
        in_features: int,
        hidden_layers: list[int] = [256, 128, 64, 16],
    ) -> None:
        super().__init__()
        fp2, fp3, fp4, fp5 = hidden_layers
        self.fp1 = nn.Linear(in_features, fp2)
        self.bn1 = nn.BatchNorm1d(fp2)
        self.fp2 = nn.Linear(fp2, fp3)
        self.bn2 = nn.BatchNorm1d(fp3)
        self.fp3 = nn.Linear(fp3, fp4)
        self.bn3 = nn.BatchNorm1d(fp4)
        self.fp4 = nn.Linear(fp4, fp5)
        self.bn4 = nn.BatchNorm1d(fp5)
        self.fp5 = nn.Linear(fp5, classes_num)
        self.pool = nn.AvgPool1d(kernel_size=1, stride=1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fp1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fp2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fp3(x)))
        x = self.dropout(x)
        x = self.pool(self.bn4(self.fp4(x)))
        #         x = x.view(-1, 1)
        x = self.fp5(x)
        return x.squeeze(1)
