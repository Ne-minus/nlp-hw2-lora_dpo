import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_layer, nn.Linear)

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Прокидываем и замораживаем веса с линейного слоя
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        # Матрицы для аппроксимации
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)

        nn.init.normal_(self.lora_A.weight, mean=0, std=math.sqrt(2.0 / self.in_features))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.dropout(self.lora_A(x))) * self.scaling
        return base_out + lora_out


