import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden_sizes=(256, 128), input_size=784, num_classes=10, dropout=0.0):
        super().__init__()

        layers = [nn.Flatten()]          # [batch, 1, 28, 28] -> [batch, 784]
        prev_size = input_size

        # Her gizli katman için: Linear + ReLU (+ isteğe bağlı Dropout)
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = size

        # Çıkış katmanı: 10 sınıf, sonrasında ReLU yok
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = MLP()
    print(model)
    print("Toplam parametre sayısı:", count_parameters(model))

    model_with_dropout = MLP(dropout=0.2)
    print(model_with_dropout)