import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden_sizes=(256, 128), input_size=784, num_classes=10):
        super().__init__()

        layers = [nn.Flatten()]          # [batch, 1, 28, 28] -> [batch, 784]
        prev_size = input_size

        # Her gizli katman için: Linear + ReLU
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        # Çıkış katmanı: 10 sınıf, sonrasında ReLU yok
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from data import get_loaders

    model = MLP()
    print(model)
    print("Toplam parametre sayısı:", count_parameters(model))

    # Henüz eğitilmemiş modele bir batch verelim
    train_loader, _, _ = get_loaders()
    images, labels = next(iter(train_loader))

    outputs = model(images)
    print("Çıkışın şekli:", outputs.shape)
    print("İlk görüntü için 10 çıkış puanı:", outputs[0].tolist())

    predictions = outputs.argmax(dim=1)
    print("Tahminler :", predictions[:10].tolist())
    print("Gerçekler :", labels[:10].tolist())
    accuracy = (predictions == labels).float().mean().item()
    print("Eğitilmemiş modelin bu batch'teki doğruluğu:", accuracy)