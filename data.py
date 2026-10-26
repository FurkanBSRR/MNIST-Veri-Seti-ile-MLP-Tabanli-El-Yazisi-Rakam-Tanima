import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# MNIST eğitim setinin ortalaması ve standart sapması
MEAN = 0.1307
STD = 0.3081


def get_loaders(batch_size=64, val_size=10000, seed=42):
    # Her görüntüye uygulanacak dönüşümler
    transform = transforms.Compose([
        transforms.ToTensor(),                     # 0-255 -> 0-1
        transforms.Normalize((MEAN,), (STD,)),     # ortalama 0 civarına çek
    ])

    full_train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    # 60.000'lik train kısmını 50.000 train + 10.000 validation olarak böl
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)   # her çalıştırmada aynı bölünme
    train_data, val_data = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# Bu blok sadece "python data.py" ile çalıştırınca devreye girer
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders()

    print("Train örnek sayısı:", len(train_loader.dataset))
    print("Validation örnek sayısı:", len(val_loader.dataset))
    print("Test örnek sayısı:", len(test_loader.dataset))
    print("Train batch sayısı:", len(train_loader))

    images, labels = next(iter(train_loader))
    print("Bir batch'teki görüntülerin şekli:", images.shape)
    print("Bir batch'teki etiketlerin şekli:", labels.shape)
    print("İlk 10 etiket:", labels[:10].tolist())
    print("Normalizasyon sonrası min:", round(images.min().item(), 4))
    print("Normalizasyon sonrası max:", round(images.max().item(), 4))

    # Train kümesinde her rakamdan kaç tane var?
    train_labels = train_loader.dataset.dataset.targets[train_loader.dataset.indices]
    print("Rakam dağılımı (0-9):", torch.bincount(train_labels).tolist())