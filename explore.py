import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1) Veri setini indir
train_data = datasets.MNIST(root="data", train=True, download=True,
                            transform=transforms.ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True,
                           transform=transforms.ToTensor())

# 2) Kaç tane görüntü var?
print("Train görüntü sayısı:", len(train_data))
print("Test görüntü sayısı:", len(test_data))

# 3) Tek bir örneği inceleyelim
image, label = train_data[0]
print("Görüntünün şekli:", image.shape)
print("Etiketi:", label)
print("En küçük piksel değeri:", image.min().item())
print("En büyük piksel değeri:", image.max().item())

# 4) İlk 10 görüntüyü çizelim
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
for i in range(10):
    img, lbl = train_data[i]
    axes[i].imshow(img.squeeze(), cmap="gray")
    axes[i].set_title(str(lbl))
    axes[i].axis("off")
plt.show()
torch.set_printoptions(linewidth=200)
print((image.squeeze() * 9).int())