import json
import os

import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from data import MEAN, STD
from model import MLP
from train import HIDDEN_SIZES

# ---------------- Ayarlar ----------------
STUDENT_NUMBER = "23120205018"
DIGITS_DIR = "my_digits/processed"
MODEL_PATH = "results/baseline_model.pt"
# -----------------------------------------

# MNIST'e uyguladığımız dönüşümlerin AYNISI
transform = transforms.Compose([
    transforms.ToTensor(),                    # 0-255 -> 0-1
    transforms.Normalize((MEAN,), (STD,)),    # data.py'deki aynı değerler
])

# 1) Hazırlanmış 28x28 rakamları sırayla oku
files = sorted(f for f in os.listdir(DIGITS_DIR) if f.endswith(".png"))
if len(files) != len(STUDENT_NUMBER):
    raise SystemExit(f"{len(files)} görüntü bulundu ama numarada {len(STUDENT_NUMBER)} rakam var.")

images = [cv2.imread(os.path.join(DIGITS_DIR, f), cv2.IMREAD_GRAYSCALE) for f in files]
batch = torch.stack([transform(img) for img in images])   # şekil: [11, 1, 28, 28]

# 2) Eğitilmiş modeli yükle
model = MLP(hidden_sizes=HIDDEN_SIZES)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 3) Tahmin
with torch.no_grad():
    outputs = model(batch)
    probs = torch.softmax(outputs, dim=1)    # puanlar -> olasılıklar

top2_probs, top2_digits = probs.topk(2, dim=1)   # her rakam için en olası iki tahmin
preds = top2_digits[:, 0]

predicted_number = "".join(str(p.item()) for p in preds)
correct = sum(p == t for p, t in zip(predicted_number, STUDENT_NUMBER))
accuracy = correct / len(STUDENT_NUMBER)

print(f"Gerçek numara : {STUDENT_NUMBER}")
print(f"Tahmin        : {predicted_number}")
print(f"Doğru rakam   : {correct}/{len(STUDENT_NUMBER)}  ({accuracy:.1%})\n")

print("Sıra | Gerçek | Tahmin | Güven  | 2. tahmin")
details = []
for i, true_digit in enumerate(STUDENT_NUMBER):
    pred, p1 = top2_digits[i, 0].item(), top2_probs[i, 0].item()
    second, p2 = top2_digits[i, 1].item(), top2_probs[i, 1].item()
    mark = "✓" if str(pred) == true_digit else "✗"
    print(f" {i:2d}  |   {true_digit}    |   {pred}    | {p1:6.1%} | {second} ({p2:.1%})  {mark}")
    details.append({"index": i, "true": int(true_digit), "pred": pred, "confidence": p1,
                    "second": second, "second_confidence": p2})

# 4) Basit bir dağılım karşılaştırması: ortalama piksel parlaklığı
own_mean = (batch * STD + MEAN).mean().item()    # normalizasyonu geri alıp 0-1 ölçeğinde ortalama
print(f"\nOrtalama piksel parlaklığı -> MNIST: {MEAN:.4f} | Benim rakamlarım: {own_mean:.4f}")

# 5) Sonuçları kaydet
os.makedirs("results", exist_ok=True)
with open("results/handwriting_results.json", "w") as f:
    json.dump({"true": STUDENT_NUMBER, "predicted": predicted_number, "accuracy": accuracy,
               "own_mean_pixel": own_mean, "details": details}, f, indent=2)

# 6) Görselleştirme: yeşil = doğru, kırmızı = yanlış
n = len(images)
fig, axes = plt.subplots(1, n, figsize=(2 * n, 3))
for i, ax in enumerate(axes):
    ax.imshow(images[i], cmap="gray")
    pred, conf = preds[i].item(), top2_probs[i, 0].item()
    is_correct = str(pred) == STUDENT_NUMBER[i]
    ax.set_title(f"Gerçek: {STUDENT_NUMBER[i]}\nTahmin: {pred}\n%{conf * 100:.0f}",
                 color="green" if is_correct else "red", fontsize=11)
    ax.axis("off")
fig.suptitle(f"Gerçek: {STUDENT_NUMBER}  |  Tahmin: {predicted_number}  |  Doğruluk: {accuracy:.1%}",
             fontsize=13)
plt.tight_layout()
plt.savefig("results/handwriting_predictions.png", dpi=150)
print("\nGrafik kaydedildi: results/handwriting_predictions.png")
plt.show()