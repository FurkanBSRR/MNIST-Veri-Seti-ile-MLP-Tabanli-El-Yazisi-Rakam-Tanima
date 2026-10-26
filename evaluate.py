import json

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data import get_loaders
from model import MLP
from train import evaluate, HIDDEN_SIZES, BATCH_SIZE, SEED

# ---------------- 1) Grafikler ----------------
with open("results/baseline_history.json") as f:
    history = json.load(f)

epochs = range(1, len(history["train_loss"]) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, history["train_loss"], "o-", label="Train")
ax1.plot(epochs, history["val_loss"], "o-", label="Validation")
ax1.set_title("Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(epochs, history["train_acc"], "o-", label="Train")
ax2.plot(epochs, history["val_acc"], "o-", label="Validation")
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/baseline_curves.png", dpi=150)
print("Grafik kaydedildi: results/baseline_curves.png")

# ---------------- 2) Test (sadece bir kez) ----------------
model = MLP(hidden_sizes=HIDDEN_SIZES)
model.load_state_dict(torch.load("results/baseline_model.pt"))

_, _, test_loader = get_loaders(batch_size=BATCH_SIZE, seed=SEED)
loss_fn = nn.CrossEntropyLoss()

test_loss, test_acc = evaluate(model, test_loader, loss_fn)
print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

# ---------------- 3) Rakam bazında test doğruluğu ----------------
correct = torch.zeros(10)
total = torch.zeros(10)
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        preds = model(images).argmax(dim=1)
        for digit in range(10):
            mask = labels == digit
            total[digit] += mask.sum()
            correct[digit] += (preds[mask] == digit).sum()

print("\nRakam bazında test doğruluğu:")
per_digit = {}
for digit in range(10):
    acc = (correct[digit] / total[digit]).item()
    per_digit[digit] = acc
    print(f"  {digit}: {acc:.4f}  ({int(correct[digit])}/{int(total[digit])})")

# ---------------- 4) Özet tablo (rapor için) ----------------
summary = {
    "final_train_loss": history["train_loss"][-1],
    "final_val_loss": history["val_loss"][-1],
    "final_train_acc": history["train_acc"][-1],
    "final_val_acc": history["val_acc"][-1],
    "test_loss": test_loss,
    "test_acc": test_acc,
    "training_time_sec": history["training_time"],
    "per_digit_test_acc": per_digit,
}
with open("results/baseline_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nÖzet kaydedildi: results/baseline_summary.json")

plt.show()