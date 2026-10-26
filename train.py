import json
import os
import time

import torch
import torch.nn as nn

from data import get_loaders
from model import MLP, count_parameters

# ---------------- Hiperparametreler ----------------
SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
HIDDEN_SIZES = (256, 128)
# ----------------------------------------------------


def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        optimizer.zero_grad()              # 1. eski gradient'leri sıfırla
        outputs = model(images)            # 2. tahmin yap
        loss = loss_fn(outputs, labels)    # 3. hatayı ölç
        loss.backward()                    # 4. gradient'leri hesapla
        optimizer.step()                   # 5. düğmeleri çevir

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():                  # öğrenme yok, sadece ölçüm
        for images, labels in loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


if __name__ == "__main__":
    torch.manual_seed(SEED)   # rastgeleliği sabitle: her çalıştırmada aynı sonuç

    # test_loader'ı alıyoruz ama eğitim boyunca KULLANMIYORUZ
    train_loader, val_loader, test_loader = get_loaders(batch_size=BATCH_SIZE, seed=SEED)

    model = MLP(hidden_sizes=HIDDEN_SIZES)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Model: 784 -> {HIDDEN_SIZES} -> 10 | Parametre: {count_parameters(model)}")
    print(f"Batch size: {BATCH_SIZE} | Learning rate: {LEARNING_RATE} | Epoch: {EPOCHS}")

    # Eğitimden önceki durum
    val_loss, val_acc = evaluate(model, val_loader, loss_fn)
    print(f"Eğitimden önce -> val loss: {val_loss:.4f} | val acc: {val_acc:.4f}")
    print("-" * 75)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
              f"val loss: {val_loss:.4f}  acc: {val_acc:.4f}")

    training_time = time.time() - start_time
    print("-" * 75)
    print(f"Eğitim süresi: {training_time:.1f} saniye")

    # Sonuçları kaydet (grafikler ve test için bir sonraki adımda kullanacağız)
    os.makedirs("results", exist_ok=True)
    history["training_time"] = training_time
    with open("results/baseline_history.json", "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), "results/baseline_model.pt")
    print("Eğitim geçmişi ve model 'results' klasörüne kaydedildi.")