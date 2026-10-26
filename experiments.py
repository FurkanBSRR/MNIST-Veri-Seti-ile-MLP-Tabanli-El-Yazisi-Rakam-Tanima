import json
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data import get_loaders
from model import MLP, count_parameters
from train import train_one_epoch, evaluate

SEED = 42
BATCH_SIZE = 64
EPOCHS = 10

# Her deney baseline'dan sadece BİR ayarı değiştiriyor
EXPERIMENTS = [
    {"name": "baseline",  "hidden_sizes": (256, 128),      "lr": 0.001,  "optimizer": "adam", "dropout": 0.0},
    {"name": "lr_0.01",   "hidden_sizes": (256, 128),      "lr": 0.01,   "optimizer": "adam", "dropout": 0.0},
    {"name": "lr_0.0001", "hidden_sizes": (256, 128),      "lr": 0.0001, "optimizer": "adam", "dropout": 0.0},
    {"name": "small",     "hidden_sizes": (128, 64),       "lr": 0.001,  "optimizer": "adam", "dropout": 0.0},
    {"name": "deep",      "hidden_sizes": (512, 256, 128), "lr": 0.001,  "optimizer": "adam", "dropout": 0.0},
    {"name": "sgd",       "hidden_sizes": (256, 128),      "lr": 0.01,   "optimizer": "sgd",  "dropout": 0.0},
    {"name": "dropout",   "hidden_sizes": (256, 128),      "lr": 0.001,  "optimizer": "adam", "dropout": 0.2},
]


def run_experiment(config):
    torch.manual_seed(SEED)   # her deney aynı başlangıç koşullarıyla başlasın
    train_loader, val_loader, _ = get_loaders(batch_size=BATCH_SIZE, seed=SEED)   # test kullanılmıyor

    model = MLP(hidden_sizes=config["hidden_sizes"], dropout=config["dropout"])
    loss_fn = nn.CrossEntropyLoss()
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"  Epoch {epoch:2d}/{EPOCHS} | train loss: {train_loss:.4f}  acc: {train_acc:.4f} | "
              f"val loss: {val_loss:.4f}  acc: {val_acc:.4f}")

    history["training_time"] = time.time() - start_time
    history["parameters"] = count_parameters(model)
    history["config"] = {**config, "hidden_sizes": list(config["hidden_sizes"])}

    torch.save(model.state_dict(), f"results/experiments/{config['name']}.pt")
    return history


if __name__ == "__main__":
    os.makedirs("results/experiments", exist_ok=True)
    all_results = {}

    for i, config in enumerate(EXPERIMENTS, start=1):
        print(f"\n[{i}/{len(EXPERIMENTS)}] Deney: {config['name']}")
        all_results[config["name"]] = run_experiment(config)

        # Her deneyden sonra kaydet: yarıda kesilirse biten deneyler kaybolmasın
        with open("results/experiments/all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # ---------------- Özet tablo ----------------
    header = "| Deney | Parametre | Train acc | Val acc | Train loss | Val loss | Süre (sn) |"
    lines = [header, "|---|---|---|---|---|---|---|"]
    for name, h in all_results.items():
        lines.append(f"| {name} | {h['parameters']:,} | {h['train_acc'][-1]:.4f} | {h['val_acc'][-1]:.4f} | "
                     f"{h['train_loss'][-1]:.4f} | {h['val_loss'][-1]:.4f} | {h['training_time']:.1f} |")
    table = "\n".join(lines)
    print("\n" + table)
    with open("results/experiments/summary_table.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")

    # ---------------- Karşılaştırma grafikleri ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name, h in all_results.items():
        epochs = range(1, len(h["val_loss"]) + 1)
        ax1.plot(epochs, h["val_loss"], "o-", label=name)
        ax2.plot(epochs, h["val_acc"], "o-", label=name)

    ax1.set_title("Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("results/experiments/comparison.png", dpi=150)
    print("\nGrafik kaydedildi: results/experiments/comparison.png")
    plt.show()