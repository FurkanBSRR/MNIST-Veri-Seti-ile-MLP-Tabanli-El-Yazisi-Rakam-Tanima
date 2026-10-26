import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Ayarlar ----------------
INPUT_PATH = "my_digits/raw/ogrenci_no.jpg"
OUTPUT_DIR = "my_digits/processed"
MAX_WIDTH = 1500         # büyük fotoğrafları bu genişliğe küçült
NOISE_RATIO = 0.15       # en uzun parçanın %15'inden kısa olanlar gürültü sayılır
MIN_HEIGHT_RATIO = 0.4   # birleştirmeden sonra en uzun rakamın %40'ından kısalar atılır
THICKEN = 0              # rakamlar çok ince çıkarsa 1, 2 veya 3 dene
# -----------------------------------------


def load_and_resize(path):
    # 1) Gri tonlamalı oku
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {path}")

    # 2) Çok büyükse küçült
    if gray.shape[1] > MAX_WIDTH:
        scale = MAX_WIDTH / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray


def binarize(gray):
    # 3) Kağıt dokusunu yumuşat
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4) Otsu eşikleme + renkleri ters çevir: mürekkep beyaz (255), kağıt siyah (0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # İsteğe bağlı: çizgileri kalınlaştır
    if THICKEN > 0:
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=THICKEN)
    return binary


def find_digit_boxes(binary):
    # 5) Birbirine bağlı beyaz piksel gruplarını bul
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = binary.shape

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        touches_border = x == 0 or y == 0 or x + w >= img_w or y + h >= img_h
        if not touches_border:
            boxes.append([x, y, w, h])

    if not boxes:
        return []

    # Çok küçük lekeleri at
    max_h = max(b[3] for b in boxes)
    boxes = [b for b in boxes if b[3] >= NOISE_RATIO * max_h]

    # Soldan sağa sırala ve yatayda üst üste binen parçaları birleştir
    boxes.sort(key=lambda b: b[0])
    merged = [boxes[0]]
    for x, y, w, h in boxes[1:]:
        mx, my, mw, mh = merged[-1]
        overlap = (mx + mw) - x
        if overlap > 0.5 * min(w, mw):        # büyük ölçüde çakışıyor: aynı rakam
            nx, ny = min(mx, x), min(my, y)
            merged[-1] = [nx, ny, max(mx + mw, x + w) - nx, max(my + mh, y + h) - ny]
        else:
            merged.append([x, y, w, h])

    # Birleştirmeden sonra hâlâ çok kısa kalanları at
    max_h = max(b[3] for b in merged)
    return [b for b in merged if b[3] >= MIN_HEIGHT_RATIO * max_h]


def to_mnist_format(binary, box):
    x, y, w, h = box
    digit = binary[y:y + h, x:x + w]

    # 6a) En-boy oranını koruyarak 20x20 kutuya sığdır
    scale = 20.0 / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 6b) 28x28 siyah tuvalin ortasına yerleştir
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x0, y0 = (28 - new_w) // 2, (28 - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = digit

    # 6c) Ağırlık merkezini tuvalin ortasına kaydır
    total = canvas.sum()
    cy = (np.arange(28)[:, None] * canvas).sum() / total
    cx = (np.arange(28)[None, :] * canvas).sum() / total
    shift = np.float32([[1, 0, round(14 - cx)], [0, 1, round(14 - cy)]])
    canvas = cv2.warpAffine(canvas, shift, (28, 28))
    return canvas


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    gray = load_and_resize(INPUT_PATH)
    binary = binarize(gray)
    boxes = find_digit_boxes(binary)
    print(f"Bulunan rakam sayısı: {len(boxes)}")

    if not boxes:
        raise SystemExit("Hiç rakam bulunamadı. Fotoğrafı veya ayarları kontrol et.")

    # Önceki çalıştırmadan kalan dosyaları temizle
    for name in os.listdir(OUTPUT_DIR):
        if name.endswith(".png"):
            os.remove(os.path.join(OUTPUT_DIR, name))

    digits = []
    for i, box in enumerate(boxes):
        digit = to_mnist_format(binary, box)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"digit_{i:02d}.png"), digit)
        digits.append(digit)
    print(f"28x28 rakamlar kaydedildi: {OUTPUT_DIR}/")

    # ---------------- Görselleştirme ----------------
    boxed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(boxed, str(i), (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    n = len(digits)
    fig = plt.figure(figsize=(14, 9))

    ax = fig.add_subplot(3, 1, 1)
    ax.imshow(boxed)
    ax.set_title("1) Orijinal (gri) + bulunan rakamlar")
    ax.axis("off")

    ax = fig.add_subplot(3, 1, 2)
    ax.imshow(binary, cmap="gray")
    ax.set_title("2) Eşikleme + ters çevirme (MNIST gibi: siyah zemin, beyaz rakam)")
    ax.axis("off")

    for i, digit in enumerate(digits):
        ax = fig.add_subplot(3, n, 2 * n + i + 1)
        ax.imshow(digit, cmap="gray")
        ax.set_title(str(i))
        ax.axis("off")
    fig.text(0.5, 0.31, "3) 28x28 MNIST formatı", ha="center", fontsize=12)

    plt.tight_layout()
    plt.savefig("results/handwriting_preprocessing.png", dpi=150)
    print("Görselleştirme kaydedildi: results/handwriting_preprocessing.png")
    plt.show()