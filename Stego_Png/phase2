# Phase 2 pipeline: embed first PNG from png_embed/ into first PNG in pngfolder/
# Output stego PNG goes to StegPng/ (auto-created)
# Deps: Pillow, numpy, matplotlib

from pathlib import Path
from zlib import crc32
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# --- Header constants (adds MIME byte vs Phase 1) ---
MAGIC = b"STEG"
VERSION = 1
HEADER_LEN = 15           # MAGIC(4)+VER(1)+FLAGS(1)+MIME(1)+LEN(4)+CRC(4)
FLAG_NONE = 0
MIME_TEXT = 0
MIME_IMAGE = 1

# ---------- helpers ----------
def build_header(payload: bytes, mime: int) -> bytes:
    length = len(payload).to_bytes(4, "big")
    checksum = crc32(payload).to_bytes(4, "big")
    return MAGIC + bytes([VERSION, FLAG_NONE, mime]) + length + checksum

def _img_to_rgb(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)

def capacity_bytes(img: Image.Image) -> int:
    arr = _img_to_rgb(img)
    h, w, _ = arr.shape
    return (h * w * 3) // 8  # 1 LSB per RGB channel

def _bits_from_bytes(data: bytes):
    for byte in data:
        for i in range(7, -1, -1):
            yield (byte >> i) & 1

# ---------- core embed ----------
def embed_image(cover: Path, payload: bytes, output: Path):
    img = Image.open(cover)
    arr = _img_to_rgb(img)
    H, W, _ = arr.shape
    flat = arr.reshape(-1, 3).copy()

    header = build_header(payload, MIME_IMAGE)
    total = header + payload

    cap = capacity_bytes(img)
    if len(total) > cap:
        raise ValueError(f"Payload too large. Capacity ≈ {cap-HEADER_LEN} bytes; got {len(payload)}")

    bits = _bits_from_bytes(total)
    for i in range(len(total) * 8):
        pidx, cidx = divmod(i, 3)
        flat[pidx, cidx] = (flat[pidx, cidx] & 0xFE) | next(bits)

    stego = flat.reshape(H, W, 3)
    output.parent.mkdir(parents=True, exist_ok=True)   # ensure StegPng/ exists
    Image.fromarray(stego, mode="RGB").save(output)

# ---------- pipeline ----------
def main():
    base = Path(__file__).resolve().parent
    pngfolder = base / "pngfolder"     # cover images live here
    png_embed = base / "png_embed"     # payload images live here
    steg_out  = base / "StegPng"       # save stego images here

    covers = sorted(pngfolder.glob("*.png"))
    payloads = sorted(png_embed.glob("*.png"))

    if not covers:
        print("❌ No cover PNG found in pngfolder/")
        return
    if not payloads:
        print("❌ No payload PNG found in png_embed/")
        return

    cover = covers[0]
    payload_img = payloads[0]
    payload_bytes = payload_img.read_bytes()

    out_path = steg_out / f"{cover.stem}__stegPhoto.png"

    print(f"[1] Using cover image: {cover.name}")
    print(f"[2] Embedding payload image: {payload_img.name}")
    print("[3] Embedding with LSB (1 bit/channel)...")

    embed_image(cover, payload_bytes, out_path)

    print(f"[4] Saved stego image: {out_path.name}")
    print(f"[5] Preview window: Original vs Stego (close to continue)")

    # Side-by-side preview
    orig_img = Image.open(cover).convert("RGB")
    steg_img = Image.open(out_path).convert("RGB")

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1); ax1.imshow(orig_img); ax1.set_title("Original"); ax1.axis("off")
    ax2 = fig.add_subplot(1, 2, 2); ax2.imshow(steg_img); ax2.set_title("Stego");    ax2.axis("off")
    plt.tight_layout(); plt.show()

    print(f"[6] Done. Stego saved at:\n    {out_path.resolve()}")

if __name__ == "__main__":
    main()
