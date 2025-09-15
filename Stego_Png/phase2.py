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

def _to_mode_array(img: Image.Image, use_alpha: bool) -> np.ndarray:
    mode = "RGBA" if use_alpha else "RGB"
    if img.mode != mode:
        img = img.convert(mode)
    return np.array(img, dtype=np.uint8)

def capacity_bytes(img: Image.Image, bpc: int, use_alpha: bool) -> int:
    """Capacity with bpc LSBs/channel minus header."""
    arr = _to_mode_array(img, use_alpha)
    h, w, c = arr.shape
    return (h * w * c * bpc) // 8 - HEADER_LEN

def _bits_from_bytes(data: bytes):
    for byte in data:
        for i in range(7, -1, -1):
            yield (byte >> i) & 1

# ---------- core embed with configurable bpc & alpha ----------
def embed_image(cover: Path, payload: bytes, output: Path, bpc: int, use_alpha: bool):
    """
    Embed header+payload using bpc LSBs per channel.
    """
    img = Image.open(cover)
    arr = _to_mode_array(img, use_alpha)
    H, W, C = arr.shape
    flat = arr.reshape(-1, C).copy()

    header = build_header(payload, MIME_IMAGE)
    total = header + payload

    cap = (H * W * C * bpc) // 8
    if len(total) > cap:
        raise ValueError(f"Payload too large. Capacity ≈ {cap - HEADER_LEN} bytes; got {len(payload)}")

    bit_iter = _bits_from_bytes(total)
    try:
        for px in range(flat.shape[0]):
            for ch in range(C):
                v = int(flat[px, ch])  # do bit math on Python int
                for k in range(bpc):
                    bit = next(bit_iter)
                    mask = 0xFF ^ (1 << k)          # clear k-th LSB safely
                    v = (v & mask) | ((bit & 1) << k)
                flat[px, ch] = np.uint8(v)          # back to uint8
    except StopIteration:
        stego = flat.reshape(H, W, C)
        mode = "RGBA" if use_alpha else "RGB"
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(stego, mode=mode).save(output)
        return

    # Should never reach here because we pre-check capacity
    raise RuntimeError("Unexpected: ran out of image space after capacity check passed.")

# Try multiple capacity settings automatically
SETTINGS_TRY = [
    (1, False), (2, False), (3, False),  # RGB with 1..3 LSBs
    (1, True),  (2, True),  (3, True),   # RGBA with 1..3 LSBs
]

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

    print(f"[1] Using cover image:   {cover.name}")
    print(f"[2] Embedding payload:   {payload_img.name}  ({len(payload_bytes):,} bytes)")

    # Try combos until one fits
    for bpc, use_alpha in SETTINGS_TRY:
        cap = capacity_bytes(Image.open(cover), bpc=bpc, use_alpha=use_alpha)
        print(f"    - Trying bpc={bpc}, {'RGBA' if use_alpha else 'RGB'} → capacity ≈ {cap:,} bytes")
        if cap >= len(payload_bytes):
            print("[3] Embedding…")
            embed_image(cover, payload_bytes, out_path, bpc=bpc, use_alpha=use_alpha)
            print(f"[4] Saved stego image: {out_path.name}")
            print("[5] Preview window: Original vs Stego (close to continue)")

            # Side-by-side preview
            orig_img = Image.open(cover).convert("RGB")
            steg_img = Image.open(out_path).convert("RGB")
            fig = plt.figure(figsize=(8, 4))
            ax1 = fig.add_subplot(1, 2, 1); ax1.imshow(orig_img); ax1.set_title("Original"); ax1.axis("off")
            ax2 = fig.add_subplot(1, 2, 2); ax2.imshow(steg_img); ax2.set_title("Stego");    ax2.axis("off")
            plt.tight_layout(); plt.show()

            print(f"[6] Done. Stego saved at:\n    {out_path.resolve()}")
            return

    # If we’re here, none of the combos could fit the payload
    # Compute best capacity to report a friendly shortfall
    best_cap = max(capacity_bytes(Image.open(cover), bpc=b, use_alpha=a) for b, a in SETTINGS_TRY)
    short = len(payload_bytes) - best_cap
    print(f"❌ Payload too large for this cover even at bpc=3 RGBA.")
    print(f"   Best capacity: {best_cap:,} bytes; payload: {len(payload_bytes):,} bytes; short by {short:,} bytes.")
    print("   Fixes: use a larger cover image, pick a smaller payload image, or move to Phase 3 multi-image packing.")

if __name__ == "__main__":
    main()
