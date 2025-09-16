# Phase 2 pipeline: embed specified BMP from bmp_embed/ into first BMP in bmpfolder/
# Output stego BMP goes to StegBmp/ (auto-created)
# Deps: Pillow, numpy, matplotlib

from pathlib import Path
from zlib import crc32
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

MAGIC = b"STEG"
VERSION = 1
HEADER_LEN = 15           # MAGIC(4)+VER(1)+FLAGS(1)+MIME(1)+LEN(4)+CRC(4)
FLAG_NONE = 0
MIME_TEXT = 0
MIME_IMAGE = 1

def build_header(payload: bytes, mime: int) -> bytes:
    length = len(payload).to_bytes(4, "big")
    checksum = crc32(payload).to_bytes(4, "big")
    return MAGIC + bytes([VERSION, FLAG_NONE, mime]) + length + checksum

def _to_mode_array(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)

def capacity_bytes(img: Image.Image, bpc: int) -> int:
    arr = _to_mode_array(img)
    h, w, c = arr.shape
    return (h * w * c * bpc) // 8 - HEADER_LEN

def _bits_from_bytes(data: bytes):
    for byte in data:
        for i in range(7, -1, -1):
            yield (byte >> i) & 1

def embed_image(cover: Path, payload: bytes, output: Path, bpc: int):
    img = Image.open(cover)
    arr = _to_mode_array(img)
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
                v = int(flat[px, ch])
                for k in range(bpc):
                    bit = next(bit_iter)
                    mask = 0xFF ^ (1 << k)
                    v = (v & mask) | ((bit & 1) << k)
                flat[px, ch] = np.uint8(v)
    except StopIteration:
        stego = flat.reshape(H, W, C)
        Image.fromarray(stego, mode="RGB").save(output)
        return
    raise RuntimeError("Unexpected: ran out of image space after capacity check passed.")

SETTINGS_TRY = [
    (1,), (2,), (3,),    # RGB with 1..3 LSBs
]

def main():
    base = Path(__file__).resolve().parent
    bmpfolder = base / "bmpfolder"
    bmp_embed = base / "bmp_embed"
    steg_out  = base / "StegBmp"

    import argparse
    parser = argparse.ArgumentParser(description="Phase 2 BMP stego")
    parser.add_argument("--payload", help="Filename of secret to embed from bmp_embed/")
    args = parser.parse_args()

    covers = sorted(bmpfolder.glob("*.bmp"))
    payloads = sorted(bmp_embed.glob("*"))

    if not covers:
        print("❌ No cover BMP found in bmpfolder/")
        return
    if not payloads:
        print("❌ No payload found in bmp_embed/")
        return

    cover = covers[0]
    if args.payload:
        chosen_payloads = [p for p in payloads if p.name == args.payload]
        if not chosen_payloads:
            print(f"❌ File {args.payload} not found in bmp_embed/")
            return
        payload_img = chosen_payloads[0]
    else:
        payload_img = payloads[0]
    payload_bytes = payload_img.read_bytes()

    out_path = steg_out / f"{cover.stem}__stegPhoto.bmp"

    print(f"[1] Using cover image:   {cover.name}")
    print(f"[2] Embedding payload:   {payload_img.name}  ({len(payload_bytes):,} bytes)")

    for (bpc,) in SETTINGS_TRY:
        cap = capacity_bytes(Image.open(cover), bpc=bpc)
        print(f"    - Trying bpc={bpc}, RGB → capacity ≈ {cap:,} bytes")
        if cap >= len(payload_bytes):
            print("[3] Embedding…")
            embed_image(cover, payload_bytes, out_path, bpc=bpc)
            print(f"[4] Saved stego image: {out_path.name}")
            print("[5] Preview window: Original vs Stego (close to continue)")

            orig_img = Image.open(cover).convert("RGB")
            steg_img = Image.open(out_path).convert("RGB")
            fig = plt.figure(figsize=(8, 4))
            ax1 = fig.add_subplot(1, 2, 1); ax1.imshow(orig_img); ax1.set_title("Original"); ax1.axis("off")
            ax2 = fig.add_subplot(1, 2, 2); ax2.imshow(steg_img); ax2.set_title("Stego");    ax2.axis("off")
            plt.tight_layout(); plt.show()

            print(f"[6] Done. Stego saved at:\n    {out_path.resolve()}")
            return

    best_cap = max(capacity_bytes(Image.open(cover), bpc=b[0]) for b in SETTINGS_TRY)
    short = len(payload_bytes) - best_cap
    print(f"❌ Payload too large for this cover even at bpc=3 RGB.")
    print(f"   Best capacity: {best_cap:,} bytes; payload: {len(payload_bytes):,} bytes; short by {short:,} bytes.")
    print("   Fixes: use a larger cover image, pick a smaller payload image, or move to Phase 3 multi-image packing.")

if __name__ == "__main__":
    main()