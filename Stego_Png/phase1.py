# Phase 1: PNG LSB steganography — minimal but solid
# Features:
# - Core LSB embed/extract helpers
# - CLI: embed/extract/inspect
# - Interactive pipeline: auto-pick first PNG in ./pngfolder, prompt for text,
#   embed, save as [name]__steg.png, and preview original vs stego
#
# Deps: Pillow, numpy, matplotlib (for preview)

from __future__ import annotations
import argparse
from pathlib import Path
from zlib import crc32

from PIL import Image
import numpy as np

# Only used in interactive preview (no seaborn, per your note)
import matplotlib.pyplot as plt

# ---------------------------
# Constants
# ---------------------------
MAGIC = b"STEG"       # 4 bytes
VERSION = 1           # 1 byte
HEADER_LEN = 14       # MAGIC(4) + VER(1) + FLAGS(1) + LEN(4) + CRC(4)
FLAG_NONE = 0         # Phase 1: no XOR/encryption in pipeline (keep it simple)


# ---------------------------
# Small helpers
# ---------------------------
def build_header(plaintext: bytes, flags: int = FLAG_NONE) -> bytes:
    """Header computed over plaintext for integrity."""
    length = len(plaintext).to_bytes(4, "big")
    checksum = crc32(plaintext).to_bytes(4, "big")
    return MAGIC + bytes([VERSION, flags]) + length + checksum


def parse_header(h: bytes) -> tuple[int, int, int, int]:
    if len(h) < HEADER_LEN or h[:4] != MAGIC:
        raise ValueError("Invalid or missing header.")
    version = h[4]
    flags = h[5]
    length = int.from_bytes(h[6:10], "big")
    checksum = int.from_bytes(h[10:14], "big")
    return version, flags, length, checksum


def _img_to_rgb(img: Image.Image) -> np.ndarray:
    """Always work with H x W x 3 uint8 RGB array."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def capacity_bytes(img: Image.Image) -> int:
    """1 LSB per R,G,B -> 3 bits/pixel -> floor(bits/8) bytes."""
    arr = _img_to_rgb(img)
    h, w, _ = arr.shape
    return (h * w * 3) // 8


def _bits_from_bytes(data: bytes):
    """Yield bits MSB->LSB for each byte."""
    for byte in data:
        for i in range(7, -1, -1):
            yield (byte >> i) & 1


def _bytes_from_bits(bits_iter, n_bytes: int) -> bytes:
    """Assemble exactly n_bytes from a bit iterator."""
    out = bytearray()
    acc = 0
    count = 0
    for _ in range(n_bytes * 8):
        b = next(bits_iter)
        acc = (acc << 1) | (b & 1)
        count += 1
        if count == 8:
            out.append(acc)
            acc = 0
            count = 0
    return bytes(out)


def default_stego_name(cover_path: Path) -> Path:
    """e.g., /path/cat.png -> /path/cat__steg.png"""
    return cover_path.with_name(f"{cover_path.stem}__steg.png")


def default_recovered_name(steg_path: Path) -> Path:
    """e.g., /path/cat__steg.png -> /path/cat__steg__recovered.bin"""
    return steg_path.with_name(f"{steg_path.stem}__recovered.bin")


# ---------------------------
# Core LSB ops (basic, sequential)
# ---------------------------
def embed(cover: Path, output: Path, payload: bytes) -> None:
    """
    Embed header+payload using 1 LSB per RGB channel, sequentially.
    No XOR/crypto in Phase 1 pipeline to keep it simple & transparent.
    """
    img = Image.open(cover)
    arr = _img_to_rgb(img)
    H, W, _ = arr.shape
    flat = arr.reshape(-1, 3).copy()

    header = build_header(payload, FLAG_NONE)
    total = header + payload

    cap = capacity_bytes(img)
    if len(total) > cap:
        raise ValueError(
            f"Payload too large. Capacity ≈ {cap - HEADER_LEN} bytes (excl. header); got {len(payload)}"
        )

    bits = _bits_from_bytes(total)
    for i in range(len(total) * 8):
        pidx, cidx = divmod(i, 3)
        bit = next(bits)
        flat[pidx, cidx] = (flat[pidx, cidx] & 0xFE) | bit  # set LSB to secret bit

    stego = flat.reshape(H, W, 3)
    Image.fromarray(stego, mode="RGB").save(output)


def extract(steg: Path) -> bytes:
    """
    Blind extract: read header (14 bytes) + payload by length.
    """
    img = Image.open(steg)
    arr = _img_to_rgb(img)
    flat = arr.reshape(-1, 3)

    # header
    header_bits = (flat[i // 3, i % 3] & 1 for i in range(HEADER_LEN * 8))
    header = _bytes_from_bits(header_bits, HEADER_LEN)
    version, flags, length, checksum = parse_header(header)

    # payload
    start = HEADER_LEN * 8
    payload_bits = (flat[(start + i) // 3, (start + i) % 3] & 1 for i in range(length * 8))
    payload = _bytes_from_bits(payload_bits, length)

    # integrity check
    from zlib import crc32 as _crc32
    if _crc32(payload) != checksum:
        raise ValueError("Checksum mismatch (file altered).")
    return payload


# ---------------------------
# Interactive Pipeline (your step-by-step flow)
# ---------------------------
def interactive_pipeline():
    """
    Phase 1 pipeline:
      1) Use first PNG in ./pngfolder as the cover
      2) Ask for a message
      3) Embed with 1 LSB / channel
      4) Save stego as StegPng/<cover_stem>__steg.png (auto-creates folder)
      5) Show Original vs Stego preview
      6) Print absolute path of saved file
    """
    base = Path(__file__).resolve().parent
    cover_dir = base / "pngfolder"
    out_dir = base / "StegPng"
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted(p for p in cover_dir.glob("*.png") if p.is_file())
    if not pngs:
        print("No PNG files found in ./pngfolder. Drop a cover image there and run again.")
        return

    cover = pngs[0]
    print(f"[1] Using cover PNG: {cover.name}")

    msg = input("[2] Enter the message to embed: ")
    payload = msg.encode()

    # Capacity check (incl. header)
    cap = capacity_bytes(Image.open(cover))
    if len(payload) + HEADER_LEN > cap:
        print(f"Message too long for this image.\nCapacity available: {cap - HEADER_LEN} bytes")
        return

    out_path = out_dir / f"{cover.stem}__stegtext.png"
    print("[3] Embedding with basic LSB (1 bit/channel)...")
    embed(cover, out_path, payload)

    print(f"[4] Saved stego image: {out_path.name}")
    print("[5] Opening preview (Original vs Stego). Close the window to continue.")

    # Preview side-by-side
    orig_img = Image.open(cover).convert("RGB")
    steg_img = Image.open(out_path).convert("RGB")
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1); ax1.imshow(orig_img); ax1.set_title("Original"); ax1.axis("off")
    ax2 = fig.add_subplot(1, 2, 2); ax2.imshow(steg_img); ax2.set_title("Stego");    ax2.axis("off")
    plt.tight_layout(); plt.show()

    print(f"[6] Done. Stego saved at:\n    {out_path.resolve()}")



# ---------------------------
# CLI
# ---------------------------
def _cmd_embed(a: argparse.Namespace):
    if not a.msg and not a.infile:
        raise SystemExit("Provide --msg or --infile")
    payload = a.msg.encode() if a.msg else Path(a.infile).read_bytes()
    embed(Path(a.cover), Path(a.output) if a.output else default_stego_name(Path(a.cover)), payload)
    print("✅ Embed complete.")

def _cmd_extract(a: argparse.Namespace):
    data = extract(Path(a.steg))
    if a.outfile:
        Path(a.outfile).write_bytes(data)
        print(f"✅ Wrote {len(data)} bytes to {a.outfile}")
    else:
        try:
            print(data.decode())
        except UnicodeDecodeError:
            print(f"(binary) {len(data)} bytes extracted; use --outfile to save.")

def _cmd_inspect(a: argparse.Namespace):
    img = Image.open(a.cover)
    cap = capacity_bytes(img)
    print(f"Capacity (incl. header): ~{cap} bytes")
    print(f"Recommended max payload: {cap - HEADER_LEN} bytes")

def _cmd_interactive(_a: argparse.Namespace):
    interactive_pipeline()

def main():
    p = argparse.ArgumentParser(description="steg1 — PNG LSB steganography (Phase 1)")
    sub = p.add_subparsers(dest="cmd")

    pe = sub.add_parser("embed", help="Embed message/file into a PNG")
    pe.add_argument("cover", help="Cover PNG path")
    pe.add_argument("--output", help="Output stego PNG path (defaults to [stem]__steg.png)")
    pe.add_argument("--msg", help="Inline message to embed")
    pe.add_argument("--infile", help="Path to file to embed")
    pe.set_defaults(func=_cmd_embed)

    px = sub.add_parser("extract", help="Extract from a stego PNG")
    px.add_argument("steg", help="Stego PNG path")
    px.add_argument("--outfile", help="Write extracted bytes to file")
    px.set_defaults(func=_cmd_extract)

    pi = sub.add_parser("inspect", help="Show image capacity")
    pi.add_argument("cover", help="Cover PNG path")
    pi.set_defaults(func=_cmd_inspect)

    pii = sub.add_parser("pipeline", help="Run interactive pipeline (grab first PNG in folder, prompt, preview)")
    pii.set_defaults(func=_cmd_interactive)

    # Parse args
    args = p.parse_args()

    if args.cmd:
        args.func(args)
    else:
        # Default: run interactive pipeline
        interactive_pipeline()


if __name__ == "__main__":
    main()
