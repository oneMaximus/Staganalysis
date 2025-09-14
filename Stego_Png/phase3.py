# phase3.py — Phase 3 minimal packer (no manifest)
# - Covers from ./pngfolder/
# - Video from ./video_embed/ (first .mp4 by default; override with --video)
# - Chooses as many images as required (largest capacity first)
# - Saves stego images to ./StegPng/ as <cover_stem>__vpl1.png, __vpl2.png, ...
# - Always shows Original vs Stego previews (like Phase 1/2)

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List
from zlib import crc32
import argparse

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ---------- Per-image header ----------
MAGIC = b"STEG"
VERSION = 1
FLAG_NONE = 0
MIME_VIDEO = 2
HEADER_LEN = 25  # MAGIC(4)|VER(1)|FLAGS(1)|MIME(1)|TOTAL_LEN(4)|IDX(2)|CNT(2)|CHUNK_LEN(4)|CRC32(4)

def _u16(n: int) -> bytes: return n.to_bytes(2, "big")
def _u32(n: int) -> bytes: return n.to_bytes(4, "big")

def build_chunk_header(total_len: int, idx: int, cnt: int, chunk: bytes) -> bytes:
    return (
        MAGIC
        + bytes([VERSION, FLAG_NONE, MIME_VIDEO])
        + _u32(total_len)
        + _u16(idx)
        + _u16(cnt)
        + _u32(len(chunk))
        + _u32(crc32(chunk))
    )

# ---------- image helpers ----------
def _to_mode_array(img: Image.Image, use_alpha: bool) -> np.ndarray:
    mode = "RGBA" if use_alpha else "RGB"
    if img.mode != mode:
        img = img.convert(mode)
    return np.array(img, dtype=np.uint8)

def per_image_capacity_bytes(img: Image.Image, bpc: int, use_alpha: bool) -> int:
    arr = _to_mode_array(img, use_alpha)
    h, w, c = arr.shape
    total_bits = h * w * c * bpc
    return max(0, total_bits // 8 - HEADER_LEN)

def bits_from_bytes(data: bytes):
    for byte in data:
        for i in range(7, -1, -1):
            yield (byte >> i) & 1

def human(n: int) -> str:
    x = float(n)
    for unit in ["B","KB","MB","GB"]:
        if x < 1024 or unit == "GB":
            return f"{x:.2f} {unit}" if unit != "B" else f"{int(x)} {unit}"
        x /= 1024

def preview_pairs(pairs: list[tuple[Path, Path]]):
    """Show Original vs Stego side-by-side (up to the first 4 pairs)."""
    if not pairs:
        return
    rows = min(len(pairs), 4)
    fig = plt.figure(figsize=(10, 4 * rows))
    for i, (orig, stego) in enumerate(pairs[:4]):
        o = Image.open(orig).convert("RGB")
        s = Image.open(stego).convert("RGB")
        ax1 = fig.add_subplot(rows, 2, 2*i+1); ax1.imshow(o); ax1.set_title(f"Original: {orig.name}"); ax1.axis("off")
        ax2 = fig.add_subplot(rows, 2, 2*i+2); ax2.imshow(s); ax2.set_title(f"Stego: {stego.name}");   ax2.axis("off")
    plt.tight_layout(); plt.show()

# ---------- core embed (STRICT LSB) ----------
def embed_header_and_chunk(cover: Path, out_path: Path, header_plus_chunk: bytes, bpc: int, use_alpha: bool):
    img = Image.open(cover)
    arr = _to_mode_array(img, use_alpha)
    H, W, C = arr.shape
    flat = arr.reshape(-1, C).copy()

    cap_bytes = (H * W * C * bpc) // 8
    if len(header_plus_chunk) > cap_bytes:
        raise ValueError(f"{cover.name}: chunk too large. Capacity ≈ {human(cap_bytes)}; got {human(len(header_plus_chunk))}")

    bit_iter = bits_from_bytes(header_plus_chunk)
    try:
        for px in range(flat.shape[0]):
            for ch in range(C):
                v = int(flat[px, ch])  # Python int for safe bit ops
                for k in range(bpc):
                    bit = next(bit_iter)
                    mask = 0xFF ^ (1 << k)          # clear k-th bit (8-bit mask)
                    v = (v & mask) | ((bit & 1) << k)
                flat[px, ch] = np.uint8(v)          # back to uint8
    except StopIteration:
        out_arr = flat.reshape(H, W, C)
        mode = "RGBA" if use_alpha else "RGB"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out_arr, mode=mode).save(out_path)
        return

    raise RuntimeError("Unexpected: ran out of image space after capacity check passed.")

# ---------- scanning & choosing ----------
@dataclass
class CarrierInfo:
    path: Path
    width: int
    height: int
    capacity: int

def scan_covers(cover_dir: Path, bpc: int, use_alpha: bool) -> List[CarrierInfo]:
    pngs = sorted(p for p in cover_dir.glob("*.png") if p.is_file())
    infos: List[CarrierInfo] = []
    for p in pngs:
        with Image.open(p) as im:
            cap = per_image_capacity_bytes(im, bpc, use_alpha)
            if cap > 0:
                w, h = im.size
                infos.append(CarrierInfo(p, w, h, cap))
    infos.sort(key=lambda ci: ci.capacity, reverse=True)  # largest capacity first
    return infos

def find_first_mp4(video_dir: Path) -> Path | None:
    vids = sorted([p for p in video_dir.glob("*.mp4") if p.is_file()])
    return vids[0] if vids else None

# ---------- main pack ----------
def main():
    base = Path(__file__).resolve().parent
    cover_dir = base / "pngfolder"
    video_dir = base / "video_embed"
    out_dir   = base / "StegPng"

    ap = argparse.ArgumentParser(description="Phase 3: auto-select carriers, embed video, preview stego vs original (no manifest).")
    ap.add_argument("--video", help="Path to .mp4 (default: first in ./video_embed)")
    ap.add_argument("--bpc", type=int, default=1, help="Bits per channel (1..3 typical). Default: 1")
    ap.add_argument("--use-alpha", action="store_true", help="Use RGBA (4 channels). Default: RGB")
    ap.add_argument("--covers", default=str(cover_dir), help="Directory with cover PNGs (default: ./pngfolder)")
    ap.add_argument("--out",    default=str(out_dir),   help="Directory for stego PNGs (default: ./StegPng)")
    args = ap.parse_args()

    if not (1 <= args.bpc <= 3):
        raise SystemExit("--bpc must be 1..3")

    # pick video
    video_path = Path(args.video) if args.video else find_first_mp4(video_dir)
    if not video_path:
        raise SystemExit(f"No .mp4 found. Place a file in {video_dir} or pass --video.")
    payload = video_path.read_bytes()
    total_len = len(payload)

    # scan covers & choose as many as required
    covers = scan_covers(Path(args.covers), bpc=args.bpc, use_alpha=args.use_alpha)
    if not covers:
        raise SystemExit(f"No usable PNGs in {args.covers}")

    # accumulate until capacity >= video size
    chosen: List[CarrierInfo] = []
    acc = 0
    for ci in covers:
        chosen.append(ci)
        acc += ci.capacity
        if acc >= total_len:
            break

    print(f"Video: {video_path.name} ({human(total_len)})")
    total_cap = sum(ci.capacity for ci in chosen)
    print(f"Selected {len(chosen)} image(s) → total capacity {human(total_cap)} @ bpc={args.bpc}, use_alpha={args.use_alpha}")

    if total_cap < total_len:
        short = total_len - total_cap
        raise SystemExit(
            f"❌ Not enough capacity. Short by {human(short)}.\n"
            f"Add more/larger covers, increase --bpc, or use --use-alpha."
        )

    # slice and embed
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    chunks: List[bytes] = []
    cursor = 0
    for ci in chosen:
        take = min(ci.capacity, total_len - cursor)
        chunks.append(payload[cursor:cursor+take])
        cursor += take
        if cursor >= total_len:
            break

    out_pairs: list[tuple[Path, Path]] = []
    for i, (ci, chunk) in enumerate(zip(chosen, chunks), start=1):
        header = build_chunk_header(total_len, i-1, len(chunks), chunk)
        blob = header + chunk
        out_name = f"{ci.path.stem}__vpl{i}.png"
        out_path = out_dir / out_name
        print(f"  -> {ci.path.name} gets chunk {i}/{len(chunks)} ({human(len(chunk))}) → {out_name}")
        embed_header_and_chunk(ci.path, out_path, blob, bpc=args.bpc, use_alpha=args.use_alpha)
        out_pairs.append((ci.path, out_path))

    print(f"✅ Done. Wrote {len(chunks)} stego image(s) to {out_dir.resolve()}")

    # Always show preview (like Phase 1/2)
    try:
        preview_pairs(out_pairs)
    except Exception as e:
        print(f"(preview failed: {e})")

if __name__ == "__main__":
    main()
