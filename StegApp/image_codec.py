# image_codec.py
from pathlib import Path
from typing import Iterator
import numpy as np
from PIL import Image

from base_codec import BaseCodec
from helpers import (
    build_header, parse_header, bits_from_bytes, bytes_from_bits,
    xor_bytes, crc32, HEADER_LEN
)

class ImageCodec(BaseCodec):
    codec_id = "image"
    pretty   = "Image (PNG/BMP/TIFF)"

    def _arr(self, img: Image.Image) -> np.ndarray:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)

    def accepts(self, path: Path) -> bool:
        return path.suffix.lower() in {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}

    def capacity_bytes(self, carrier: Path, bpc: int) -> int:
        arr = self._arr(Image.open(carrier))
        H, W, C = arr.shape
        return max(0, (H * W * C * bpc) // 8 - HEADER_LEN)

    def embed(self, carrier: Path, payload: bytes, out_path: Path, bpc: int, key: str) -> dict:
        img = Image.open(carrier)
        arr = self._arr(img); H, W, C = arr.shape
        flat = arr.reshape(-1, C).copy()

        obf = xor_bytes(payload, key)
        header = build_header(obf, 0)
        total = header + obf

        cap = (H * W * C * bpc) // 8
        if len(total) > cap:
            raise ValueError(f"Capacity too small: need {len(total)} B, have ~{cap} B at {bpc} bpc")

        orig = arr.copy()
        bits = bits_from_bytes(total)
        try:
            for px in range(flat.shape[0]):
                for ch in range(C):
                    v = int(flat[px, ch])
                    for k in range(bpc):
                        bit = next(bits)
                        v = (v & (0xFF ^ (1 << k))) | ((bit & 1) << k)
                    flat[px, ch] = np.uint8(v)
        except StopIteration:
            out = flat.reshape(H, W, C)
            Image.fromarray(out, mode=("RGBA" if arr.shape[2] == 4 else "RGB")).save(out_path.with_suffix(".png"))
            # bit-change mask (white = changed)
            diff = (orig ^ out)
            mask = np.zeros((H, W), dtype=np.uint8)
            for k in range(bpc):
                chg = (diff & (1 << k)) != 0
                pix = np.any(chg, axis=2)
                mask = np.where(pix, 255, mask)
            return {"orig": orig, "steg": out, "mask": mask, "out_path": out_path.with_suffix(".png")}

        raise RuntimeError("Unexpected: ran out of space after capacity check passed")

    def extract(self, stego: Path, bpc: int, key: str) -> bytes:
        arr = self._arr(Image.open(stego))
        H, W, C = arr.shape
        flat = arr.reshape(-1, C)

        def reader() -> Iterator[int]:
            for px in range(flat.shape[0]):
                for ch in range(C):
                    v = int(flat[px, ch])
                    for k in range(bpc):
                        yield (v >> k) & 1

        r = reader()
        header = bytes_from_bits(r, HEADER_LEN)
        ver, flags, length, check = parse_header(header)
        data = bytes_from_bits(r, length)
        data = xor_bytes(data, key)
        if crc32(data) != check:
            raise ValueError("Checksum mismatch (wrong key or corrupted carrier)")
        return data
