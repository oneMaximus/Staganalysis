from pathlib import Path
from typing import Iterator, Optional, Tuple
import re
import hashlib

import numpy as np
from PIL import Image

from base_codec import BaseCodec
from helpers import (
    build_header, parse_header, bits_from_bytes, bytes_from_bits,
    xor_bytes, crc32, HEADER_LEN, FLAG_PLAINTEXT_CRC
)

# ===== Custom-LSB directives (parsed from the Key field) =====
# Supported (all optional, order-insensitive; separate with space or ';'):
#   channels=rgb(a)      -> subset of r,g,b,a
#   planes=0,1,2         -> bit-planes to use (0 = LSB)
#   shuffle=on|off       -> shuffle pixel/channel order (PRNG from user_key)
# Everything else in the key becomes user_key (used for XOR obfuscation and RNG seed).
_CHANNEL_MAP = {'r': 0, 'g': 1, 'b': 2, 'a': 3}
Region = Optional[Tuple[int, int, int, int]]  # (x,y,w,h)


def _parse_key_directives(key: str):
    if not key:
        return {'user_key': '', 'planes': None, 'channels': None, 'shuffle': False}

    parts = re.split(r'[;\s]+', key.strip())
    user_key_parts, planes, channels, shuffle = [], None, None, None

    for p in parts:
        if not p:
            continue
        pl = p.lower()
        if pl.startswith('planes='):
            nums = p.split('=', 1)[1].strip()
            if nums:
                planes = [int(x) for x in re.split(r'[,\s]+', nums) if x != '']
        elif pl.startswith('channels='):
            val = p.split('=', 1)[1].strip().lower()
            ch_list = [c for c in val if c in _CHANNEL_MAP]
            channels = [_CHANNEL_MAP[c] for c in ch_list]
        elif pl.startswith('shuffle='):
            val = p.split('=', 1)[1].strip().lower()
            shuffle = val in ('on', 'true', '1', 'yes', 'y')
        else:
            user_key_parts.append(p)

    return {
        'user_key': ' '.join(user_key_parts),
        'planes'  : planes,
        'channels': channels,
        'shuffle' : bool(shuffle) if shuffle is not None else False
    }


def _rng_from_key(user_key: str) -> np.random.Generator:
    if not user_key:
        return np.random.default_rng(0xC0FFEE)
    h = hashlib.sha256(user_key.encode('utf-8')).digest()
    seed = int.from_bytes(h[:8], 'big', signed=False)
    return np.random.default_rng(seed)


class ImageCodec(BaseCodec):
    codec_id = "image"
    pretty   = "Image (PNG/BMP/TIFF)"

    def _arr(self, img: Image.Image) -> np.ndarray:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)

    def accepts(self, path: Path) -> bool:
        # We’ll load JPEG for convenience but will NOT save stego as JPEG
        return path.suffix.lower() in {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}

    def _effective_params(self, arr: np.ndarray, bpc: int, key: str):
        """
        Compute selected channel indices, bit-plane indices, effective bpc, and RNG/shuffle.
        """
        H, W, C = arr.shape
        d = _parse_key_directives(key)

        # channels
        if d['channels']:
            chs = [c for c in d['channels'] if c < C]
            if not chs:
                chs = list(range(min(C, 3)))  # fallback to RGB
        else:
            chs = list(range(C))  # default: all available channels

        # planes
        if d['planes'] and len(d['planes']) > 0:
            planes = sorted({p for p in d['planes'] if 0 <= p <= 7})
            if not planes:
                planes = list(range(max(1, min(4, int(bpc)))))
        else:
            planes = list(range(max(1, min(4, int(bpc)))))  # default: 0..bpc-1

        effective_bpc = len(planes)
        rng = _rng_from_key(d['user_key'])
        return chs, planes, effective_bpc, d['user_key'], d['shuffle'], rng

    def capacity_bytes(self, carrier: Path, bpc: int) -> int:
        arr = self._arr(Image.open(carrier))
        H, W, C = arr.shape
        # We don’t know the key here, so assume default: all channels, planes 0..bpc-1
        effective_bpc = max(1, min(4, int(bpc)))
        return max(0, (H * W * C * effective_bpc) // 8 - HEADER_LEN)

    # ---------- helper: choose output suffix & save ----------
    @staticmethod
    def _choose_output_suffix(carrier_path: Path) -> str:
        ext = carrier_path.suffix.lower()
        if ext in (".png", ".bmp", ".tif", ".tiff"):
            return ext  # keep same lossless format
        # JPEG carriers: write PNG to avoid lossy recompress (and losing LSBs)
        return ".png"

    @staticmethod
    def _save_image(arr: np.ndarray, mode: str, out_path: Path):
        ext = out_path.suffix.lower()
        img = Image.fromarray(arr, mode=("RGBA" if arr.shape[2] == 4 else "RGB"))
        if ext == ".png":
            img.save(out_path)
        elif ext == ".bmp":
            img.save(out_path, format="BMP")
        elif ext in (".tif", ".tiff"):
            # LZW is lossless; you can switch to None if you prefer uncompressed
            img.save(out_path, format="TIFF", compression="tiff_lzw")
        else:
            # Fallback (shouldn't happen): save PNG
            img.save(out_path.with_suffix(".png"))

    # ---------------- region helpers ----------------
    @staticmethod
    def _crop_for_region(arr: np.ndarray, region: Region) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Return (view, (x,y,w,h)) where view is either full arr or a cropped view."""
        H, W, _ = arr.shape
        if not region:
            return arr, (0, 0, W, H)
        x, y, w, h = region
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > W or y + h > H:
            raise ValueError(f"Invalid region {region} for image {W}x{H}")
        return arr[y:y+h, x:x+w, :], (x, y, w, h)

    # ---------------- embed/extract with optional region ----------------
    def embed(self, carrier: Path, payload: bytes, out_path: Path, bpc: int, key: str,
              region: Region = None) -> dict:
        img = Image.open(carrier)
        arr = self._arr(img)
        H, W, C = arr.shape

        # Select view (full image or region)
        target, (rx, ry, rw, rh) = self._crop_for_region(arr, region)
        viewH, viewW = target.shape[:2]

        chs, planes, effective_bpc, user_key, do_shuffle, rng = self._effective_params(target, bpc, key)

        # Obfuscate payload, but header CRC covers PLAINTEXT (enforce key)
        obf = xor_bytes(payload, user_key)
        header = build_header(payload, 0, flags=FLAG_PLAINTEXT_CRC)
        total = header + obf

        cap_bytes = (viewH * viewW * len(chs) * effective_bpc) // 8
        if len(total) > cap_bytes:
            raise ValueError(
                f"Capacity too small in region: need {len(total)} B, have ~{cap_bytes} B at {effective_bpc} bpc across {len(chs)} channel(s)"
            )

        flat = target.reshape(-1, C).copy()
        num_pixels = flat.shape[0]
        pairs = np.array([(px, ch) for px in range(num_pixels) for ch in chs], dtype=np.int64)
        if do_shuffle:
            rng.shuffle(pairs)

        orig_sub = target.copy()
        bit_stream = bits_from_bytes(total)

        try:
            for idx in range(pairs.shape[0]):
                px, ch = pairs[idx]
                v = int(flat[px, ch])
                for p in planes:
                    bit = next(bit_stream)
                    v = (v & ~(1 << p) & 0xFF) | ((bit & 1) << p)
                flat[px, ch] = np.uint8(v)
        except StopIteration:
            out_sub = flat.reshape(viewH, viewW, C)
            # Write back into full image if region used
            if region:
                out_full = arr.copy()
                out_full[ry:ry+rh, rx:rx+rw, :] = out_sub
            else:
                out_full = out_sub

            # choose output suffix based on carrier
            out_suffix = self._choose_output_suffix(carrier)
            save_path = out_path.with_suffix(out_suffix)
            self._save_image(out_full, "RGBA" if out_full.shape[2] == 4 else "RGB", save_path)

            # change mask (only for the modified view)
            diff = (orig_sub ^ out_sub)
            mask = np.zeros((viewH, viewW), dtype=np.uint8)
            for p in planes:
                plane_mask = (diff & (1 << p)) != 0  # viewH x viewW x C
                if len(chs) < C:
                    filt = np.zeros_like(plane_mask)
                    filt[..., chs] = plane_mask[..., chs]
                    plane_mask = filt
                mask = np.where(np.any(plane_mask, axis=2), 255, mask)

            out_dict = {
                "out": save_path,
                "bytes_embedded": len(payload),
                "metric_label": "Payload size",
                "metric_value": f"{len(payload)} bytes",
                "steg": out_full,
                "orig": arr,
                "mask": mask if region else mask,  # mask is view-sized
                "region": region,
            }
            return out_dict

        raise RuntimeError("Unexpected: ran out of space after capacity check passed")

    def extract(self, stego: Path, bpc: int, key: str, region: Region = None) -> bytes:
        arr = self._arr(Image.open(stego))
        src, _ = self._crop_for_region(arr, region)

        H, W, C = src.shape
        flat = src.reshape(-1, C)

        chs, planes, effective_bpc, user_key, do_shuffle, rng = self._effective_params(src, bpc, key)

        num_pixels = flat.shape[0]
        pairs = np.array([(px, ch) for px in range(num_pixels) for ch in chs], dtype=np.int64)
        if do_shuffle:
            rng.shuffle(pairs)

        def reader() -> Iterator[int]:
            for px, ch in pairs:
                v = int(flat[px, ch])
                for p in planes:
                    yield (v >> p) & 1

        r = reader()
        header = bytes_from_bits(r, HEADER_LEN)
        ver, flags, length, check = parse_header(header)

        enc = bytes_from_bits(r, length)
        dec = xor_bytes(enc, user_key)

        if (flags & FLAG_PLAINTEXT_CRC) != 0:
            if crc32(dec) != check:
                raise ValueError("Checksum mismatch (wrong key/options, region, or corrupted carrier)")
            return dec

        if crc32(enc) == check:
            return dec
        if crc32(dec) == check:
            return dec

        raise ValueError("Checksum mismatch (wrong key/options, region, or corrupted carrier)")
