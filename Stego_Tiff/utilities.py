"""
utilities.py
Helper functions and constants for TIFF LSB steganography.
Focus: raw pixel manipulation on 8-bit channels (L or RGB/RGBA).
"""

from __future__ import annotations

import hashlib
import random
import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image

# ---------- Header / Protocol ----------
# MAGIC + version + k + ext_len + ext + payload_len(4) + sha256(32)
MAGIC = b"STG2!"
VERSION = 1
MAX_EXT = 15  # store up to 15 chars of extension

@dataclass
class PayloadMeta:
    ext: str
    length: int
    sha256_hex: str
    k_bits: int  # number of LSBs used (stored in header for convenience)


def _sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def pack_header(payload: bytes, k_bits: int, ext: str) -> bytes:
    """
    Build the header placed before the payload.
    """
    if len(ext) > MAX_EXT:
        raise ValueError(f"File extension too long (>{MAX_EXT} chars).")
    ext_b = ext.encode("ascii", errors="ignore")

    header = bytearray()
    header += MAGIC                  # 5
    header += bytes([VERSION])       # 1
    header += bytes([k_bits])        # 1
    header += bytes([len(ext_b)])    # 1
    header += ext_b                  # <=15
    header += struct.pack(">I", len(payload))  # 4
    header += _sha256(payload)       # 32
    return bytes(header)


def unpack_header(raw: bytes) -> Tuple[PayloadMeta, int]:
    """
    Parse a header from raw bytes. Returns (meta, header_length).
    Raises ValueError if not valid (e.g., wrong key/start/k_bits).
    """
    i = 0
    if raw[i:i+5] != MAGIC:
        raise ValueError("Magic not found. Wrong key/start/LSBs or not a stego file.")
    i += 5

    version = raw[i]; i += 1
    if version != VERSION:
        raise ValueError(f"Unsupported version {version} (expected {VERSION}).")

    k_bits = raw[i]; i += 1
    ext_len = raw[i]; i += 1
    ext = raw[i:i+ext_len].decode("ascii", errors="ignore"); i += ext_len

    payload_len = struct.pack(">B", 0)  # just to keep type checkers calm
    payload_len = struct.unpack(">I", raw[i:i+4])[0]; i += 4
    sha = raw[i:i+32]; i += 32

    meta = PayloadMeta(ext=ext, length=payload_len, sha256_hex=sha.hex(), k_bits=k_bits)
    return meta, i


# ---------- Image I/O ----------
def load_tiff_bytes(path: str):
    """
    Load an image (TIFF preferred) as flat uint8 array.
    Returns: (flat, (w,h), channels, mode)
    """
    im = Image.open(path)
    if im.mode not in ("L", "RGB", "RGBA"):
        # Convert other modes (e.g., P, CMYK) to RGB so we always have 8-bit channels.
        im = im.convert("RGB")
    mode = im.mode
    w, h = im.size
    arr = np.array(im, dtype=np.uint8)

    if mode == "L":
        channels = 1
        flat = arr.reshape(-1)
    else:
        channels = arr.shape[2]
        flat = arr.reshape(-1)

    return flat, (w, h), channels, mode


def save_tiff_bytes(path: str, flat: np.ndarray, size: Tuple[int, int], channels: int, mode: str):
    """
    Save a flat uint8 array back to an image file (TIFF).
    """
    w, h = size
    if channels == 1:
        arr = flat.reshape(h, w)
    else:
        arr = flat.reshape(h, w, channels)
    im = Image.fromarray(arr, mode=mode)
    im.save(path, format="TIFF")


# ---------- Capacity & Indexing ----------
def capacity_bytes(num_cover_bytes: int, k_bits: int) -> int:
    """
    Maximum payload bytes that can be embedded with k LSBs per cover byte.
    """
    return (num_cover_bytes * k_bits) // 8


def pixel_to_offset(pixel_index: int, channels: int) -> int:
    """
    Convert a pixel index (0-based) to a byte offset in the flat cover array.
    """
    return pixel_index * channels


def cover_indices(length: int, start_offset: int, key: int) -> np.ndarray:
    """
    Build a deterministic permutation of cover indices from [start_offset, length),
    using the provided integer key as the PRNG seed.
    """
    if start_offset < 0 or start_offset >= length:
        return np.array([], dtype=np.int64)
    indices = np.arange(start_offset, length, dtype=np.int64)
    rnd = random.Random(int(key) & 0xFFFFFFFF)
    idx_list = indices.tolist()
    rnd.shuffle(idx_list)
    return np.array(idx_list, dtype=np.int64)


# ---------- Bit-level Embed / Extract ----------
def bytes_to_bits(b: bytes) -> np.ndarray:
    """
    Convert bytes -> bit array (LSB-first within each byte).
    """
    arr = np.frombuffer(b, dtype=np.uint8)
    bits = np.unpackbits(arr)  # MSB-first per byte
    bits = bits.reshape(-1, 8)[:, ::-1].reshape(-1)  # flip to LSB-first per byte
    return bits.astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Convert bit array (LSB-first within each byte) -> bytes.
    """
    nbits = len(bits)
    pad = (-nbits) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    m = bits.reshape(-1, 8)[:, ::-1].reshape(-1)  # back to MSB-first for packbits
    return np.packbits(m).tobytes()


def embed_bits_into_cover(cover: np.ndarray, bitstream: np.ndarray, k_bits: int, indices: np.ndarray) -> np.ndarray:
    """
    Embed bitstream (0/1) into cover LSBs at positions given by indices, using k_bits per byte.
    """
    if k_bits < 1 or k_bits > 8:
        raise ValueError("k_bits must be 1..8")
    out = cover.copy()
    num_bytes_needed = (len(bitstream) + k_bits - 1) // k_bits
    if num_bytes_needed > len(indices):
        raise ValueError("Not enough cover capacity for provided bitstream/indices.")

    mask = (0xFF ^ ((1 << k_bits) - 1))  # clear k LSBs
    ptr = 0
    for i in range(num_bytes_needed):
        b = 0
        for bitpos in range(k_bits):
            if ptr < len(bitstream) and bitstream[ptr] == 1:
                b |= (1 << bitpos)
            ptr += 1
        cov_idx = indices[i]
        out[cov_idx] = (out[cov_idx] & mask) | b
    return out


def extract_bits_from_cover(cover: np.ndarray, num_bits: int, k_bits: int, indices: np.ndarray) -> np.ndarray:
    """
    Extract num_bits from cover using k_bits per byte and the given indices order.
    Returns an array of 0/1 bits.
    """
    if k_bits < 1 or k_bits > 8:
        raise ValueError("k_bits must be 1..8")
    num_cover_bytes = (num_bits + k_bits - 1) // k_bits
    if num_cover_bytes > len(indices):
        raise ValueError("Not enough indices to extract the requested number of bits.")

    bits = np.zeros(num_bits, dtype=np.uint8)
    ptr = 0
    for i in range(num_cover_bytes):
        v = cover[indices[i]]
        for bitpos in range(k_bits):
            if ptr >= num_bits:
                break
            bits[ptr] = (v >> bitpos) & 1
            ptr += 1
    return bits


# ---------- Visualization Helpers ----------
def diff_mask(cover: np.ndarray, stego: np.ndarray, k_bits: int) -> np.ndarray:
    """
    Return a flat uint8 mask (0 or 255) where any of the k LSBs changed.
    """
    delta = (cover ^ stego) & ((1 << k_bits) - 1)
    return ((delta != 0).astype(np.uint8)) * 255


def mask_to_image(mask_flat: np.ndarray, size: Tuple[int, int]) -> Image.Image:
    """
    Convert a flat 0/255 mask (LSB changes) into a grayscale image.
    Works for both single-channel (L) and multi-channel (RGB/RGBA) covers.
    If multi-channel, we collapse channels to one plane (max over channels)
    so any changed channel lights up that pixel.
    """
    w, h = size
    total_pixels = w * h
    mask_flat = np.asarray(mask_flat, dtype=np.uint8)

    if mask_flat.size == total_pixels:
        # L (grayscale) cover: already per-pixel
        arr = mask_flat.reshape(h, w)
    else:
        # Multi-channel cover: collapse to per-pixel
        channels = mask_flat.size // total_pixels
        arr = mask_flat.reshape(h, w, channels).max(axis=2).astype(np.uint8)

    return Image.fromarray(arr, mode="L")


def lsb_plane_image(cover: np.ndarray, k_bits: int, size: Tuple[int, int]) -> Image.Image:
    """
    Visualize the combined k LSBs as a grayscale image (optional helper).
    """
    plane = cover & ((1 << k_bits) - 1)
    scale = 255 // ((1 << k_bits) - 1) if k_bits < 8 else 1
    vis = (plane * scale).astype(np.uint8)

    w, h = size
    if vis.size != w * h:
        # average over channels if present
        channels = vis.size // (w * h)
        vis = vis.reshape(h, w, channels).mean(axis=2).astype(np.uint8)
    else:
        vis = vis.reshape(h, w)
    return Image.fromarray(vis, mode="L")
