# stegano.py — engine: embed/decode + diff map support

from __future__ import annotations
import os
from typing import Dict

import numpy as np

from utilities import (
    load_tiff_bytes, save_tiff_bytes,
    capacity_bytes, pixel_to_offset, cover_indices,
    embed_bits_into_cover, extract_bits_from_cover,
    bytes_to_bits, bits_to_bytes,
    pack_header, unpack_header, _sha256,
    diff_mask, mask_to_image, lsb_plane_image,  # lsb_plane_image optional
    PayloadMeta
)


class SteganographyEngine:
    def get_capacity_report(self, cover_path: str, k_bits: int, start_pixel: int = 0) -> Dict[str, int]:
        flat, size, channels, mode = load_tiff_bytes(cover_path)
        start_offset = pixel_to_offset(start_pixel, channels)
        usable = max(0, len(flat) - start_offset)
        max_bytes = capacity_bytes(usable, k_bits)
        return {
            "total_cover_bytes": int(len(flat)),
            "width": int(size[0]),
            "height": int(size[1]),
            "channels": int(channels),
            "start_pixel": int(start_pixel),
            "start_offset": int(start_offset),
            "usable_cover_bytes": int(usable),
            "k_bits": int(k_bits),
            "max_payload_bytes": int(max_bytes),
        }

    def embed(self, cover_path: str, payload_path: str, out_path: str,
              k_bits: int, key: int, start_pixel: int = 0) -> Dict:
        # Load cover
        cover, size, channels, mode = load_tiff_bytes(cover_path)
        start_offset = pixel_to_offset(start_pixel, channels)
        if start_offset >= len(cover):
            raise ValueError("Start pixel is outside cover image.")

        # Read payload + header
        payload = open(payload_path, "rb").read()
        ext = os.path.splitext(payload_path)[1].lstrip(".")[:15]
        header = pack_header(payload, k_bits, ext)
        full = header + payload
        bits = bytes_to_bits(full)

        # Build deterministic cover index order from key
        idx = cover_indices(len(cover), start_offset, key)

        # Capacity check
        need_bytes = (len(bits) + k_bits - 1) // k_bits
        if need_bytes > len(idx):
            raise ValueError("Payload too large for this image / k / start position.")

        # Embed
        stego = embed_bits_into_cover(cover, bits, k_bits, idx)
        save_tiff_bytes(out_path, stego, size, channels, mode)

        # Diff mask for visualization
        dmask = diff_mask(cover, stego, k_bits)

        return {
            "out_path": out_path,
            "size": size,
            "channels": channels,
            "mode": mode,
            "bits_embedded": int(len(bits)),
            "cover_bytes_used": int(need_bytes),
            "diff_mask_flat": dmask,  # flat (0/255) vector
        }

    def decode(self, stego_path: str, k_bits: int, key: int, start_pixel: int = 0) -> Dict:
        stego, size, channels, mode = load_tiff_bytes(stego_path)
        start_offset = pixel_to_offset(start_pixel, channels)
        if start_offset >= len(stego):
            raise ValueError("Start pixel is outside image.")

        # Same permutation as during embed
        idx = cover_indices(len(stego), start_offset, key)

        # Read max header then parse to learn true header length
        prefix_bits = 59 * 8  # MAGIC(5)+ver(1)+k(1)+ext_len(1)+ext(<=15)+len(4)+sha(32) = up to 59 bytes
        bits_prefix = extract_bits_from_cover(stego, prefix_bits, k_bits, idx)
        prefix_bytes = bits_to_bytes(bits_prefix)
        meta, header_len = unpack_header(prefix_bytes)

        # Read exact header
        header_bits = header_len * 8
        bits_header = extract_bits_from_cover(stego, header_bits, k_bits, idx)
        header_bytes = bits_to_bytes(bits_header)
        meta, _ = unpack_header(header_bytes)

        total_bits = (header_len + meta.length) * 8
        bits_full = extract_bits_from_cover(stego, total_bits, k_bits, idx)
        full_bytes = bits_to_bytes(bits_full)
        payload = full_bytes[header_len: header_len + meta.length]

        # Verify integrity
        if _sha256(payload).hex() != meta.sha256_hex:
            raise ValueError("Checksum mismatch — wrong key/start/k_bits or corrupted stego.")

        return {
            "payload_bytes": payload,
            "meta": meta,
            "image_size": size,
            "channels": channels,
            "mode": mode,
            "start_pixel": start_pixel,
        }
