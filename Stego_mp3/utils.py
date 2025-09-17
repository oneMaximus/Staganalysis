"""
utils.py - Helper functions for MP3 steganography
"""

import zlib

def bytes_to_bits(data: bytes) -> list[int]:
    """Convert a byte array into a list of bits (0/1)."""
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)  # MSB first
    return bits

def bits_to_bytes(bits: list[int]) -> bytes:
    """Convert a list of bits (0/1) into a byte array."""
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte = (byte << 1) | bits[i + j]
        out.append(byte)
    return bytes(out)

def crc32(data: bytes) -> int:
    """Return CRC32 checksum of given bytes."""
    return zlib.crc32(data) & 0xffffffff
