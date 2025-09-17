"""
stego_core.py - Core MP3 steganography encode/decode logic
"""

from utils import bytes_to_bits, bits_to_bytes, crc32
from mp3_parser import find_frames

MAGIC = b"SGMP"
VERSION = 1

def build_header(payload: bytes, fname: str, k: int) -> bytes:
    """Constructs the stego header (MAGIC, version, K, length, fname, crc)."""
    fname_bytes = fname.encode() if fname else b""
    header = bytearray()
    header += MAGIC
    header += VERSION.to_bytes(1, "big")
    header += k.to_bytes(1, "big")
    header += len(payload).to_bytes(4, "big")
    header += len(fname_bytes).to_bytes(1, "big")
    header += fname_bytes
    header += crc32(payload).to_bytes(4, "big")
    return bytes(header)

def encode_mp3(cover_bytes: bytes, payload: bytes, k: int, fname="hello.txt") -> bytes:
    """Embed payload bits into the ancillary bytes of an MP3 file."""
    header = build_header(payload, fname, k)
    bitstream = bytes_to_bits(header + payload)

    frames = find_frames(cover_bytes)
    anc_positions = [pos for f in frames for pos in f["ancillary"]]

    capacity_bits = len(anc_positions) * k
    if len(bitstream) > capacity_bits:
        raise ValueError("Not enough capacity in MP3 for this payload")

    stego = bytearray(cover_bytes)
    idx = 0
    for pos in anc_positions:
        if idx >= len(bitstream):
            break
        # Take k bits
        chunk_bits = bitstream[idx: idx+k]
        chunk_value = 0
        for b in chunk_bits:
            chunk_value = (chunk_value << 1) | b
        # Replace LSBs
        stego[pos] = (stego[pos] & ~((1 << k) - 1)) | chunk_value
        idx += k
    return bytes(stego)

def decode_mp3(stego_bytes: bytes) -> tuple[bytes, str]:
    """Extract hidden payload from stego MP3 file."""
    frames = find_frames(stego_bytes)
    anc_positions = [pos for f in frames for pos in f["ancillary"]]

    # Try all K values until MAGIC found
    for k in range(1, 9):
        bits = []
        for pos in anc_positions:
            val = stego_bytes[pos] & ((1 << k) - 1)
            for j in reversed(range(k)):
                bits.append((val >> j) & 1)
            if len(bits) >= 32:  # enough to check MAGIC
                break
        data = bits_to_bytes(bits)
        if data.startswith(MAGIC):
            found_k = k
            break
    else:
        raise ValueError("No stego header found")

    # Read full header
    all_bits = []
    for pos in anc_positions:
        val = stego_bytes[pos] & ((1 << found_k) - 1)
        for j in reversed(range(found_k)):
            all_bits.append((val >> j) & 1)

    raw = bits_to_bytes(all_bits)
    # Parse header
    payload_len = int.from_bytes(raw[6:10], "big")
    fname_len = raw[10]
    fname = raw[11:11+fname_len].decode(errors="ignore")
    crc_val = int.from_bytes(raw[11+fname_len:15+fname_len], "big")

    payload = raw[15+fname_len: 15+fname_len+payload_len]
    if crc32(payload) != crc_val:
        raise ValueError("CRC mismatch: corrupted payload")

    return payload, fname
