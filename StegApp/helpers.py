# helpers.py
import hashlib
from zlib import crc32 as _crc32

# ----- constants -----
MAGIC = b"STEG"
VERSION = 1
FLAG_NONE = 0
HEADER_LEN = 15  # MAGIC(4) | VER(1) | FLAGS(1) | MIME(1) | LEN(4) | CRC32(4)

# ----- tiny utils -----
def u32(n: int) -> bytes:
    return n.to_bytes(4, "big")

def crc32(data: bytes) -> int:
    # Normalize to unsigned 32-bit
    return _crc32(data) & 0xFFFFFFFF

# Build a small header placed before the obfuscated payload.
# mime_byte is just a tag (0 for generic in our UI).
def build_header(payload: bytes, mime_byte: int, flags: int = FLAG_NONE) -> bytes:
    return (
        MAGIC
        + bytes([VERSION, flags, mime_byte])
        + u32(len(payload))
        + u32(crc32(payload))
    )

# Parse the fixed-length header
# Returns: (version, flags, length, checksum)
def parse_header(h: bytes):
    if len(h) < HEADER_LEN or h[:4] != MAGIC:
        raise ValueError("Invalid/missing header")
    ver   = h[4]
    flags = h[5]
    # mime_b = h[6]   # not needed by the UI, so we skip returning it
    length = int.from_bytes(h[7:11], "big")
    check  = int.from_bytes(h[11:15], "big")
    return ver, flags, length, check

# Bit helpers
def bits_from_bytes(buf: bytes):
    for b in buf:
        for i in range(7, -1, -1):
            yield (b >> i) & 1

def bytes_from_bits(bit_iter, n_bytes: int) -> bytes:
    out = bytearray()
    acc = 0
    cnt = 0
    for _ in range(n_bytes * 8):
        acc = (acc << 1) | (next(bit_iter) & 1)
        cnt += 1
        if cnt == 8:
            out.append(acc)
            acc = 0
            cnt = 0
    return bytes(out)

# Very lightweight XOR stream (for obfuscation, not crypto)
def derive_keystream(key: str, n: int) -> bytes:
    if not key:
        return b"\x00" * n
    out = bytearray()
    block = b""
    seed = key.encode()
    while len(out) < n:
        block = hashlib.sha256(block + seed).digest()
        out.extend(block)
    return bytes(out[:n])

def xor_bytes(data: bytes, key: str) -> bytes:
    ks = derive_keystream(key, len(data))
    return bytes(d ^ k for d, k in zip(data, ks))
