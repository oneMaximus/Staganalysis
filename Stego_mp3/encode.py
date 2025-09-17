"""
encode.py - CLI to embed payload into MP3
"""

import sys
from stego_core import encode_mp3

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python encode.py <10_sec.mp3> <hello.txt> <o3.mp3> [k=1-8]")
        sys.exit(1)

    cover_path, payload_path, out_path = sys.argv[1:4]
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    with open(cover_path, "rb") as f:
        cover = f.read()
    with open(payload_path, "rb") as f:
        payload = f.read()

    stego = encode_mp3(cover, payload, k, fname=payload_path)
    with open(out_path, "wb") as f:
        f.write(stego)

    print(f"[+] Payload embedded with {k} LSBs → {out_path}")
