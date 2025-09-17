"""
decode.py - CLI to extract payload from MP3
"""

import sys
from stego_core import decode_mp3

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python decode.py <o3.mp3> <secret.bin>")
        sys.exit(1)

    stego_path, out_path = sys.argv[1:3]

    with open(stego_path, "rb") as f:
        stego = f.read()

    payload, fname = decode_mp3(stego)

    with open(out_path, "wb") as f:
        f.write(payload)

    print(f"[+] Extracted payload → {out_path} (original filename: {fname})")
