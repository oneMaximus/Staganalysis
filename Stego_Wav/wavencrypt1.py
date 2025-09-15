#!/usr/bin/env python3
"""
wavencrypt.py — LSB steganography for WAV audio (text <-> .wav)

Features
- Encode: hide UTF-8 text inside 16-bit PCM WAV using 1..8 LSBs
- Decode: recover text using the same integer key
- Key-required permutation: PRNG (seeded by key) shuffles bit positions for both header and payload
- Simple keystream XOR (derived from the same key) over payload bytes prior to embedding
- Capacity check and friendly CLI

Typical folder layout
.
├── wavencrypt1.py
└── wav/
    ├── cover.wav        # your source/cover audio (16-bit PCM)
    └── stego.wav        # will be created on encode

Usage
  Encode (inline text):
    python wavencrypt.py encode --in wav/cover.wav --out wav/stego.wav --key 12345 --lsb 2 --text "hello world"

  Encode (from file):
    python wavencrypt.py encode --in wav/cover.wav --out wav/stego.wav --key 12345 --lsb 2 --text-file message.txt

  Decode (prints recovered text):
    python wavencrypt.py decode --in wav/stego.wav --key 12345

Notes
- Supports 16-bit PCM WAV (mono or stereo). Other formats will abort with an error.
- The same key is REQUIRED for decoding (both for permutation and keystream).
- Header is also embedded using the same permuted positions, so guessing without the key is hard.
"""

import argparse
import random
import wave
import struct
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

MAGIC = b"WAVS"         # 4 bytes
VERSION = 1             # 1 byte
HEADER_LEN_BYTES = 4 + 1 + 1 + 4  # magic(4) + version(1) + n_lsb(1) + msg_len(4) = 10 bytes
HEADER_BITS = HEADER_LEN_BYTES * 8

def _read_wav(path: str):
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        comp_type = wf.getcomptype()
        comp_name = wf.getcompname()
        frames = wf.readframes(n_frames)

    if comp_type != 'NONE':
        raise ValueError(f"Unsupported WAV compression: {comp_type} ({comp_name})")
    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM WAV supported (sampwidth=2), got sampwidth={sampwidth}")

    # Unpack as little-endian 16-bit signed shorts
    sample_count = n_frames * n_channels
    fmt = "<" + "h" * sample_count
    samples = list(struct.unpack(fmt, frames))
    params = {
        "n_channels": n_channels,
        "sampwidth": sampwidth,
        "framerate": framerate,
        "n_frames": n_frames,
        "comptype": comp_type,
        "compname": comp_name,
    }
    return samples, params

def _write_wav(path: str, samples: List[int], params: dict):
    n_channels = params["n_channels"]
    framerate = params["framerate"]
    sampwidth = params["sampwidth"]
    comp_type = params["comptype"]
    comp_name = params["compname"]

    assert sampwidth == 2, "Internal error: sampwidth must be 2"
    # Ensure values are in int16 range
    clipped = [max(-32768, min(32767, int(x))) for x in samples]
    frames = struct.pack("<" + "h"*len(clipped), *clipped)
    n_frames = len(clipped) // n_channels

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.setcomptype(comp_type, comp_name)
        wf.writeframes(frames)

def _bytes_to_bits(data: bytes) -> List[int]:
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return bits

def _bits_to_bytes(bits: List[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("Bitstream length must be multiple of 8")
    out = bytearray()
    for i in range(0, len(bits), 8):
        b = 0
        for j in range(8):
            b = (b << 1) | (bits[i + j] & 1)
        out.append(b)
    return bytes(out)

def _keystream_bytes(length: int, rng: random.Random) -> bytes:
    # Simple keystream derived from RNG; one byte per payload byte
    return bytes(rng.getrandbits(8) for _ in range(length))

def _xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes((x ^ y) for x, y in zip(a, b))

def _perm_positions(total_samples: int, n_lsb: int, key: int) -> List[int]:
    """
    Build a list of permuted bit positions (length = total_samples * n_lsb).
    Each integer p in [0, total_positions) maps to (sample_idx, bit_slot),
    via: sample_idx = p // n_lsb, bit_slot = p % n_lsb.
    """
    total_positions = total_samples * n_lsb
    rng = random.Random(key)
    positions = list(range(total_positions))
    rng.shuffle(positions)
    return positions

def _get_slot(pos: int, n_lsb: int) -> Tuple[int, int]:
    """ Map a position index to (sample_index, bit_slot_index). """
    sample_idx = pos // n_lsb
    bit_slot = pos % n_lsb
    return sample_idx, bit_slot

def _set_bit(value: int, bit_index: int, bit: int) -> int:
    """
    Set bit_index (0 = LSB) of 16-bit signed integer 'value' to 'bit' (0/1).
    """
    # Convert to unsigned 16-bit space
    u = value & 0xFFFF
    mask = 1 << bit_index
    if bit:
        u |= mask
    else:
        u &= ~mask
    # Back to signed
    if u >= 0x8000:
        return u - 0x10000
    return u

def _get_bit(value: int, bit_index: int) -> int:
    u = value & 0xFFFF
    return (u >> bit_index) & 1

def _pack_header(n_lsb: int, msg_len_bytes: int) -> bytes:
    if not (1 <= n_lsb <= 8):
        raise ValueError("n_lsb must be in 1..8")
    return MAGIC + bytes([VERSION, n_lsb]) + struct.pack("<I", msg_len_bytes)

def _unpack_header(header: bytes) -> Tuple[int, int]:
    if len(header) != HEADER_LEN_BYTES:
        raise ValueError("Invalid header length")
    magic = header[:4]
    ver = header[4]
    n_lsb = header[5]
    msg_len = struct.unpack("<I", header[6:10])[0]
    if magic != MAGIC:
        raise ValueError("Magic mismatch (not a stego WAV made by this tool)")
    if ver != VERSION:
        raise ValueError(f"Unsupported version: {ver}")
    if not (1 <= n_lsb <= 8):
        raise ValueError("Corrupt header: n_lsb out of range")
    return n_lsb, msg_len

def encode(cover_wav: str, out_wav: str, key: int, n_lsb: int,
           text: Optional[str] = None, text_file: Optional[str] = None):
    if (text is None) == (text_file is None):
        raise ValueError("Provide exactly one of --text or --text-file")

    if text_file is not None:
        with open(text_file, "rb") as f:
            payload = f.read()
    else:
        payload = text.encode("utf-8")

    samples, params = _read_wav(cover_wav)
    total_samples = len(samples)

    header = _pack_header(n_lsb, len(payload))

    # Payload encryption (XOR with keystream derived from key)
    rng_stream = random.Random(key ^ 0xA5A5_5A5A)  # slight diversification
    ks = _keystream_bytes(len(payload), rng_stream)
    enc_payload = _xor_bytes(payload, ks)

    bitstream = _bytes_to_bits(header + enc_payload)

    positions = _perm_positions(total_samples, n_lsb, key)
    capacity_bits = len(positions)

    if len(bitstream) > capacity_bits:
        raise ValueError(
            f"Payload too large. Need {len(bitstream)} bits, capacity is {capacity_bits} bits "
            f"({total_samples} samples * {n_lsb} LSBs)"
        )

    # Embed
    for i, bit in enumerate(bitstream):
        pos = positions[i]
        sample_idx, bit_slot = _get_slot(pos, n_lsb)
        samples[sample_idx] = _set_bit(samples[sample_idx], bit_slot, bit)

    _write_wav(out_wav, samples, params)

def decode(stego_wav: str, key: int) -> str:
    samples, params = _read_wav(stego_wav)
    total_samples = len(samples)

    # Try all n_lsb from 1..8 to find correct header (since we need n_lsb to build positions)
    for trial_lsb in range(1, 9):
        positions = _perm_positions(total_samples, trial_lsb, key)
        # Extract header bits
        if len(positions) < HEADER_BITS:
            continue  # impossible
        hdr_bits = []
        for i in range(HEADER_BITS):
            pos = positions[i]
            sample_idx, bit_slot = _get_slot(pos, trial_lsb)
            hdr_bits.append(_get_bit(samples[sample_idx], bit_slot))
        try:
            header = _bits_to_bytes(hdr_bits)
            n_lsb, msg_len = _unpack_header(header)
        except Exception:
            continue

        # If header parsed with this trial_lsb, we found the real n_lsb (it should equal trial_lsb)
        if n_lsb != trial_lsb:
            # false positive; continue trying
            continue

        # Now extract the rest of the payload bits
        total_bits_needed = HEADER_BITS + msg_len * 8
        if total_bits_needed > len(positions):
            raise ValueError("Corrupt stego or wrong key: claimed payload exceeds capacity")

        payload_bits = []
        for i in range(HEADER_BITS, total_bits_needed):
            pos = positions[i]
            sample_idx, bit_slot = _get_slot(pos, n_lsb)
            payload_bits.append(_get_bit(samples[sample_idx], bit_slot))

        enc_payload = _bits_to_bytes(payload_bits)

        # Decrypt payload with keystream
        rng_stream = random.Random(key ^ 0xA5A5_5A5A)
        ks = _keystream_bytes(len(enc_payload), rng_stream)
        payload = _xor_bytes(enc_payload, ks)

        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError:
            # If not valid UTF-8, return ISO-8859-1 as a fallback string
            return payload.decode("latin1")

    raise ValueError("Failed to find a valid header. Wrong key or not a valid stego WAV.")

def preview_audio_side_by_side(cover_path: str, stego_path: str, n_samples: int = 4000):
    """Show side-by-side waveform preview of cover vs stego audio."""
    import wave

    def read_wav(path):
        with wave.open(path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
        samples = np.frombuffer(frames, dtype='<i2')
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels)[:, 0]  # take left channel only
        return samples

    cover = read_wav(cover_path)
    stego = read_wav(stego_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax1.plot(cover[:n_samples])
    ax1.set_title("Cover audio (before encoding)")
    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("Amplitude")

    ax2.plot(stego[:n_samples])
    ax2.set_title("Stego audio (after encoding)")
    ax2.set_xlabel("Sample index")

    plt.tight_layout()
    plt.show()

def _wav_capacity_bytes(total_samples: int, n_lsb: int) -> int:
    """How many payload bytes can fit with given n_lsb (excludes header automatically)."""
    total_bits = total_samples * n_lsb
    usable_bits = max(0, total_bits - HEADER_BITS)  # leave room for header
    return usable_bits // 8

def interactive_pipeline():
    """
    Interactive flow (like phase1):
      1) Pick cover from ./wav (prefer wav/cover.wav; else first .wav)
      2) Ask for key, LSBs, and a message
      3) Encode to ./wav/stego.wav
      4) Show side-by-side waveform preview (cover vs stego)
      5) Print instructions to decode
    """
    base = Path(__file__).resolve().parent
    wav_dir = base / "cover"
    wav_dir_stego = base / "stego"
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_dir_stego.mkdir(parents=True, exist_ok=True)

    # choose cover: prefer wav/cover.wav, else first .wav in folder
    cover = wav_dir / "cover.wav"
    if not cover.exists():
        candidates = sorted([p for p in wav_dir.glob("*.wav")])
        if not candidates:
            print("❌ No WAV found in ./wav. Put a 16-bit PCM WAV there (e.g., wav/cover.wav).")
            return
        cover = candidates[0]

    out_wav = wav_dir_stego / "stego.wav"
    print(f"[1] Using cover WAV: {cover.name}")

    # prompts (with safe defaults)
    try:
        key = int(input("[2] Enter integer key (default 12345): ") or "12345")
    except ValueError:
        print("Invalid key. Use an integer."); return

    try:
        n_lsb = int(input("[3] LSBs per sample 1..8 (default 2): ") or "2")
        if not (1 <= n_lsb <= 8):
            print("n_lsb must be between 1 and 8."); return
    except ValueError:
        print("Invalid LSB count."); return

    msg = input("[4] Enter the message to embed: ")
    if not msg:
        print("No message provided."); return

    # quick capacity info
    samples, _params = _read_wav(str(cover))
    cap = _wav_capacity_bytes(len(samples), n_lsb)
    need = len(msg.encode("utf-8"))
    if need > cap:
        print(f"❌ Message too long for this audio at --lsb {n_lsb}.")
        print(f"   Capacity ≈ {cap} bytes; message is {need} bytes.")
        print("   Fix: increase LSBs, use a longer WAV, or shorten the message.")
        return

    print("[5] Embedding…")
    encode(str(cover), str(out_wav), key, n_lsb, text=msg)
    print(f"[6] ✅ Encoded successfully into: {out_wav}")

    # preview
    try:
        print("[7] Opening preview (Cover vs Stego). Close the window to continue.")
        preview_audio_side_by_side(str(cover), str(out_wav))
    except Exception as e:
        print(f"(preview failed: {e})")

    print("\nTo decode later:")
    print(f"  python wavencrypt1.py decode --in {out_wav} --key {key}")

def _build_parser():
    p = argparse.ArgumentParser(description="LSB WAV steganography (text <-> .wav) with key-based permutation")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="Encode text into a WAV file")
    pe.add_argument("--in", dest="in_wav", required=True, help="Input cover WAV (16-bit PCM)")
    pe.add_argument("--out", dest="out_wav", required=True, help="Output stego WAV")
    pe.add_argument("--key", type=int, required=True, help="Integer key (required for both encode & decode)")
    pe.add_argument("--lsb", type=int, required=True, choices=range(1, 9), help="Number of LSBs to use (1-8)")
    g = pe.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="Inline text to hide (UTF-8)")
    g.add_argument("--text-file", type=str, help="Path to text file to hide (raw bytes)")

    pd = sub.add_parser("decode", help="Decode text from a stego WAV")
    pd.add_argument("--in", dest="in_wav", default="stego/stego.wav",
                help="Input stego WAV (default: stego/stego.wav)")
    pd.add_argument("--key", type=int, required=True, help="Integer key used during encoding")

    # NEW: interactive mode like phase1
    sub.add_parser("pipeline", help="Interactive prompts: pick files in ./wav, ask for key/LSB/message, preview")

    return p

def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "encode":
        encode(args.in_wav, args.out_wav, args.key, args.lsb,
               text=args.text, text_file=args.text_file)
        print(f"✅ Encoded successfully into: {args.out_wav}")
        try:
            preview_audio_side_by_side(args.in_wav, args.out_wav)
        except Exception as e:
            print(f"(preview failed: {e})")

    elif args.cmd == "decode":
        text = decode(args.in_wav, args.key)
        print(text)

    elif args.cmd == "pipeline":
        interactive_pipeline()

if __name__ == "__main__":
    main()


