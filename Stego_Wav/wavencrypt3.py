#!/usr/bin/env python3
"""
wavencrypt3.py — Video <-> WAV LSB steganography (Phase 3)

- Hide a VIDEO file (MP4/MOV/AVI/MKV) inside a 16-bit PCM WAV using 1..8 LSBs per sample.
- Recover the embedded video with the SAME integer key.
- Key seeds a PRNG that permutes bit positions (used for header + payload).
- Header: magic, version, n_lsb, mime=video, payload length, crc32.
- Interactive "pipeline": auto-pick files in your folders and preview waveforms.

Folder layout expected (relative to this script):
project-root/
├─ video_embed/         # payload videos
├─ StegPng/             # recovered outputs (kept consistent with earlier phases)
└─ wav_encrypt/
   ├─ wavencrypt3.py
   └─ wav/
      ├─ cover.wav
      └─ stego.wav
"""

import argparse
import random
import wave
import struct
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from zlib import crc32

# ---------------------------
# Path setup (outside wav_encrypt for payload/output)
# ---------------------------
BASE = Path(__file__).resolve().parent               # .../wav_encrypt
DIR_WAV = BASE / "cover"                               # .../wav_encrypt/wav
DIR_VIDEO = BASE / "video_payload"              # .../video_embed
DIR_OUT = BASE / "stego"                    # .../StegPng
DIR_WAV.mkdir(parents=True, exist_ok=True)
DIR_VIDEO.mkdir(parents=True, exist_ok=True)
DIR_OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Constants
# ---------------------------
MAGIC = b"WVID"   # "WaV VIDeo"
VERSION = 1
MIME_VIDEO = 2

# magic(4) + version(1) + n_lsb(1) + mime(1) + length(4, LE) + crc32(4, LE) = 15 bytes
HEADER_LEN = 4 + 1 + 1 + 1 + 4 + 4
HEADER_BITS = HEADER_LEN * 8

# ---------------------------
# WAV helpers
# ---------------------------
def _read_wav(path: str):
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth  = wf.getsampwidth()
        framerate  = wf.getframerate()
        n_frames   = wf.getnframes()
        comp_type  = wf.getcomptype()
        comp_name  = wf.getcompname()
        frames     = wf.readframes(n_frames)

    if comp_type != 'NONE':
        raise ValueError(f"Unsupported WAV compression: {comp_type} ({comp_name})")
    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM WAV supported (sampwidth=2), got {sampwidth}")

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
    framerate  = params["framerate"]
    sampwidth  = params["sampwidth"]
    comp_type  = params["comptype"]
    comp_name  = params["compname"]

    assert sampwidth == 2, "Internal: sampwidth must be 2"
    clipped = [max(-32768, min(32767, int(x))) for x in samples]
    frames = struct.pack("<" + "h"*len(clipped), *clipped)

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.setcomptype(comp_type, comp_name)
        wf.writeframes(frames)

# ---------------------------
# Bit helpers
# ---------------------------
def _bytes_to_bits(data: bytes) -> List[int]:
    out = []
    for b in data:
        for i in range(8):
            out.append((b >> (7 - i)) & 1)
    return out

def _bits_to_bytes(bits: List[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("Bitstream length must be a multiple of 8")
    out = bytearray()
    for i in range(0, len(bits), 8):
        b = 0
        for j in range(8):
            b = (b << 1) | (bits[i + j] & 1)
        out.append(b)
    return bytes(out)

def _perm_positions(total_samples: int, n_lsb: int, key: int) -> List[int]:
    total_positions = total_samples * n_lsb
    rng = random.Random(key)
    pos = list(range(total_positions))
    rng.shuffle(pos)
    return pos

def _slot(pos: int, n_lsb: int) -> Tuple[int, int]:
    return pos // n_lsb, pos % n_lsb

def _set_bit(val: int, bit_idx: int, bit: int) -> int:
    u = val & 0xFFFF
    mask = 1 << bit_idx
    if bit:
        u |= mask
    else:
        u &= ~mask
    if u >= 0x8000:
        return u - 0x10000
    return u

def _get_bit(val: int, bit_idx: int) -> int:
    return ((val & 0xFFFF) >> bit_idx) & 1

# ---------------------------
# Header helpers
# ---------------------------
def _pack_header(n_lsb: int, payload: bytes) -> bytes:
    if not (1 <= n_lsb <= 8):
        raise ValueError("n_lsb must be 1..8")
    return MAGIC + bytes([VERSION, n_lsb, MIME_VIDEO]) \
           + struct.pack("<I", len(payload)) + struct.pack("<I", crc32(payload))

def _unpack_header(b: bytes) -> Tuple[int, int, int, int]:
    if len(b) != HEADER_LEN:
        raise ValueError("Bad header length")
    if b[:4] != MAGIC:
        raise ValueError("Magic mismatch")
    ver   = b[4]
    n_lsb = b[5]
    mime  = b[6]
    length = struct.unpack("<I", b[7:11])[0]
    csum   = struct.unpack("<I", b[11:15])[0]
    if ver != VERSION:
        raise ValueError("Version mismatch")
    if not (1 <= n_lsb <= 8):
        raise ValueError("n_lsb out of range")
    return n_lsb, mime, length, csum

# ---------------------------
# Preview helpers
# ---------------------------
def preview_waveforms(cover_path: str, stego_path: str, n_samples: int = 4000, n_lsb: int | None = None):
    import wave, numpy as np, matplotlib.pyplot as plt

    def read_for_plot(path):
        with wave.open(path, 'rb') as wf:
            ch = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
        arr = np.frombuffer(frames, dtype='<i2')
        if ch > 1:
            arr = arr.reshape(-1, ch)[:,0]
        return arr.astype(np.int32)

    c = read_for_plot(cover_path)
    s = read_for_plot(stego_path)
    n = min(n_samples, len(c), len(s))
    c, s = c[:n], s[:n]
    d = s - c

    # Auto y-limits: if n_lsb known, bound by +/- (2^n_lsb); else percentile-based
    if n_lsb is not None and 1 <= n_lsb <= 8:
        ylim = (- (2**n_lsb + 2), (2**n_lsb + 2))
    else:
        m = np.percentile(np.abs(np.concatenate([c, s, d])), 99.5)
        ylim = (-m, m)

    fig, axes = plt.subplots(1, 3, figsize=(14,4), sharey=True)
    axes[0].plot(c); axes[0].set_title("Cover (before)"); axes[0].set_xlabel("Sample"); axes[0].set_ylabel("Amplitude"); axes[0].set_ylim(*ylim)
    axes[1].plot(s); axes[1].set_title("Stego (after)");  axes[1].set_xlabel("Sample"); axes[1].set_ylim(*ylim)
    axes[2].plot(d); axes[2].set_title("Difference (stego - cover)"); axes[2].set_xlabel("Sample"); axes[2].set_ylim(*ylim)
    plt.tight_layout(); plt.show()

# ---------------------------
# Core encode/decode
# ---------------------------
def _guess_video_ext(data: bytes) -> str:
    # super-lightweight signature checks
    if b"ftyp" in data[:64]:  # MP4/MOV family
        return ".mp4"
    if data.startswith(b"\x1A\x45\xDF\xA3"):  # EBML -> MKV/WebM
        return ".mkv"
    if data.startswith(b"RIFF") and b"AVI" in data[:16]:
        return ".avi"
    return ".bin"

def encode(cover_wav: str, out_wav: str, key: int, n_lsb: int, video_path: str):
    payload = Path(video_path).read_bytes()
    samples, params = _read_wav(cover_wav)
    positions = _perm_positions(len(samples), n_lsb, key)

    total = _pack_header(n_lsb, payload) + payload
    bits  = _bytes_to_bits(total)

    if len(bits) > len(positions):
        need, cap = len(bits), len(positions)
        raise ValueError(f"Payload too large: need {need} bits, capacity {cap} bits "
                         f"({len(samples)} samples × {n_lsb} LSBs)")

    for i, bit in enumerate(bits):
        p = positions[i]
        sidx, bslot = _slot(p, n_lsb)
        samples[sidx] = _set_bit(samples[sidx], bslot, bit)

    _write_wav(out_wav, samples, params)

def decode(stego_wav: str, key: int) -> bytes:
    samples, _params = _read_wav(stego_wav)

    for trial in range(1, 9):
        pos = _perm_positions(len(samples), trial, key)
        if len(pos) < HEADER_BITS:
            continue
        hdr_bits = []
        for i in range(HEADER_BITS):
            sidx, bslot = _slot(pos[i], trial)
            hdr_bits.append(_get_bit(samples[sidx], bslot))
        try:
            header = _bits_to_bytes(hdr_bits)
            n_lsb, mime, length, csum = _unpack_header(header)
        except Exception:
            continue
        if n_lsb != trial or mime != MIME_VIDEO:
            continue
        total_bits = HEADER_BITS + length * 8
        if total_bits > len(pos):
            raise ValueError("Corrupt stego or wrong key: length exceeds capacity")

        data_bits = []
        for i in range(HEADER_BITS, total_bits):
            sidx, bslot = _slot(pos[i], n_lsb)
            data_bits.append(_get_bit(samples[sidx], bslot))
        payload = _bits_to_bytes(data_bits)
        if crc32(payload) != csum:
            raise ValueError("CRC mismatch (wrong key or corrupted file)")
        return payload

    raise ValueError("Failed to find a valid header. Wrong key or not a video-stego WAV.")

# ---------------------------
# Interactive pipeline
# ---------------------------
def _first_in(dir: Path, pats) -> Optional[Path]:
    for pat in pats:
        files = sorted(dir.glob(pat))
        if files:
            return files[0]
    return None

def _capacity_bytes(total_samples: int, n_lsb: int) -> int:
    total_bits = total_samples * n_lsb
    usable_bits = max(0, total_bits - HEADER_BITS)
    return usable_bits // 8

def interactive_pipeline():
    cover = DIR_WAV / "coverlong.wav"
    if not cover.exists():
        cover = _first_in(DIR_WAV, ("*.wav","*.WAV"))
        if cover is None:
            print("❌ No WAV found in ./cover. Place cover.wav or any 16-bit PCM WAV there.")
            return

    video = _first_in(DIR_VIDEO, ("*.mp4","*.mov","*.avi","*.mkv","*.MP4","*.MOV","*.AVI","*.MKV"))
    if video is None:
        print("❌ No video found in ../video_embed. Put an mp4/mov/avi/mkv there.")
        return

    print(f"[1] Cover WAV : {cover.name}")
    print(f"[2] Payload   : {video.name}")

    try:
        key = int(input("[3] Enter integer key (default 12345): ") or "12345")
    except ValueError:
        print("Invalid key. Use an integer."); return

    try:
        n_lsb = int(input("[4] LSBs per sample 1..8 (default 2): ") or "2")
        if not (1 <= n_lsb <= 8):
            print("n_lsb must be 1..8."); return
    except ValueError:
        print("Invalid LSB count."); return

    # Capacity check
    samples, _params = _read_wav(str(cover))
    cap = _capacity_bytes(len(samples), n_lsb)
    need = len(Path(video).read_bytes())
    if need > cap:
        print(f"❌ Video too large. Capacity ≈ {cap} bytes at LSB={n_lsb}; video is {need} bytes.")
        print("   Fix: increase LSBs, use a longer WAV, or a smaller/shorter video.")
        return

    out_wav = DIR_OUT / "stegoVid.wav"
    print("[5] Embedding video into WAV…")
    encode(str(cover), str(out_wav), key, n_lsb, str(video))
    print(f"[6] ✅ Encoded → {out_wav}")

    try:
        print("[7] Previewing waveforms (Cover vs Stego). Close the window to continue.")
        preview_waveforms(str(cover), str(out_wav))
    except Exception as e:
        print(f"(waveform preview failed: {e})")

    # Decode immediately for verification
    print("[8] Decoding to verify…")
    payload = decode(str(out_wav), key)
    ext = _guess_video_ext(payload)
    out_vid = DIR_OUT / f"recovered_video{ext}"
    out_vid.write_bytes(payload)
    print(f"[9] ✅ Recovered video → {out_vid}")
    print("    (Open it with your media player to confirm playback.)")

    print("\nTo run non-interactively:")
    print(f"  python wavencrypt3.py encode --in {cover} --out {out_wav} --key {key} --lsb {n_lsb} --video {video}")
    print(f"  python wavencrypt3.py decode --in {out_wav} --key {key} --out {out_vid}")

# ---------------------------
# CLI
# ---------------------------
def _build_parser():
    p = argparse.ArgumentParser(description="Phase 3: Hide a video inside a 16-bit PCM WAV (LSB, key permutation).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="Encode video into WAV")
    pe.add_argument("--in", dest="in_wav", required=True, help="Cover WAV (16-bit PCM)")
    pe.add_argument("--out", dest="out_wav", required=True, help="Output stego WAV")
    pe.add_argument("--key", type=int, required=True, help="Integer key")
    pe.add_argument("--lsb", type=int, choices=range(1,9), required=True, help="Number of LSBs to use (1..8)")
    pe.add_argument("--video", required=True, help="Video file to embed (MP4/MOV/AVI/MKV)")

    pd = sub.add_parser("decode", help="Decode video from WAV")
    pd.add_argument("--in", dest="in_wav", default=str(DIR_WAV / "stegoVid.wav"),
                    help="Stego WAV (default: stego/stegoVid.wav under wav_encrypt)")
    pd.add_argument("--key", type=int, required=True, help="Integer key used during encode")
    pd.add_argument("--out", dest="out_file", default=str(DIR_OUT / "recovered_video.bin"),
                    help="Output path for recovered video (default: ../StegPng/recovered_video.bin)")

    sub.add_parser("pipeline", help="Interactive prompts; auto-pick files; preview & verify")

    return p

def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.cmd == "encode":
        encode(args.in_wav, args.out_wav, args.key, args.lsb, args.video)
        print(f"✅ Encoded video into: {args.out_wav}")
        try:
            preview_waveforms(args.in_wav, args.out_wav)
        except Exception as e:
            print(f"(waveform preview failed: {e})")

    elif args.cmd == "decode":
        data = decode(args.in_wav, args.key)
        out_path = Path(args.out_file)
        if out_path.suffix.lower() == ".bin":
            out_path = out_path.with_suffix(_guess_video_ext(data))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        print(f"✅ Recovered video → {out_path}")

    elif args.cmd == "pipeline":
        interactive_pipeline()

if __name__ == "__main__":
    main()
