# wav_codec.py
import wave
from pathlib import Path
from typing import Tuple, Iterator

import numpy as np

from base_codec import BaseCodec
from helpers import (
    build_header, parse_header, bits_from_bytes, bytes_from_bits,
    xor_bytes, crc32, HEADER_LEN
)


class WavCodec(BaseCodec):
    codec_id = "wav"
    pretty = "Audio (WAV PCM)"

    # ---------- file acceptance ----------
    def accepts(self, path: Path) -> bool:
        return path.suffix.lower() == ".wav"

    # ---------- I/O ----------
    def _read_wav(self, path: Path) -> Tuple[np.ndarray, dict]:
        with wave.open(str(path), "rb") as w:
            nchan, sampwidth, fr, nframes, _, _ = w.getparams()
            raw = w.readframes(nframes)
        if sampwidth == 1:
            dtype = np.uint8         # 8-bit PCM (unsigned)
        elif sampwidth == 2:
            dtype = np.int16         # 16-bit PCM (signed)
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth*8} bits")
        arr = np.frombuffer(raw, dtype=dtype).copy().reshape(-1, nchan)
        return arr, {"nchan": nchan, "sampwidth": sampwidth, "fr": fr}

    def _write_wav(self, path: Path, arr: np.ndarray, meta: dict) -> None:
        nchan, sampwidth, fr = meta["nchan"], meta["sampwidth"], meta["fr"]
        with wave.open(str(path), "wb") as w:
            w.setnchannels(nchan)
            w.setsampwidth(sampwidth)
            w.setframerate(fr)
            if sampwidth == 1:
                w.writeframes(arr.astype(np.uint8).tobytes())
            else:
                w.writeframes(arr.astype(np.int16).tobytes())

    # ---------- capacity ----------
    def capacity_bytes(self, carrier: Path, bpc: int) -> int:
        arr, _ = self._read_wav(carrier)
        frames, chans = arr.shape
        return max(0, (frames * chans * bpc) // 8 - HEADER_LEN)

    # ---------- helpers ----------
    def _modify_lsb_unsigned(self, u: int, bpc: int, bit_iter: Iterator[int]) -> int:
        """
        Given an unsigned 8- or 16-bit integer u, replace its lowest bpc bits
        from bit_iter and return the new unsigned value.
        """
        for k in range(bpc):
            b = next(bit_iter)
            if b:
                u |= (1 << k)
            else:
                u &= ~(1 << k)
        return u

    # ---------- embed ----------
    def embed(self, carrier: Path, payload: bytes, out_path: Path, bpc: int, key: str) -> dict:
        arr, meta = self._read_wav(carrier)
        frames, chans = arr.shape
        sampwidth = meta["sampwidth"]  # 1 or 2 bytes

        if bpc < 1 or bpc > 8:
            raise ValueError("bpc must be between 1 and 8 for PCM WAV")

        obf = xor_bytes(payload, key)
        total = build_header(obf, 0) + obf

        cap = (frames * chans * bpc) // 8
        if len(total) > cap:
            raise ValueError(f"Capacity too small: need {len(total)} B, have ~{cap} B at {bpc} bpc, {chans} ch")

        flat = arr.copy().reshape(-1)
        bits = bits_from_bytes(total)

        try:
            for i in range(flat.size):
                if sampwidth == 1:
                    # unsigned byte
                    v = int(flat[i])            # 0..255
                    u = self._modify_lsb_unsigned(v, bpc, bits)
                    flat[i] = np.uint8(u)
                else:
                    # 16-bit signed sample: convert to unsigned for LSB ops
                    v_signed = int(flat[i])     # -32768..32767
                    u = v_signed & 0xFFFF       # 0..65535
                    u = self._modify_lsb_unsigned(u, bpc, bits)
                    # Convert back to signed range
                    if u >= 0x8000:
                        v_new = u - 0x10000     # two's complement -> negative
                    else:
                        v_new = u
                    flat[i] = np.int16(v_new)
        except StopIteration:
            # Finished embedding earlier than filling all samples
            pass

        steg = flat.reshape(frames, chans)
        out_path = out_path.with_suffix(".wav")
        self._write_wav(out_path, steg, meta)

        # metric: % frames with any change in the used bpc bit-planes
        if sampwidth == 1:
            diff = (arr ^ steg) & ((1 << bpc) - 1)
        else:
            # view as unsigned for bitwise XOR
            orig_u = arr.astype(np.uint16)
            steg_u = steg.astype(np.uint16)
            diff = (orig_u ^ steg_u) & ((1 << bpc) - 1)
        changed = np.any(diff != 0, axis=1).mean() * 100.0

        return {"changed_pct": changed, "out": out_path}

    # ---------- extract ----------
    def extract(self, stego: Path, bpc: int, key: str) -> bytes:
        arr, meta = self._read_wav(stego)
        frames, chans = arr.shape
        sampwidth = meta["sampwidth"]

        if bpc < 1 or bpc > 8:
            raise ValueError("bpc must be between 1 and 8 for PCM WAV")

        def reader() -> Iterator[int]:
            flat = arr.reshape(-1)
            for i in range(flat.size):
                if sampwidth == 1:
                    v = int(flat[i])  # 0..255
                    for k in range(bpc):
                        yield (v >> k) & 1
                else:
                    v_signed = int(flat[i])          # -32768..32767
                    u = v_signed & 0xFFFF            # 0..65535 unsigned view
                    for k in range(bpc):
                        yield (u >> k) & 1

        r = reader()
        header = bytes_from_bits(r, HEADER_LEN)
        ver, flags, length, check = parse_header(header)
        data = bytes_from_bits(r, length)
        data = xor_bytes(data, key)
        if crc32(data) != check:
            raise ValueError("Checksum mismatch (wrong key or corrupted carrier)")
        return data