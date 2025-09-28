# wav_codec.py (extended to support 8/16/24/32-bit PCM WAV)
import wave
from pathlib import Path
from typing import Tuple, Iterator

import numpy as np

from base_codec import BaseCodec
from helpers import (
    build_header, parse_header, bits_from_bytes, bytes_from_bits,
    xor_bytes, crc32, HEADER_LEN
)

SUPPORTED_WIDTHS = {1, 2, 3, 4}  # bytes per sample

class WavCodec(BaseCodec):
    codec_id = "wav"
    pretty = "Audio (WAV PCM)"

    def accepts(self, path: Path) -> bool:
        return path.suffix.lower() == ".wav"

    # ---------- WAV I/O (supports 8/16/24/32-bit PCM) ----------
    def _read_wav(self, path: Path) -> Tuple[np.ndarray, dict]:
        """
        Returns: (samples, meta)
        samples shape: (frames, channels)
        dtype:
          - For 8-bit:  np.uint8  (0..255)
          - For 16-bit: np.int16  (-32768..32767)
          - For 24-bit: np.int32  (we keep full 24 bits in lower 24 bits, sign-extended)
          - For 32-bit: np.int32
        """
        with wave.open(str(path), "rb") as w:
            nchan, sampwidth, fr, nframes, _, _ = w.getparams()
            raw = w.readframes(nframes)

        if sampwidth not in SUPPORTED_WIDTHS:
            raise ValueError(f"Unsupported sample width: {sampwidth*8} bits")

        if sampwidth == 1:
            arr = np.frombuffer(raw, dtype=np.uint8).copy()
            arr = arr.reshape(-1, nchan)
        elif sampwidth == 2:
            arr = np.frombuffer(raw, dtype="<i2").copy().reshape(-1, nchan)
        elif sampwidth == 3:
            # 24-bit little endian -> sign-extend into int32
            byte_view = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
            sign = (byte_view[:, 2] & 0x80) != 0
            ext = np.where(sign, 0xFF, 0x00).astype(np.uint8)
            full = np.column_stack([byte_view, ext])
            arr32 = full.view("<i4")[:, 0]
            arr = arr32.reshape(-1, nchan)
        else:  # 4 bytes
            arr = np.frombuffer(raw, dtype="<i4").copy().reshape(-1, nchan)

        meta = {
            "nchan": nchan,
            "sampwidth": sampwidth,
            "fr": fr,
            "frames": nframes
        }
        return arr, meta

    def _write_wav(self, path: Path, arr: np.ndarray, meta: dict) -> None:
        """
        Writes array back with original sample width.
        For 24-bit we repack lower 24 bits little-endian.
        """
        nchan = meta["nchan"]
        sampwidth = meta["sampwidth"]
        fr = meta["fr"]

        with wave.open(str(path), "wb") as w:
            w.setnchannels(nchan)
            w.setsampwidth(sampwidth)
            w.setframerate(fr)

            if sampwidth == 1:
                # Data expected 0..255
                data = np.clip(arr, 0, 255).astype(np.uint8)
                w.writeframes(data.reshape(-1).tobytes())
            elif sampwidth == 2:
                data = np.clip(arr, -32768, 32767).astype("<i2")
                w.writeframes(data.reshape(-1).tobytes())
            elif sampwidth == 3:
                # Clamp to 24-bit range, then repack little-endian
                data = np.clip(arr, -2**23, 2**23 - 1).astype(np.int32).reshape(-1)
                # Convert each sample to 3 bytes little-endian
                out = np.empty(data.size * 3, dtype=np.uint8)
                # little-endian -> take lower 3 bytes
                out[0::3] = data & 0xFF
                out[1::3] = (data >> 8) & 0xFF
                out[2::3] = (data >> 16) & 0xFF
                w.writeframes(out.tobytes())
            else:  # 32-bit
                data = np.clip(arr, -2**31, 2**31 - 1).astype("<i4")
                w.writeframes(data.reshape(-1).tobytes())

    # ---------- capacity ----------
    def capacity_bytes(self, carrier: Path, bpc: int, start_sample: int = 0) -> int:
        """
        Returns usable payload capacity (excluding header) given bpc LSBs per sample
        starting at start_sample.
        """
        if bpc < 1 or bpc > 8:
            return 0
        arr, meta = self._read_wav(carrier)
        frames, chans = arr.shape
        if start_sample < 0 or start_sample >= frames:
            return 0
        available_frames = frames - start_sample
        cap_bits = available_frames * chans * bpc
        return max(0, cap_bits // 8 - HEADER_LEN)

    # ---------- bit helpers ----------
    def _embed_bits_into_value(self, val: int, bpc: int, bit_iter: Iterator[int], width_bytes: int) -> int:
        """
        Embed bpc bits into least significant bits of val (signed integer) preserving sign range.
        width_bytes: 1,2,3,4 (original storage width)
        We treat val as unsigned for bit ops, then restore signed domain if needed.
        """
        if width_bytes == 1:
            # val is 0..255
            u = val & 0xFF
            for k in range(bpc):
                b = next(bit_iter)
                if b:
                    u |= (1 << k)
                else:
                    u &= ~(1 << k)
            return u & 0xFF

        # For 16/24/32 we interpret val as signed
        bit_mask_all = {2: 0xFFFF, 3: 0xFFFFFF, 4: 0xFFFFFFFF}[width_bytes]
        # Convert to unsigned bit pattern
        if width_bytes == 2:
            u = val & 0xFFFF
        elif width_bytes == 3:
            u = val & 0xFFFFFF
        else:
            u = val & 0xFFFFFFFF

        for k in range(bpc):
            b = next(bit_iter)
            if b:
                u |= (1 << k)
            else:
                u &= ~(1 << k)
        u &= bit_mask_all

        # Restore to signed range
        if width_bytes == 2:
            if u & 0x8000:  # sign bit
                u -= 0x10000
        elif width_bytes == 3:
            if u & 0x800000:
                u -= 0x1000000
        elif width_bytes == 4:
            if u & 0x80000000:
                u -= 0x100000000
        return u

    def _extract_bits_from_value(self, val: int, bpc: int, width_bytes: int) -> Iterator[int]:
        """
        Yield the lowest bpc bits from val (signed) for given width_bytes.
        """
        if width_bytes == 1:
            u = val & 0xFF
        elif width_bytes == 2:
            u = val & 0xFFFF
        elif width_bytes == 3:
            u = val & 0xFFFFFF
        else:
            u = val & 0xFFFFFFFF
        for k in range(bpc):
            yield (u >> k) & 1

    # ---------- embed ----------
    def embed(self, carrier: Path, payload: bytes, out_path: Path, bpc: int, key: str, start_sample: int = 0) -> dict:
        if bpc < 1 or bpc > 8:
            raise ValueError("bpc must be between 1 and 8")
        arr, meta = self._read_wav(carrier)
        frames, chans = arr.shape
        width = meta["sampwidth"]

        if start_sample < 0 or start_sample >= frames:
            raise ValueError(f"start_sample {start_sample} out of range (0..{frames-1})")

        obf = xor_bytes(payload, key)
        total = build_header(obf, 0) + obf  # MIME=0 generic
        available_frames = frames - start_sample
        capacity_bits = available_frames * chans * bpc
        capacity_bytes = capacity_bits // 8
        if len(total) > capacity_bytes:
            raise ValueError(
                f"Capacity too small: need {len(total)} B, have ~{capacity_bytes} B "
                f"(offset {start_sample}, bpc {bpc}, {chans} ch, {width*8}-bit)"
            )

        flat = arr.copy().reshape(-1)  # interleaved signed samples (except 8-bit unsigned)
        bits_iter = bits_from_bytes(total)
        start_idx = start_sample * chans

        try:
            for i in range(start_idx, flat.size):
                flat[i] = self._embed_bits_into_value(int(flat[i]), bpc, bits_iter, width)
        except StopIteration:
            steg = flat.reshape(frames, chans)
            out_path = out_path.with_suffix(".wav")
            self._write_wav(out_path, steg, meta)

            # compute change metric only over the used region
            region_orig = arr[start_sample:]
            region_steg = steg[start_sample:]
            if width == 1:
                diff = (region_orig.astype(np.int16) ^ region_steg.astype(np.int16)) & ((1 << bpc) - 1)
            else:
                # Convert to unsigned mask domain for diff
                if width == 2:
                    orig_u = region_orig.astype(np.int32) & 0xFFFF
                    steg_u = region_steg.astype(np.int32) & 0xFFFF
                elif width == 3:
                    orig_u = region_orig.astype(np.int64) & 0xFFFFFF
                    steg_u = region_steg.astype(np.int64) & 0xFFFFFF
                else:
                    orig_u = region_orig.astype(np.int64) & 0xFFFFFFFF
                    steg_u = region_steg.astype(np.int64) & 0xFFFFFFFF
                diff = (orig_u ^ steg_u) & ((1 << bpc) - 1)
            changed_pct = float(np.any(diff != 0, axis=1).mean() * 100.0)

            return {
                "out": out_path,
                "bytes_embedded": len(obf),
                "metric_label": "Samples changed (region)",
                "metric_value": changed_pct,
                "changed_pct": changed_pct,
                "start_sample": start_sample,
                "bit_depth": width * 8
            }

        raise RuntimeError("Unexpected: ran out of space after capacity check passed")

    # ---------- extract ----------
    def extract(self, stego: Path, bpc: int, key: str, start_sample: int = 0) -> bytes:
        if bpc < 1 or bpc > 8:
            raise ValueError("bpc must be between 1 and 8")
        arr, meta = self._read_wav(stego)
        frames, chans = arr.shape
        width = meta["sampwidth"]
        if start_sample < 0 or start_sample >= frames:
            raise ValueError(f"start_sample {start_sample} out of range (0..{frames-1})")

        def reader() -> Iterator[int]:
            flat = arr.reshape(-1)
            start_idx = start_sample * chans
            for i in range(start_idx, flat.size):
                val = int(flat[i])
                yield from self._extract_bits_from_value(val, bpc, width)

        r = reader()
        header = bytes_from_bits(r, HEADER_LEN)
        ver, flags, length, check = parse_header(header)
        data = bytes_from_bits(r, length)
        data = xor_bytes(data, key)
        if crc32(data) != check:
            raise ValueError("Checksum mismatch (wrong key/bpc/start offset or corruption)")
        return data