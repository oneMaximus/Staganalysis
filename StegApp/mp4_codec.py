# mp4_codec.py
from base_codec import BaseCodec
from helpers import (
    build_header, parse_header, bits_from_bytes, bytes_from_bits,
    xor_bytes, crc32, HEADER_LEN
)
from pathlib import Path
import cv2
import numpy as np


class Mp4Codec(BaseCodec):
    codec_id = "mp4"
    pretty   = "Video (MP4 H.264)"

    def accepts(self, path: Path) -> bool:
        return path.suffix.lower() == ".mp4"

    def capacity_bytes(self, carrier: Path, bpc: int) -> int:
        cap = 0
        vid = cv2.VideoCapture(str(carrier))
        while True:
            ret, frame = vid.read()
            if not ret:
                break
            H, W, C = frame.shape
            cap += (H * W * C * bpc) // 8
        vid.release()
        return max(0, cap - HEADER_LEN)

    def embed(self, carrier: Path, payload: bytes, out_path: Path, bpc: int, key: str) -> dict:
        cap = self.capacity_bytes(carrier, bpc)
        obf = xor_bytes(payload, key)
        total = build_header(obf, 0) + obf
        if len(total) > cap:
            raise ValueError(f"Capacity too small: need {len(total)} B, have ~{cap} B at {bpc} bpc")

        bits = bits_from_bytes(total)

        vid = cv2.VideoCapture(str(carrier))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = vid.get(cv2.CAP_PROP_FPS)
        W = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = out_path.with_suffix(".mp4")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

        try:
            while True:
                ret, frame = vid.read()
                if not ret:
                    break
                flat = frame.reshape(-1, 3).copy()
                for px in range(flat.shape[0]):
                    for ch in range(3):
                        v = int(flat[px, ch])
                        for k in range(bpc):
                            try:
                                bit = next(bits)
                            except StopIteration:
                                frame_out = flat.reshape(H, W, 3)
                                writer.write(frame_out)
                                # Write remaining frames unchanged
                                while True:
                                    ret, f2 = vid.read()
                                    if not ret:
                                        break
                                    writer.write(f2)
                                writer.release()
                                vid.release()
                                return {"out": out_path}
                            v = (v & (0xFF ^ (1 << k))) | ((bit & 1) << k)
                        flat[px, ch] = np.uint8(v)
                frame_out = flat.reshape(H, W, 3)
                writer.write(frame_out)
        finally:
            writer.release()
            vid.release()

        raise RuntimeError("Unexpected: ran out of space after capacity check passed")

    def extract(self, stego: Path, bpc: int, key: str) -> bytes:
        vid = cv2.VideoCapture(str(stego))
        def reader():
            while True:
                ret, frame = vid.read()
                if not ret:
                    break
                flat = frame.reshape(-1, 3)
                for px in range(flat.shape[0]):
                    for ch in range(3):
                        v = int(flat[px, ch])
                        for k in range(bpc):
                            yield (v >> k) & 1

        r = reader()
        header = bytes_from_bits(r, HEADER_LEN)
        ver, flags, length, check = parse_header(header)
        data = bytes_from_bits(r, length)
        vid.release()
        data = xor_bytes(data, key)
        if crc32(data) != check:
            raise ValueError("Checksum mismatch (wrong key or corrupted carrier)")
        return data
