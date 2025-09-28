# mp4_codec.py (enhanced with region-based embedding)
from pathlib import Path
import cv2
import numpy as np

from base_codec import BaseCodec
from helpers import (
    build_header, parse_header, bits_from_bytes, bytes_from_bits,
    xor_bytes, crc32, HEADER_LEN
)

class Mp4Codec(BaseCodec):
    codec_id = "mp4"
    pretty   = "Video (MP4 H.264)"

    def accepts(self, path: Path) -> bool:
        return path.suffix.lower() in {".mp4", ".avi"}

    # NEW: region optional
    def capacity_bytes(self, carrier: Path, bpc: int, region=None) -> int:
        """
        region: (x, y, w, h) or None for entire frame
        """
        vid = cv2.VideoCapture(str(carrier))
        if not vid.isOpened():
            vid.release()
            return 0

        if region is None:
            # Original full-frame iterative method (kept to maintain accuracy)
            cap_bits = 0
            while True:
                ret, frame = vid.read()
                if not ret:
                    break
                H, W, C = frame.shape
                cap_bits += H * W * C * bpc
            vid.release()
            return max(0, cap_bits // 8 - HEADER_LEN)

        # Region-based capacity using frame count (faster)
        x, y, w, h = region
        frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = vid.read()
        if not ret:
            vid.release()
            return 0
        H, W, C = frame.shape
        # Validate region
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > W or y + h > H:
            vid.release()
            raise ValueError(f"Invalid region {(x,y,w,h)} for frame size {(W,H)}")
        if frame_count <= 0:
            # Fallback: count manually
            frame_count = 1
            while True:
                ret, frame = vid.read()
                if not ret:
                    break
                frame_count += 1

        vid.release()
        cap_bits = frame_count * w * h * 3 * bpc
        return max(0, cap_bits // 8 - HEADER_LEN)

    # CHANGED: added region
    def embed(self, carrier: Path, payload: bytes, out_path: Path,
              bpc: int, key: str, region=None) -> dict:
        cap = self.capacity_bytes(carrier, bpc, region=region)
        obf = xor_bytes(payload, key)
        total = build_header(obf, 0) + obf
        if len(total) > cap:
            raise ValueError(f"Capacity too small: need {len(total)} B, have ~{cap} B at {bpc} bpc (region={region})")

        bits = bits_from_bytes(total)

        vid = cv2.VideoCapture(str(carrier))
        if not vid.isOpened():
            raise ValueError("Could not open carrier video")

        fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_lossless = out_path.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        writer = cv2.VideoWriter(str(out_lossless), fourcc, fps, (W, H))
        if not writer.isOpened():
            # fallback (lossy — may break extraction)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_lossless = out_path.with_suffix(".mp4")
            writer = cv2.VideoWriter(str(out_lossless), fourcc, fps, (W, H))

        # Validate region if provided
        if region is not None:
            x, y, w, h = region
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > W or y + h > H:
                writer.release()
                vid.release()
                raise ValueError(f"Invalid region {(x,y,w,h)} for frame size {(W,H)}")

        try:
            while True:
                ret, frame = vid.read()
                if not ret:
                    break

                if region is None:
                    # Original full-frame behavior
                    flat = frame.reshape(-1, 3).copy()
                    if self._embed_into_flat(flat, bpc, bits):
                        frame_out = flat.reshape(H, W, 3)
                        writer.write(frame_out)
                        self._write_remaining_frames(vid, writer)
                        return {"out": out_lossless}
                    frame_out = flat.reshape(H, W, 3)
                    writer.write(frame_out)
                else:
                    x, y, w, h = region
                    # Work only on region
                    sub = frame[y:y+h, x:x+w, :]
                    region_flat = sub.reshape(-1, 3)
                    if self._embed_into_flat(region_flat, bpc, bits):
                        # region_flat already modifies 'sub' view -> frame updated
                        writer.write(frame)
                        self._write_remaining_frames(vid, writer)
                        return {
                            "out": out_lossless,
                            "bytes_embedded": len(obf),
                            "metric_label": "Appended tail",
                            "metric_value": len(obf),
                            "appended": len(obf)
                        }

                    writer.write(frame)
        finally:
            writer.release()
            vid.release()

        raise RuntimeError("Unexpected: ran out of space after capacity check passed")

    # NEW: helper to embed bits into flat pixel array in-place.
    def _embed_into_flat(self, flat: np.ndarray, bpc: int, bits_iter) -> bool:
        """
        Returns True if all bits consumed and embedding finished.
        """
        for px in range(flat.shape[0]):
            for ch in range(3):
                v = int(flat[px, ch])
                for k in range(bpc):
                    try:
                        bit = next(bits_iter)
                    except StopIteration:
                        flat[px, ch] = np.uint8(v)
                        return True
                    v = (v & (0xFF ^ (1 << k))) | ((bit & 1) << k)
                flat[px, ch] = np.uint8(v)
        return False  # still have bits

    # NEW: helper for finishing frames
    def _write_remaining_frames(self, vid, writer):
        while True:
            ret, f2 = vid.read()
            if not ret:
                break
            writer.write(f2)

    # CHANGED: added region
    def extract(self, stego: Path, bpc: int, key: str, region=None) -> bytes:
        vid = cv2.VideoCapture(str(stego))
        if not vid.isOpened():
            raise ValueError("Could not open stego video")

        def reader():
            while True:
                ret, frame = vid.read()
                if not ret:
                    break
                if region is None:
                    flat = frame.reshape(-1, 3)
                    for px in range(flat.shape[0]):
                        for ch in range(3):
                            v = int(flat[px, ch])
                            for k in range(bpc):
                                yield (v >> k) & 1
                else:
                    x, y, w, h = region
                    Hf, Wf, _ = frame.shape
                    if x < 0 or y < 0 or x + w > Wf or y + h > Hf:
                        raise ValueError("Region mismatch on extraction")
                    sub = frame[y:y+h, x:x+w, :]
                    flat = sub.reshape(-1, 3)
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
            raise ValueError("Checksum mismatch (wrong key, region, or corrupted carrier)")
        return data


# NEW: optional preview utility for region visualization
def preview_region(video_path: Path, x: int, y: int, w: int, h: int,
                   frame_index: int = 0, window: bool = True, save_path: Path = None):
    vid = cv2.VideoCapture(str(video_path))
    if not vid.isOpened():
        raise ValueError("Could not open video for preview")

    current = 0
    frame = None
    while current <= frame_index:
        ret, f = vid.read()
        if not ret:
            break
        frame = f
        current += 1
    vid.release()

    if frame is None:
        raise ValueError("Frame index out of range")

    H, W, _ = frame.shape
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > W or y + h > H:
        raise ValueError(f"Invalid region {(x,y,w,h)} for frame size {(W,H)}")

    preview = frame.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if save_path:
        cv2.imwrite(str(save_path), preview)
    if window:
        cv2.imshow("Region Preview", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# NEW: optional interactive selector (first frame)
def interactive_select_region(video_path: Path):
    vid = cv2.VideoCapture(str(video_path))
    if not vid.isOpened():
        raise ValueError("Could not open video")
    ret, frame = vid.read()
    vid.release()
    if not ret:
        raise ValueError("Could not read first frame")

    clone = frame.copy()
    pts = {"start": None, "end": None}
    selecting = {"active": False}
    window_name = "Select Region (drag LMB, press ENTER to confirm, ESC to cancel)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts["start"] = (x, y)
            pts["end"] = (x, y)
            selecting["active"] = True
        elif event == cv2.EVENT_MOUSEMOVE and selecting["active"]:
            pts["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            pts["end"] = (x, y)
            selecting["active"] = False

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        disp = clone.copy()
        if pts["start"] and pts["end"]:
            cv2.rectangle(disp, pts["start"], pts["end"], (0, 255, 0), 2)
        cv2.imshow(window_name, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):  # Enter
            break
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()
    if not pts["start"] or not pts["end"]:
        return None
    (x0, y0), (x1, y1) = pts["start"], pts["end"]
    x, y = min(x0, x1), min(y0, y1)
    w, h = abs(x1 - x0), abs(y1 - y0)
    return (x, y, w, h)