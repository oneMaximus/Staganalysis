from __future__ import annotations
import sys, hashlib, mimetypes, wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Iterator

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets

# =========================
# Shared header + helpers
# =========================
MAGIC = b"STEG"; VERSION = 1; FLAG_NONE = 0
HEADER_LEN = 15  # MAGIC(4)|VER(1)|FLAGS(1)|MIME(1)|LEN(4)|CRC32(4)

def u32(n: int) -> bytes: return n.to_bytes(4, "big")
def crc32(data: bytes) -> int:
    from zlib import crc32 as _c; return _c(data) & 0xFFFFFFFF

def build_header(payload: bytes, mime_byte: int, flags: int = FLAG_NONE) -> bytes:
    return MAGIC + bytes([VERSION, flags, mime_byte]) + u32(len(payload)) + u32(crc32(payload))

def parse_header(h: bytes) -> Tuple[int,int,int,int]:
    if len(h) < HEADER_LEN or h[:4] != MAGIC:
        raise ValueError("Invalid/missing header")
    ver, flags, mime_b = h[4], h[5], h[6]
    ln  = int.from_bytes(h[7:11], "big")
    ch  = int.from_bytes(h[11:15], "big")
    return ver, flags, ln, ch  # (mime_b is optional for this UI; keep header small)

def bits_from_bytes(buf: bytes) -> Iterator[int]:
    for b in buf:
        for i in range(7, -1, -1):
            yield (b >> i) & 1

def bytes_from_bits(bit_iter: Iterator[int], n_bytes: int) -> bytes:
    out = bytearray(); acc = 0; cnt = 0
    for _ in range(n_bytes * 8):
        acc = (acc << 1) | (next(bit_iter) & 1); cnt += 1
        if cnt == 8: out.append(acc); acc = 0; cnt = 0
    return bytes(out)

def derive_keystream(key: str, n: int) -> bytes:
    if not key: return b"\x00"*n
    out = bytearray(); block = b""; seed = key.encode()
    while len(out) < n:
        block = hashlib.sha256(block + seed).digest()
        out.extend(block)
    return bytes(out[:n])

def xor_bytes(data: bytes, key: str) -> bytes:
    ks = derive_keystream(key, len(data))
    return bytes(d ^ k for d, k in zip(data, ks))

# =========================
# Base codec interface
# =========================
class BaseCodec:
    codec_id: str
    pretty: str
    def capacity_bytes(self, carrier: Path, bpc: int) -> int: raise NotImplementedError
    def embed(self, carrier: Path, payload: bytes, out_path: Path, bpc: int, key: str) -> dict: raise NotImplementedError
    def extract(self, stego: Path, bpc: int, key: str) -> bytes: raise NotImplementedError
    def accepts(self, path: Path) -> bool: raise NotImplementedError

# =========================
# Image codec (PNG/BMP/TIFF)
# =========================
class ImageCodec(BaseCodec):
    codec_id = "image"
    pretty   = "Image (PNG/BMP/TIFF)"

    def _arr(self, img: Image.Image) -> np.ndarray:
        if img.mode not in ("RGB","RGBA"):
            img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)

    def accepts(self, path: Path) -> bool:
        return path.suffix.lower() in {".png",".bmp",".tif",".tiff",".jpg",".jpeg"}  # loadable, but save as PNG

    def capacity_bytes(self, carrier: Path, bpc: int) -> int:
        arr = self._arr(Image.open(carrier))
        H,W,C = arr.shape
        return max(0, (H*W*C*bpc)//8 - HEADER_LEN)

    def embed(self, carrier: Path, payload: bytes, out_path: Path, bpc: int, key: str) -> dict:
        img = Image.open(carrier)
        arr = self._arr(img); H,W,C = arr.shape
        flat = arr.reshape(-1, C).copy()

        obf = xor_bytes(payload, key)
        header = build_header(obf, 0)
        total = header + obf

        cap = (H*W*C*bpc)//8
        if len(total) > cap:
            raise ValueError(f"Capacity too small: need {len(total)} B, have ~{cap} B at {bpc} bpc")

        orig = arr.copy()
        bits = bits_from_bytes(total)
        try:
            for px in range(flat.shape[0]):
                for ch in range(C):
                    v = int(flat[px,ch])
                    for k in range(bpc):
                        bit = next(bits)
                        v = (v & (0xFF ^ (1<<k))) | ((bit & 1) << k)
                    flat[px,ch] = np.uint8(v)
        except StopIteration:
            out = flat.reshape(H,W,C)
            Image.fromarray(out, mode=("RGBA" if arr.shape[2]==4 else "RGB")).save(out_path)
            # bit-change map (white = changed)
            diff = (orig ^ out)
            mask = np.zeros((H,W), dtype=np.uint8)
            for k in range(bpc):
                chg = (diff & (1<<k)) != 0
                pix = np.any(chg, axis=2)
                mask = np.where(pix, 255, mask)
            return {"orig": orig, "steg": out, "mask": mask}

        raise RuntimeError("Unexpected: ran out of space after capacity check passed")

    def extract(self, stego: Path, bpc: int, key: str) -> bytes:
        arr = self._arr(Image.open(stego))
        H,W,C = arr.shape
        flat = arr.reshape(-1, C)

        def reader():
            for px in range(flat.shape[0]):
                for ch in range(C):
                    v = int(flat[px,ch])
                    for k in range(bpc):
                        yield (v >> k) & 1

        r = reader()
        header = bytes_from_bits(r, HEADER_LEN)
        ver, flags, length, check = parse_header(header)
        data = bytes_from_bits(r, length)
        data = xor_bytes(data, key)
        if crc32(data) != check: raise ValueError("Checksum mismatch (wrong key or corrupted carrier)")
        return data

# =========================
# WAV codec (PCM 8/16-bit)
# =========================
class WavCodec(BaseCodec):
    codec_id = "wav"
    pretty   = "Audio (WAV PCM)"

    def accepts(self, path: Path) -> bool:
        return path.suffix.lower() == ".wav"

    def _read_wav(self, path: Path) -> Tuple[np.ndarray, dict]:
        with wave.open(str(path), "rb") as w:
            nchan, sampwidth, fr, nframes, _, _ = w.getparams()
            raw = w.readframes(nframes)
        if sampwidth == 1:
            dtype = np.uint8
        elif sampwidth == 2:
            dtype = np.int16
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth*8} bits")
        arr = np.frombuffer(raw, dtype=dtype).copy()
        arr = arr.reshape(-1, nchan)
        return arr, {"nchan": nchan, "sampwidth": sampwidth, "fr": fr}

    def _write_wav(self, path: Path, arr: np.ndarray, meta: dict) -> None:
        nchan, sampwidth, fr = meta["nchan"], meta["sampwidth"], meta["fr"]
        with wave.open(str(path), "wb") as w:
            w.setnchannels(nchan); w.setsampwidth(sampwidth); w.setframerate(fr)
            w.writeframes(arr.astype(np.int16 if sampwidth==2 else np.uint8).tobytes())

    def capacity_bytes(self, carrier: Path, bpc: int) -> int:
        arr, meta = self._read_wav(carrier)
        H, C = arr.shape  # frames, channels
        bits_per_sample = bpc * C
        return max(0, (H * bits_per_sample)//8 - HEADER_LEN)

    def _set_bits_value(self, v: int, bpc: int, bit_iter: Iterator[int]) -> int:
        # operate on unsigned space sized to sample width
        u = v & 0xFFFF if v < 0 or v > 255 else v
        for k in range(bpc):
            b = next(bit_iter)
            u = (u & (~(1<<k) & 0xFFFF)) | ((b & 1) << k)
        return u

    def embed(self, carrier: Path, payload: bytes, out_path: Path, bpc: int, key: str) -> dict:
        arr, meta = self._read_wav(carrier)
        frames, chans = arr.shape
        sampwidth = meta["sampwidth"]  # 1 or 2 bytes
        cap = (frames * chans * bpc)//8
        obf = xor_bytes(payload, key)
        total = build_header(obf, 0) + obf
        if len(total) > cap:
            raise ValueError(f"Capacity too small: need {len(total)} B, have ~{cap} B at {bpc} bpc, {chans} ch")

        flat = arr.copy().reshape(-1)  # interleaved samples
        bits = bits_from_bytes(total)
        try:
            for i in range(flat.size):
                v = int(flat[i])
                if sampwidth == 1:  # uint8
                    u = v
                    for k in range(bpc):
                        b = next(bits)
                        u = (u & (0xFF ^ (1<<k))) | ((b & 1) << k)
                    flat[i] = np.uint8(u)
                else:               # int16
                    u = self._set_bits_value(v, bpc, bits)  # returns 0..65535 with bits set
                    flat[i] = np.int16(u)                   # wrap to int16
        except StopIteration:
            steg = flat.reshape(frames, chans)
            out_path = out_path.with_suffix(".wav")
            self._write_wav(out_path, steg, meta)
            # simple metric: % samples modified in any of the used bit-planes
            if sampwidth == 1:
                diff = (arr ^ steg) & ((1<<bpc)-1)
            else:
                diff = ((arr.astype(np.uint16) ^ steg.astype(np.uint16)) & ((1<<bpc)-1))
            changed = np.any(diff != 0, axis=1).mean()*100.0
            return {"changed_pct": changed, "out": out_path}
        raise RuntimeError("Unexpected: ran out of space after capacity check passed")

    def extract(self, stego: Path, bpc: int, key: str) -> bytes:
        arr, meta = self._read_wav(stego)
        frames, chans = arr.shape
        sampwidth = meta["sampwidth"]

        def reader():
            flat = arr.reshape(-1)
            for i in range(flat.size):
                v = int(flat[i])
                if sampwidth == 1:
                    for k in range(bpc):
                        yield (v >> k) & 1
                else:
                    u = v & 0xFFFF
                    for k in range(bpc):
                        yield (u >> k) & 1

        r = reader()
        header = bytes_from_bits(r, HEADER_LEN)
        ver, flags, length, check = parse_header(header)
        data = bytes_from_bits(r, length)
        data = xor_bytes(data, key)
        if crc32(data) != check: raise ValueError("Checksum mismatch (wrong key or corrupted carrier)")
        return data

# =========================
# PyQt UI
# =========================
CODECS: dict[str, BaseCodec] = {
    "Image (PNG/BMP/TIFF)": ImageCodec(),
    "Audio (WAV PCM)"     : WavCodec(),
}

class DropBox(QtWidgets.QGroupBox):
    fileDropped = QtCore.pyqtSignal(Path)
    def __init__(self, title: str):
        super().__init__(title); self.setAcceptDrops(True)
        self.label = QtWidgets.QLabel("Drop a file here", alignment=QtCore.Qt.AlignCenter)
        self.label.setWordWrap(True); lay = QtWidgets.QVBoxLayout(self); lay.addWidget(self.label)
        self.setMinimumHeight(120)
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e: QtGui.QDropEvent):
        urls = e.mimeData().urls()
        if not urls: return
        p = Path(urls[0].toLocalFile()); self.label.setText(p.name); self.fileDropped.emit(p)

class ImageView(QtWidgets.QLabel):
    def __init__(self, title: str):
        super().__init__(); self.setAlignment(QtCore.Qt.AlignCenter); self.setFrameShape(QtWidgets.QFrame.Box)
        self.setMinimumSize(240, 240); self.setToolTip(title)
    def set_image_from_array(self, arr: np.ndarray):
        if arr.ndim == 2:
            h,w = arr.shape; qimg = QtGui.QImage(arr.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        else:
            h,w,c = arr.shape
            if c == 3:
                qimg = QtGui.QImage(arr.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
            else:
                qimg = QtGui.QImage(arr.data, w, h, 4*w, QtGui.QImage.Format_RGBA8888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.width(), self.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(pix)

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Steg Lab — Image & WAV LSB")
        self.codec_name = "Image (PNG/BMP/TIFF)"; self.codec = CODECS[self.codec_name]
        self.carrier: Optional[Path] = None; self.payload: Optional[Path] = None; self.stego: Optional[Path] = None

        self.codec_combo = QtWidgets.QComboBox(); self.codec_combo.addItems(CODECS.keys())
        self.box_carrier = DropBox("Carrier"); self.box_payload = DropBox("Payload (any file)"); self.box_stego = DropBox("Stego (for Extract)")
        self.bpc_spin = QtWidgets.QSpinBox(); self.bpc_spin.setRange(1,8); self.bpc_spin.setValue(1)
        self.key_edit = QtWidgets.QLineEdit(); self.key_edit.setPlaceholderText("Key (optional)")
        self.embed_btn = QtWidgets.QPushButton("Embed ▶"); self.extract_btn = QtWidgets.QPushButton("Extract ⏏")
        self.status = QtWidgets.QLabel("Ready."); self.status.setWordWrap(True)

        self.view_orig = ImageView("Original"); self.view_steg = ImageView("Embedded"); self.view_diff = ImageView("Change map / metric")

        # Layout
        form = QtWidgets.QFormLayout(); form.addRow("Carrier Type:", self.codec_combo); form.addRow("LSBs per channel:", self.bpc_spin); form.addRow("Key:", self.key_edit)
        grid = QtWidgets.QGridLayout(self)
        grid.addLayout(form, 0, 0, 1, 2)
        grid.addWidget(self.box_carrier, 1, 0, 1, 2)
        grid.addWidget(self.box_payload, 2, 0, 1, 2)
        grid.addWidget(self.box_stego,   3, 0, 1, 2)
        grid.addWidget(self.embed_btn,   4, 0)
        grid.addWidget(self.extract_btn, 4, 1)
        imgs = QtWidgets.QHBoxLayout(); imgs.addWidget(self.view_orig); imgs.addWidget(self.view_steg); imgs.addWidget(self.view_diff)
        grid.addLayout(imgs, 5, 0, 1, 2)
        grid.addWidget(self.status, 6, 0, 1, 2)

        # Signals
        self.codec_combo.currentTextChanged.connect(self.on_codec_change)
        self.box_carrier.fileDropped.connect(self.on_carrier)
        self.box_payload.fileDropped.connect(self.on_payload)
        self.box_stego.fileDropped.connect(self.on_stego)
        self.embed_btn.clicked.connect(self.on_embed)
        self.extract_btn.clicked.connect(self.on_extract)

    def on_codec_change(self, txt: str):
        self.codec_name = txt; self.codec = CODECS[txt]
        self.status.setText(f"Carrier type set to: {txt}")

    def on_carrier(self, p: Path):
        self.carrier = p
        # preview only for images
        if isinstance(self.codec, ImageCodec):
            try:
                img = Image.open(p).convert("RGBA" if Image.open(p).mode=="RGBA" else "RGB")
                self.view_orig.set_image_from_array(np.array(img))
            except Exception as e:
                self.view_orig.setText(f"No preview:\n{e}")
        else:
            self.view_orig.setText(p.name)

    def on_payload(self, p: Path): self.payload = p
    def on_stego(self, p: Path):
        self.stego = p
        if isinstance(self.codec, ImageCodec):
            try:
                self.view_steg.set_image_from_array(np.array(Image.open(p).convert("RGB")))
            except Exception: self.view_steg.setText(p.name)
        else:
            self.view_steg.setText(p.name)

    def on_embed(self):
        if not self.carrier or not self.payload:
            self.status.setText("Select a carrier and a payload first."); return
        if not self.codec.accepts(self.carrier):
            self.status.setText(f"{self.codec.pretty} expects a different carrier file type."); return
        bpc = int(self.bpc_spin.value()); key = self.key_edit.text()
        try:
            payload = Path(self.payload).read_bytes()
            stem = Path(self.carrier).stem
            out_path = Path(self.carrier).with_name(f"{stem}__steg")
            result = self.codec.embed(self.carrier, payload, out_path, bpc, key)

            if isinstance(self.codec, ImageCodec):
                self.view_steg.set_image_from_array(result["steg"])
                self.view_orig.set_image_from_array(result["orig"])
                mask_rgb = np.stack([result["mask"]]*3, axis=2)
                self.view_diff.set_image_from_array(mask_rgb)
                self.status.setText(f"✅ Embedded → {out_path.with_suffix('.png')}")
            else:  # WAV
                self.view_diff.setText(f"Modified samples ≈ {result['changed_pct']:.2f}%")
                self.status.setText(f"✅ Embedded → {result['out']}")
        except Exception as e:
            self.status.setText(f"❌ Embed failed: {e}")

    def on_extract(self):
        if not self.stego:
            self.status.setText("Drop a stego file first."); return
        if not self.codec.accepts(self.stego):
            self.status.setText(f"{self.codec.pretty} expects a different stego file type."); return
        bpc = int(self.bpc_spin.value()); key = self.key_edit.text()
        try:
            data = self.codec.extract(self.stego, bpc, key)
            out = Path(self.stego).with_name(Path(self.stego).stem + "__recovered.bin")
            out.write_bytes(data)
            self.status.setText(f"✅ Extracted payload → {out}")
        except Exception as e:
            self.status.setText(f"❌ Extract failed: {e}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.resize(1100, 720); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
