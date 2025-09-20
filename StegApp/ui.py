from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from pathlib import Path
from image_codec import ImageCodec
from wav_codec import WavCodec
from PIL import Image
from typing import Optional


CODECS = {
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
        self.last_output_path: Optional[Path] = None  # file produced by last Embed
        self.save_output_btn = QtWidgets.QPushButton("Save Output As…")
        self.save_output_btn.setEnabled(False)


        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("Carrier Type:", self.codec_combo)
        form.addRow("LSBs per channel:", self.bpc_spin)
        form.addRow("Key:", self.key_edit)

        grid = QtWidgets.QGridLayout(self)
        grid.addLayout(form, 0, 0, 1, 2)

        grid.addWidget(self.box_carrier, 1, 0, 1, 2)
        grid.addWidget(self.box_payload, 2, 0, 1, 2)
        grid.addWidget(self.box_stego,   3, 0, 1, 2)

        # Action buttons
        grid.addWidget(self.embed_btn,   4, 0)
        grid.addWidget(self.extract_btn, 4, 1)

        # Image previews (row 5)
        imgs = QtWidgets.QHBoxLayout()
        imgs.addWidget(self.view_orig)
        imgs.addWidget(self.view_steg)
        imgs.addWidget(self.view_diff)
        grid.addLayout(imgs, 5, 0, 1, 2)

        # Save row (row 6)
        grid.addWidget(self.save_output_btn, 6, 0, 1, 2)

        # Status bar (row 7)
        grid.addWidget(self.status, 7, 0, 1, 2)

        # Signals
        self.codec_combo.currentTextChanged.connect(self.on_codec_change)
        self.box_carrier.fileDropped.connect(self.on_carrier)
        self.box_payload.fileDropped.connect(self.on_payload)
        self.box_stego.fileDropped.connect(self.on_stego)
        self.embed_btn.clicked.connect(self.on_embed)
        self.extract_btn.clicked.connect(self.on_extract)
        self.save_output_btn.clicked.connect(self.on_save_output_as)

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

                # Where the codec saved the file:
                out_file = Path(result.get("out_path", out_path.with_suffix(".png")))
                self.last_output_path = out_file
                self.save_output_btn.setEnabled(True)
                self.status.setText(f"✅ Embedded → {out_file}")
            else:  # WAV
                self.view_diff.setText(f"Modified samples ≈ {result['changed_pct']:.2f}%")
                out_file = Path(result["out"])
                self.last_output_path = out_file
                self.save_output_btn.setEnabled(True)
                self.status.setText(f"✅ Embedded → {out_file}")
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
    
    def on_save_output_as(self):
        if not self.last_output_path or not Path(self.last_output_path).exists():
            self.status.setText("No output to save yet.")
            return

        suffix = self.last_output_path.suffix.lower()
        if suffix == ".png":
            filt = "PNG Image (*.png);;All Files (*)"
        elif suffix == ".wav":
            filt = "WAV Audio (*.wav);;All Files (*)"
        else:
            filt = "All Files (*)"

        dest, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Stego Output As…",
            str(self.last_output_path.name),
            filt
        )
        if not dest:
            return
        try:
            Path(dest).write_bytes(Path(self.last_output_path).read_bytes())
            self.status.setText(f"✅ Saved a copy to: {dest}")
        except Exception as e:
            self.status.setText(f"❌ Save failed: {e}")
