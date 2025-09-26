# analysis_widget.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets

# optional: first-frame video support if OpenCV is present
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


IMG_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
VID_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


class DropBox(QtWidgets.QGroupBox):
    fileDropped = QtCore.pyqtSignal(Path)
    def __init__(self, title: str = "Drop an image or video"):
        super().__init__(title)
        self.setAcceptDrops(True)
        self.label = QtWidgets.QLabel("Drop a file here", alignment=QtCore.Qt.AlignCenter)
        self.label.setWordWrap(True)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.label)
        self.setMinimumHeight(120)

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent):
        urls = e.mimeData().urls()
        if not urls:
            return
        p = Path(urls[0].toLocalFile())
        self.label.setText(p.name)
        self.fileDropped.emit(p)


class ImageView(QtWidgets.QLabel):
    def __init__(self, title: str = ""):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFrameShape(QtWidgets.QFrame.Box)
        self.setMinimumSize(320, 240)
        if title:
            self.setToolTip(title)

    def set_image_from_array(self, arr: np.ndarray):
        # Accepts 2D (grayscale) or 3D (RGB/RGBA uint8)
        if arr.ndim == 2:
            h, w = arr.shape
            qimg = QtGui.QImage(arr.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        else:
            h, w, c = arr.shape
            if c == 3:
                qimg = QtGui.QImage(arr.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            else:
                qimg = QtGui.QImage(arr.data, w, h, 4 * w, QtGui.QImage.Format_RGBA8888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.setPixmap(pix)


class AnalysisWidget(QtWidgets.QWidget):
    """
    Steg analysis page:
      - Drop an image/video
      - Choose channel: R/G/B
      - Choose bit-plane: 1..8 (1=LSB, 8=MSB)
      - Preview selected bit-plane as black & white
      - Optional: 'Show All 8 Bits' grid
    """
    def __init__(self):
        super().__init__()
        self.src_path: Optional[Path] = None
        self.src_rgb: Optional[np.ndarray] = None  # HxWx3 uint8

        # Controls
        self.drop = DropBox("Steg Analysis — Drop an image or video")
        self.chan_combo = QtWidgets.QComboBox()
        self.chan_combo.addItems(["Red", "Green", "Blue"])
        self.bit_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bit_slider.setRange(1, 8)  # 1..8
        self.bit_slider.setValue(1)
        self.bit_label = QtWidgets.QLabel("Bit: 1 (LSB)")
        self.btn_all = QtWidgets.QPushButton("Show All 8 Bit-Planes")

        # Previews
        self.preview = ImageView("Bit-plane preview (B/W)")

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("Channel:", self.chan_combo)
        form.addRow("View Bit (1=LSB, 8=MSB):", self.bit_slider)

        top = QtWidgets.QVBoxLayout(self)
        top.addWidget(self.drop)
        top.addLayout(form)
        top.addWidget(self.bit_label)
        top.addWidget(self.preview)
        top.addWidget(self.btn_all, alignment=QtCore.Qt.AlignRight)

        # Signals
        self.drop.fileDropped.connect(self.on_file)
        self.chan_combo.currentIndexChanged.connect(self.update_preview)
        self.bit_slider.valueChanged.connect(self.on_bit_change)
        self.btn_all.clicked.connect(self.show_all_bits)

    # ---------- file/load helpers ----------
    def on_file(self, p: Path):
        self.src_path = p
        ext = p.suffix.lower()
        try:
            if ext in IMG_EXTS:
                self.src_rgb = self._load_image_as_rgb(p)
            elif ext in VID_EXTS:
                self.src_rgb = self._load_video_first_frame(p)
            else:
                raise ValueError("Unsupported file type. Drop an image (PNG/BMP/TIFF/JPG) or video (MP4/MOV/AVI/MKV).")
            self.update_preview()
        except Exception as e:
            self.src_rgb = None
            self.preview.setText(f"Load failed:\n{e}")

    def _load_image_as_rgb(self, p: Path) -> np.ndarray:
        im = Image.open(p).convert("RGB")
        return np.array(im, dtype=np.uint8)

    def _load_video_first_frame(self, p: Path) -> np.ndarray:
        if not _HAS_CV2:
            raise RuntimeError("OpenCV not available. Install 'opencv-python' to analyze videos.")
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError("Could not open video.")
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError("Could not read first frame.")
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame.astype(np.uint8)

    # ---------- bit-plane logic ----------
    @staticmethod
    def _bit_plane(arr: np.ndarray, channel_idx: int, bit_ui: int) -> np.ndarray:
        """
        arr: HxWx3 uint8 RGB
        channel_idx: 0=R,1=G,2=B
        bit_ui: 1..8 (1=LSB → bit 0; 8=MSB → bit 7)
        returns HxW uint8 (0 or 255)
        """
        bit = bit_ui - 1
        plane = ((arr[:, :, channel_idx] >> bit) & 1) * 255
        return plane.astype(np.uint8)

    def on_bit_change(self, v: int):
        self.bit_label.setText(f"Bit: {v} ({'LSB' if v == 1 else 'MSB' if v == 8 else ''})")
        self.update_preview()

    def update_preview(self):
        if self.src_rgb is None:
            self.preview.setText("No media loaded.")
            return
        cidx = self.chan_combo.currentIndex()  # 0 R,1 G,2 B
        b = self.bit_slider.value()
        try:
            plane = self._bit_plane(self.src_rgb, cidx, b)
            self.preview.set_image_from_array(plane)
        except Exception as e:
            self.preview.setText(f"Preview failed:\n{e}")

    # ---------- all-bits grid ----------
    def show_all_bits(self):
        if self.src_rgb is None:
            return
        cidx = self.chan_combo.currentIndex()
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"All 8 bit-planes — {['Red','Green','Blue'][cidx]}")
        grid = QtWidgets.QGridLayout(dlg)

        # 2 rows x 4 cols
        labels = []
        for bit_ui in range(1, 9):
            plane = self._bit_plane(self.src_rgb, cidx, bit_ui)
            lab = ImageView(f"Bit {bit_ui}")
            lab.setMinimumSize(200, 140)
            lab.set_image_from_array(plane)
            labels.append(lab)

        positions = [(r, c) for r in range(2) for c in range(4)]
        for pos, lab in zip(positions, labels):
            grid.addWidget(lab, *pos)

        dlg.resize(900, 400)
        dlg.exec_()
