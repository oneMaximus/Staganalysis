# analysis_widget.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets

import numpy as np


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
    def __init__(self):
        super().__init__()
        # --- controls ---
        self.file_lbl = QtWidgets.QLabel("Drop an image or video (mp4/mov/m4v) here")
        self.file_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.file_lbl.setFrameShape(QtWidgets.QFrame.Box)
        self.file_lbl.setMinimumHeight(80)

        self.channel = QtWidgets.QComboBox()
        self.channel.addItems(["Red","Green","Blue"])
        self.bit_spin = QtWidgets.QSpinBox(); self.bit_spin.setRange(1,8); self.bit_spin.setValue(1)

        self.view = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.view.setFrameShape(QtWidgets.QFrame.Box)
        self.view.setMinimumSize(480,270)
        self.view.setMaximumHeight(360)
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        lay = QtWidgets.QVBoxLayout(self)
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Channel:")); ctrl.addWidget(self.channel)
        ctrl.addWidget(QtWidgets.QLabel("Bit plane (1=LSB):")); ctrl.addWidget(self.bit_spin)
        lay.addWidget(self.file_lbl); lay.addLayout(ctrl); lay.addWidget(self.view)

        # drag&drop
        self.setAcceptDrops(True)

        # state
        self._img = None                # numpy array for still images
        self._cap = None                # cv2.VideoCapture for video
        self._timer = QtCore.QTimer(self); self._timer.timeout.connect(self._tick)
        self._fps_interval = 33

        self.channel.currentIndexChanged.connect(self._refresh)
        self.bit_spin.valueChanged.connect(self._refresh)

    # ----- drag & drop for analysis tab -----
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if not urls: return
        p = Path(urls[0].toLocalFile())
        self.load_path(p)

    def load_path(self, p: Path):
        self.file_lbl.setText(p.name)
        self._stop_video()

        ext = p.suffix.lower()
        if ext in {".mp4", ".mov", ".m4v", ".avi", ".mkv"}:
            if not _HAS_CV2:
                self.view.setText("OpenCV not installed; cannot open video.")
                return
            self._cap = cv2.VideoCapture(str(p))
            if not self._cap.isOpened():
                self.view.setText("Could not open video.")
                return
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._fps_interval = int(max(15, 1000.0 / fps))
            self._timer.start(self._fps_interval)
            self._img = None
        else:
            # still image
            im = Image.open(p).convert("RGB")
            self._img = np.array(im)
            self._refresh()

    def _stop_video(self):
        if self._timer.isActive(): self._timer.stop()
        if self._cap is not None:
            try: self._cap.release()
            except: pass
            self._cap = None

    def _tick(self):
        if not self._cap: return
        ok, frame = self._cap.read()
        if not ok:
            self._stop_video(); return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._render_bitplane(frame)

    def _refresh(self):
        if self._img is not None:
            self._render_bitplane(self._img)

    def _render_bitplane(self, rgb: np.ndarray):
        # channel index
        c = self.channel.currentIndex()  # 0=R,1=G,2=B
        bit = self.bit_spin.value() - 1  # 0..7
        chan = rgb[:,:,c]
        plane = ((chan >> bit) & 1).astype(np.uint8) * 255
        h,w = plane.shape
        qimg = QtGui.QImage(plane.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.view.width(), self.view.height(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.view.setPixmap(pix)

    def closeEvent(self, e):
        self._stop_video()
        super().closeEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._img is not None:
            self._refresh()


