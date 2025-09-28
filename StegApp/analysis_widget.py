# analysis_widget.py
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# optional video
try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


class ImageView(QtWidgets.QLabel):
    def __init__(self, title: str = ""):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFrameShape(QtWidgets.QFrame.Box)
        self.setMinimumSize(480, 270)
        self.setMaximumHeight(360)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        if title:
            self.setToolTip(title)

    def set_image_from_array(self, arr: np.ndarray):
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
            self.width(), self.height(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.setPixmap(pix)


class IntensityHistogram(QtWidgets.QWidget):
    """Matplotlib histogram 0..255 for the selected color channel."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(6, 2.2), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Channel intensity histogram")
        self.ax.set_xlabel("Pixel intensity (0–255)")
        self.ax.set_ylabel("No. of pixels")
        # Pre-create 256 bars so we only change heights
        self.bars = self.ax.bar(np.arange(256), np.zeros(256), width=1.0, align="center")
        self.ax.set_xlim(-0.5, 255.5)
        self.ax.grid(True, axis="y", alpha=0.25)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.canvas)

    def update_hist(self, channel_u8: np.ndarray):
        # channel_u8: 2D uint8 (single color channel)
        hist = np.bincount(channel_u8.ravel(), minlength=256)
        # update heights without re-plotting
        for i, h in enumerate(hist):
            self.bars[i].set_height(int(h))
        self.ax.relim()
        self.ax.autoscale_view(scaley=True)
        self.canvas.draw_idle()


class AnalysisWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # --- small drop/filename strip ---
        self.file_lbl = QtWidgets.QLabel("Drop an image or video (mp4/mov/m4v) here")
        self.file_lbl.setObjectName("dropBanner")
        self.file_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.file_lbl.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.file_lbl.setMinimumHeight(44)
        self.file_lbl.setMaximumHeight(60)

        # controls
        self.channel = QtWidgets.QComboBox()
        self.channel.addItems(["Red", "Green", "Blue"])
        self.bit_spin = QtWidgets.QSpinBox(); self.bit_spin.setRange(1, 8); self.bit_spin.setValue(1)
        self.bit_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.bit_spin.setMinimumHeight(30)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("Channel:"))
        ctrl.addWidget(self.channel)
        ctrl.addSpacing(12)
        ctrl.addWidget(QtWidgets.QLabel("Bit plane (1=LSB):"))
        ctrl.addWidget(self.bit_spin)
        ctrl.addStretch(1)

        # left: histogram   right: preview  (in a resizable splitter)
        self.hist = IntensityHistogram()
        self.view = ImageView()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.hist)
        splitter.addWidget(self.view)

        # give left a reasonable minimum so it doesn’t collapse entirely
        self.hist.setMinimumWidth(360)
        self.view.setMinimumWidth(360)
        splitter.setSizes([500, 700])                 # initial ratio
        splitter.setChildrenCollapsible(False)

        # page layout
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.file_lbl)
        lay.addLayout(ctrl)
        lay.addWidget(splitter)

        # DnD
        self.setAcceptDrops(True)

        # state
        self._img: Optional[np.ndarray] = None
        self._cap = None
        self._timer = QtCore.QTimer(self); self._timer.timeout.connect(self._tick)
        self._fps_interval = 33
        self._frame_count = 0
        self._stats_every = 1      # histogram update every N frames for video

        self.channel.currentIndexChanged.connect(self._refresh)
        self.bit_spin.valueChanged.connect(self._refresh)

    # ----- DnD -----
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if not urls: return
        self.load_path(Path(urls[0].toLocalFile()))

    # ----- loading -----
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
        self._render(frame, is_video=True)

    def _refresh(self):
        if self._img is not None:
            self._render(self._img, is_video=False)

    def _render(self, rgb: np.ndarray, is_video: bool):
        c = self.channel.currentIndex()     # 0=R,1=G,2=B
        bit = self.bit_spin.value() - 1     # 0..7
        chan = rgb[:, :, c].astype(np.uint8)
        plane = ((chan >> bit) & 1).astype(np.uint8)

        # preview: selected bit-plane as BW
        vis = (plane * 255).astype(np.uint8)
        h, w = vis.shape
        qimg = QtGui.QImage(vis.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.view.width(), self.view.height(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.view.setPixmap(pix)

        # histogram (raw intensities of the selected channel)
        if not is_video:
            self.hist.update_hist(chan)
        else:
            self._frame_count = (self._frame_count + 1) % self._stats_every
            if self._frame_count == 0:
                self.hist.update_hist(chan)

    def closeEvent(self, e):
        self._stop_video()
        super().closeEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._img is not None:
            self._refresh()
