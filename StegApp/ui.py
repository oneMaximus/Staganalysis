from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import numpy as np
import cv2
from pathlib import Path
from image_codec import ImageCodec
from wav_codec import WavCodec
from PIL import Image
from typing import Optional
from mp4_codec import Mp4Codec
from analysis_widget import AnalysisWidget

CODECS = {
    "Image (PNG/BMP/TIFF)": ImageCodec(),
    "Audio (WAV PCM)"     : WavCodec(),
    "Video (MP4 H.264)"   : Mp4Codec(),
}

# add near top of ui.py
EXT_TO_CODEC = {
    ".mp4": "Video (MP4 H.264)",
    ".mov": "Video (MP4 H.264)",
    ".m4v": "Video (MP4 H.264)",
    ".wav": "Audio (WAV PCM)",
    ".png": "Image (PNG/BMP/TIFF)",
    ".bmp": "Image (PNG/BMP/TIFF)",
    ".tif": "Image (PNG/BMP/TIFF)",
    ".tiff":"Image (PNG/BMP/TIFF)",
    ".jpg": "Image (PNG/BMP/TIFF)",
    ".jpeg":"Image (PNG/BMP/TIFF)",
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

class VideoPlayer(QtWidgets.QWidget):
    """Try QMediaPlayer first; if it errors/unsupported, fallback to OpenCV frame pump."""
    def __init__(self, title: str):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.title_lbl = QtWidgets.QLabel(title)
        lay.addWidget(self.title_lbl)

        # --- Primary player ---
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.video_widget = QVideoWidget()
        # keep layout stable
        if hasattr(self.video_widget, "setAspectRatioMode"):
            self.video_widget.setAspectRatioMode(QtCore.Qt.KeepAspectRatio)
        self.video_widget.setMinimumSize(320, 180)
        self.video_widget.setMaximumHeight(260)
        self.video_widget.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        lay.addWidget(self.video_widget)
        self.player.setVideoOutput(self.video_widget)
        self.player.setMuted(True)

        # --- Fallback (OpenCV) ---
        self.cv_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.cv_label.setFrameShape(QtWidgets.QFrame.Box)
        self.cv_label.setMinimumSize(320, 180)
        self.cv_label.setMaximumHeight(260)
        self.cv_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.cv_label.setScaledContents(True)
        self.cv_label.hide()
        lay.addWidget(self.cv_label)

        # Constrain the container so row height stays steady
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMaximumHeight(300)

        # Fallback machinery
        self._cap = None
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self._path = None

        # Qt Multimedia errors -> fallback
        if hasattr(self.player, "errorOccurred"):
            self.player.errorOccurred.connect(self._on_qt_error)
        else:
            self.player.error.connect(self._on_qt_error)
        self.player.mediaStatusChanged.connect(self._on_media_status)

    def load(self, path: Path):
        self.stop()
        self._path = Path(path)
        self.video_widget.show()
        self.cv_label.hide()
        self.player.setMedia(QMediaContent(QtCore.QUrl.fromLocalFile(str(path))))
        self.player.play()

    def _on_media_status(self, status):
        if status == QMediaPlayer.InvalidMedia and self._path:
            self._switch_to_cv()

    def _on_qt_error(self, *_):
        self._switch_to_cv()

    def _switch_to_cv(self):
        self.player.stop()
        self.video_widget.hide()
        self._start_cv_fallback(self._path)

    def _start_cv_fallback(self, path: Path):
        import cv2
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            self.cv_label.setText("Could not open video (fallback).")
            self.cv_label.show()
            return
        fps = self._cap.get(5) or 30.0  # CAP_PROP_FPS = 5
        interval = int(max(15, 1000.0 / fps))
        self.cv_label.show()
        self._timer.start(interval)

    def _next_frame(self):
        import cv2
        if not self._cap:
            return
        ok, frame = self._cap.read()
        if not ok:
            self.stop()
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QtGui.QImage(frame.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.cv_label.width(), self.cv_label.height(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.cv_label.setPixmap(pix)

    def stop(self):
        try:
            self.player.stop()
        except:
            pass
        if self._timer.isActive():
            self._timer.stop()
        if self._cap is not None:
            try:
                self._cap.release()
            except:
                pass
            self._cap = None

    def closeEvent(self, e):
        self.stop()
        super().closeEvent(e)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__(); self.setWindowTitle("Steg Lab — Image & WAV LSB")
        self.codec_name = "Image (PNG/BMP/TIFF)"; self.codec = CODECS[self.codec_name]
        self.carrier: Optional[Path] = None; self.payload: Optional[Path] = None; self.stego: Optional[Path] = None
        self.view_video = VideoPlayer("Video Preview")

        self.codec_combo = QtWidgets.QComboBox(); self.codec_combo.addItems(CODECS.keys())
        self.box_carrier = DropBox("Carrier"); self.box_payload = DropBox("Payload (any file)"); self.box_stego = DropBox("Stego (for Extract)")
        self.bpc_spin = QtWidgets.QSpinBox(); self.bpc_spin.setRange(1,8); self.bpc_spin.setValue(1)
        self.key_edit = QtWidgets.QLineEdit(); self.key_edit.setPlaceholderText("Key (optional)")
        self.embed_btn = QtWidgets.QPushButton("Embed ▶"); self.extract_btn = QtWidgets.QPushButton("Extract ⏏")
        self.status = QtWidgets.QLabel("Ready."); self.status.setWordWrap(True)
        self.status.setText("⚠️ Falling back to OpenCV preview")

        self.view_orig = ImageView("Original"); self.view_steg = ImageView("Embedded"); self.view_diff = ImageView("Change map / metric")
        self.last_output_path: Optional[Path] = None  # file produced by last Embed
        self.save_output_btn = QtWidgets.QPushButton("Save Output As…")
        self.save_output_btn.setEnabled(False)


        # --- Tabs ---
        self.tabs = QtWidgets.QTabWidget(self)

        # Tab 1: Embed / Extract
        self.embed_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.embed_tab, "Embed / Extract")

        embed_grid = QtWidgets.QGridLayout(self.embed_tab)

        form = QtWidgets.QFormLayout()
        form.addRow("Carrier Type:", self.codec_combo)
        form.addRow("LSBs per channel:", self.bpc_spin)
        form.addRow("Key:", self.key_edit)
        embed_grid.addLayout(form, 0, 0, 1, 2)

        embed_grid.addWidget(self.box_carrier, 1, 0, 1, 2)
        embed_grid.addWidget(self.box_payload, 2, 0, 1, 2)
        embed_grid.addWidget(self.box_stego,   3, 0, 1, 2)

        embed_grid.addWidget(self.embed_btn,   4, 0)
        embed_grid.addWidget(self.extract_btn, 4, 1)

        imgs = QtWidgets.QHBoxLayout()
        imgs.addWidget(self.view_orig)
        imgs.addWidget(self.view_steg)
        imgs.addWidget(self.view_diff)
        imgs.addWidget(self.view_video)
        imgs.addStretch(1) 
        self.view_video.hide()
        embed_grid.addLayout(imgs, 5, 0, 1, 2)

        embed_grid.addWidget(self.save_output_btn, 6, 0, 1, 2)
        embed_grid.addWidget(self.status,          7, 0, 1, 2)

        # Tab 2: Steg Analysis
        self.analysis_tab = AnalysisWidget()
        self.tabs.addTab(self.analysis_tab, "Steg Analysis")

        # Make tabs the outer layout
        outer = QtWidgets.QVBoxLayout(self)
        outer.addWidget(self.tabs)

        # Signals
        self.codec_combo.currentTextChanged.connect(self.on_codec_change)
        self.box_carrier.fileDropped.connect(self.on_carrier)
        self.box_payload.fileDropped.connect(self.on_payload)
        self.box_stego.fileDropped.connect(self.on_stego)
        self.embed_btn.clicked.connect(self.on_embed)
        self.extract_btn.clicked.connect(self.on_extract)
        self.save_output_btn.clicked.connect(self.on_save_output_as)

    def on_codec_change(self, txt: str):
        self.codec_name = txt
        self.codec = CODECS[txt]
        # Reset preview visibilities on change
        if isinstance(self.codec, ImageCodec):
            self.view_video.hide()
            self.view_orig.show()
            self.view_steg.show()
            self.view_diff.show()
        elif isinstance(self.codec, Mp4Codec):
            # For video, we’ll show the video player; image labels are still there
            pass
        self.status.setText(f"Carrier type set to: {txt}")

    def on_carrier(self, p: Path):
        self.carrier = p

        # auto-switch codec by extension
        ext = p.suffix.lower()
        auto = EXT_TO_CODEC.get(ext)
        if auto and self.codec_combo.currentText() != auto:
            self.codec_combo.setCurrentText(auto)
            # on_codec_change() will run and adjust visibilities

        if isinstance(self.codec, ImageCodec):
            try:
                im = Image.open(p)
                rgb = im.convert("RGBA" if im.mode == "RGBA" else "RGB")
                self.view_video.hide()
                self.view_orig.show()
                self.view_steg.show()
                self.view_diff.show()
                self.view_orig.set_image_from_array(np.array(rgb))
            except Exception as e:
                self.view_orig.setText(f"No preview:\n{e}")

        elif isinstance(self.codec, Mp4Codec):
            try:
                self.view_orig.hide()
                self.view_steg.show()
                self.view_diff.show()
                self.view_video.show()
                self.view_video.load(p)   # see improved VideoPlayer below
            except Exception as e:
                self.status.setText(f"No video preview: {e}")

        else:
            self.view_orig.setText(p.name)

    def on_stego(self, p: Path):
        self.stego = p
        if isinstance(self.codec, ImageCodec):
            try:
                im = Image.open(p).convert("RGB")
                self.view_steg.set_image_from_array(np.array(im))
            except Exception:
                self.view_steg.setText(p.name)
        elif isinstance(self.codec, Mp4Codec):
            # Option: play stego video too; for now we just set a label
            self.view_steg.setText("Preview not implemented yet")
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
        elif suffix == ".mp4":
            filt = "MP4 Video (*.mp4);;All Files (*)"
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

    def on_payload(self, p: Path):
        self.payload = p
        # Optional: give some feedback
        self.status.setText(f"Payload set: {p.name}")

