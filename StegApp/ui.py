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
# UPDATED import to include region helpers
from mp4_codec import Mp4Codec, preview_region, interactive_select_region
from analysis_widget import AnalysisWidget

CODECS = {
    "Image (PNG/BMP/TIFF)": ImageCodec(),
    "Audio (WAV PCM)"     : WavCodec(),
    "Video (MP4 H.264)"   : Mp4Codec(),
}

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

class PayloadWidget(QtWidgets.QGroupBox):
    changed = QtCore.pyqtSignal()
    def __init__(self, title="Payload"):
        super().__init__(title)
        self.tabs = QtWidgets.QTabWidget()
        self.file_tab = QtWidgets.QWidget()
        self.text_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.file_tab, "File")
        self.tabs.addTab(self.text_tab, "Text")

        self.drop = DropBox("Drop a file here")
        lay_file = QtWidgets.QVBoxLayout(self.file_tab)
        lay_file.addWidget(self.drop)

        self.text_edit = QtWidgets.QPlainTextEdit()
        self.text_edit.setPlaceholderText("Type payload text here…")
        self.text_edit.setMinimumHeight(120)

        self.encoding = QtWidgets.QComboBox()
        self.encoding.addItems(["utf-8", "utf-16-le", "utf-16-be", "latin-1"])
        enc_row = QtWidgets.QHBoxLayout()
        enc_row.addWidget(QtWidgets.QLabel("Encoding:"))
        enc_row.addWidget(self.encoding)
        enc_row.addStretch(1)

        lay_text = QtWidgets.QVBoxLayout(self.text_tab)
        lay_text.addLayout(enc_row)
        lay_text.addWidget(self.text_edit)

        outer = QtWidgets.QVBoxLayout(self)
        outer.addWidget(self.tabs)

        self._file_path: Optional[Path] = None

        self.drop.fileDropped.connect(self._on_file)
        self.text_edit.textChanged.connect(self.changed)
        self.encoding.currentIndexChanged.connect(self.changed)
        self.tabs.currentChanged.connect(lambda _: self.changed.emit())

    def has_payload(self) -> bool:
        return (self.tabs.currentIndex() == 0 and self._file_path is not None) or \
               (self.tabs.currentIndex() == 1 and len(self.text_edit.toPlainText()) > 0)

    def payload_name(self) -> str:
        if self.tabs.currentIndex() == 0 and self._file_path:
            return self._file_path.name
        txt = self.text_edit.toPlainText()
        return f"text:{min(20, len(txt))} chars" if txt else "text:empty"

    def payload_bytes(self) -> bytes:
        if self.tabs.currentIndex() == 0:
            if not self._file_path:
                return b""
            return Path(self._file_path).read_bytes()
        enc = self.encoding.currentText()
        return self.text_edit.toPlainText().encode(enc, errors="replace")

    def _on_file(self, p: Path):
        self._file_path = p
        self.changed.emit()


class DropBox(QtWidgets.QGroupBox):
    fileDropped = QtCore.pyqtSignal(Path)
    def __init__(self, title: str):
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
        if not urls: return
        p = Path(urls[0].toLocalFile())
        self.label.setText(p.name)
        self.fileDropped.emit(p)


class ImageView(QtWidgets.QLabel):
    def __init__(self, title: str):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFrameShape(QtWidgets.QFrame.Box)
        self.setMinimumSize(240, 240)
        self.setToolTip(title)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMaximumHeight(300)
    def set_image_from_array(self, arr: np.ndarray):
        if arr.ndim == 2:
            h,w = arr.shape
            qimg = QtGui.QImage(arr.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        else:
            h,w,c = arr.shape
            if c == 3:
                qimg = QtGui.QImage(arr.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
            else:
                qimg = QtGui.QImage(arr.data, w, h, 4*w, QtGui.QImage.Format_RGBA8888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.width(), self.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(pix)


class FixedVideoWidget(QVideoWidget):
    def sizeHint(self):
        return QtCore.QSize(480,270)


class VideoPlayer(QtWidgets.QWidget):
    def __init__(self, title: str):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.title_lbl = QtWidgets.QLabel(title); lay.addWidget(self.title_lbl)
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.video_widget = FixedVideoWidget()
        self.video_widget.setMinimumSize(480,270)
        self.video_widget.setMaximumHeight(270)
        self.video_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        if hasattr(self.video_widget,"setAspectRatioMode"):
            self.video_widget.setAspectRatioMode(QtCore.Qt.KeepAspectRatio)
        lay.addWidget(self.video_widget)
        self.player.setVideoOutput(self.video_widget); self.player.setMuted(True)
        self.cv_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.cv_label.setFrameShape(QtWidgets.QFrame.Box)
        self.cv_label.setMinimumSize(480,270)
        self.cv_label.setMaximumHeight(270)
        self.cv_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.cv_label.setScaledContents(True); self.cv_label.hide(); lay.addWidget(self.cv_label)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumHeight(300); self.setMaximumHeight(300)
        self._cap=None; self._timer=QtCore.QTimer(self); self._timer.timeout.connect(self._next_frame); self._path=None
        if hasattr(self.player,"errorOccurred"):
            self.player.errorOccurred.connect(self._on_qt_error)
        else:
            self.player.error.connect(self._on_qt_error)
        self.player.mediaStatusChanged.connect(self._on_media_status)

    def load(self, path: Path):
        self.stop(); self._path=Path(path)
        self.video_widget.show(); self.cv_label.hide()
        self.player.setMedia(QMediaContent(QtCore.QUrl.fromLocalFile(str(path))))
        self.player.play()

    def _on_media_status(self, status):
        if status == QMediaPlayer.InvalidMedia and self._path:
            self._switch_to_cv()

    def _on_qt_error(self,*_):
        self._switch_to_cv()

    def _switch_to_cv(self):
        self.player.stop(); self.video_widget.hide(); self._start_cv_fallback(self._path)

    def _start_cv_fallback(self, path: Path):
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            self.cv_label.setText("Could not open video (fallback)."); self.cv_label.show(); return
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._timer.start(int(max(15,1000.0/fps))); self.cv_label.show()

    def _next_frame(self):
        if not self._cap: return
        ok, frame = self._cap.read()
        if not ok: self.stop(); return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = frame.shape
        qimg = QtGui.QImage(frame.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.cv_label.width(), self.cv_label.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.cv_label.setPixmap(pix)

    def stop(self):
        try: self.player.stop()
        except: pass
        if self._timer.isActive(): self._timer.stop()
        if self._cap is not None:
            try: self._cap.release()
            except: pass
            self._cap=None

    def closeEvent(self,e):
        self.stop(); super().closeEvent(e)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Steg Lab — Image, WAV, MP4 LSB")
        self.codec_name = "Image (PNG/BMP/TIFF)"
        self.codec = CODECS[self.codec_name]
        self.carrier: Optional[Path] = None
        self.payload: Optional[Path] = None
        self.stego: Optional[Path] = None

        self.view_video = VideoPlayer("Video Preview")

        self.codec_combo = QtWidgets.QComboBox(); self.codec_combo.addItems(CODECS.keys())
        self.box_carrier = DropBox("Carrier")
        self.payload_widget = PayloadWidget("Payload")
        self.box_stego   = DropBox("Stego (for Extract)")
        self.bpc_spin = QtWidgets.QSpinBox(); self.bpc_spin.setRange(0,7); self.bpc_spin.setValue(1)
        self.bpc_spin.setMinimumHeight(30)

        self.key_edit = QtWidgets.QLineEdit(); self.key_edit.setPlaceholderText("Key (optional)")
        self.embed_btn = QtWidgets.QPushButton("Embed ▶")
        self.extract_btn = QtWidgets.QPushButton("Extract ⏏")
        self.status = QtWidgets.QLabel("Ready."); self.status.setWordWrap(True)

        self.view_orig = ImageView("Original")
        self.view_steg = ImageView("Embedded")
        self.view_diff = ImageView("Change map / metric")

        self.last_output_path: Optional[Path] = None
        self.save_output_btn = QtWidgets.QPushButton("Save Output As…")
        self.save_output_btn.setEnabled(False)

        self.tabs = QtWidgets.QTabWidget(self)
        self.embed_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.embed_tab, "Embed / Extract")
        embed_grid = QtWidgets.QGridLayout(self.embed_tab)

        form = QtWidgets.QFormLayout()
        form.addRow("Carrier Type:", self.codec_combo)
        form.addRow("LSBs per channel:", self.bpc_spin)
        form.addRow("Key:", self.key_edit)
        embed_grid.addLayout(form, 0,0,1,3)

        # ---------- NEW REGION UI (Audio Only) ----------
        self.audio_region_group = QtWidgets.QGroupBox("Audio Region (LSB area)")
        ag_lay = QtWidgets.QGridLayout(self.audio_region_group)

        self.audio_region_enable_chk = QtWidgets.QCheckBox("Enable audio start (WAV only)")
        self.audio_region_enable_chk.setChecked(True)
        ag_lay.addWidget(self.audio_region_enable_chk, 0, 0, 1, 3)

        ag_lay.addWidget(QtWidgets.QLabel("Start Sample:"), 1, 0)
        # Create the audio start-sample control locally (no need to add it anywhere else)
        self.start_sample_spin = QtWidgets.QSpinBox()
        self.start_sample_spin.setRange(0, 1_000_000_000)
        self.start_sample_spin.setValue(0)
        self.start_sample_spin.setToolTip("WAV only: frame index to start embedding/extracting (header+payload).")
        
        ag_lay.addWidget(self.start_sample_spin, 1, 1)

        self.audio_region_enable_chk.toggled.connect(self.start_sample_spin.setEnabled)

        self.audio_region_group.hide()
        embed_grid.addWidget(self.audio_region_group, 1, 0, 1, 3)
        # ------------------------------------------------

        # ---------- NEW REGION UI (Video Only) ----------
        self.region_group = QtWidgets.QGroupBox("Video Region (LSB area)")
        rg_lay = QtWidgets.QGridLayout(self.region_group)
        self.region_enable_chk = QtWidgets.QCheckBox("Enable region (video only)")
        self.region_x = QtWidgets.QSpinBox(); self.region_x.setRange(0, 100000)
        self.region_y = QtWidgets.QSpinBox(); self.region_y.setRange(0, 100000)
        self.region_w = QtWidgets.QSpinBox(); self.region_w.setRange(0, 100000)
        self.region_h = QtWidgets.QSpinBox(); self.region_h.setRange(0, 100000)
        for sb in (self.region_x, self.region_y, self.region_w, self.region_h):
            sb.setEnabled(False)

        rg_lay.addWidget(self.region_enable_chk, 0,0,1,3)
        rg_lay.addWidget(QtWidgets.QLabel("X:"),1,0); rg_lay.addWidget(self.region_x,1,1)
        rg_lay.addWidget(QtWidgets.QLabel("Y:"),1,2); rg_lay.addWidget(self.region_y,1,3)
        rg_lay.addWidget(QtWidgets.QLabel("W:"),2,0); rg_lay.addWidget(self.region_w,2,1)
        rg_lay.addWidget(QtWidgets.QLabel("H:"),2,2); rg_lay.addWidget(self.region_h,2,3)

        self.region_preview_btn = QtWidgets.QPushButton("Preview Region")
        self.region_pick_btn = QtWidgets.QPushButton("Pick (Interactive)")
        self.region_preview_btn.setEnabled(False)
        self.region_pick_btn.setEnabled(False)
        rg_lay.addWidget(self.region_preview_btn,3,0,1,2)
        rg_lay.addWidget(self.region_pick_btn,3,2,1,2)

        embed_grid.addWidget(self.region_group,1,0,1,3)
        # ------------------------------------------------

        # Side-by-side: Carrier | Payload | Stego
        embed_grid.addWidget(self.box_carrier, 2, 0)
        embed_grid.addWidget(self.payload_widget, 2, 1)
        embed_grid.addWidget(self.box_stego,     2, 2)

        # Make the three columns share space evenly
        embed_grid.setColumnStretch(0, 1)
        embed_grid.setColumnStretch(1, 1)
        embed_grid.setColumnStretch(2, 1)

        # Single-row buttons under the three boxes
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.embed_btn)
        btn_row.addWidget(self.extract_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.save_output_btn)
        embed_grid.addLayout(btn_row, 3, 0, 1, 3)

        imgs = QtWidgets.QHBoxLayout()
        imgs.addWidget(self.view_orig)
        imgs.addWidget(self.view_steg)
        imgs.addWidget(self.view_diff)

        self.view_video.hide()
        video_holder = QtWidgets.QFrame()
        video_holder.setFrameShape(QtWidgets.QFrame.NoFrame)
        video_holder.setMinimumHeight(300); video_holder.setMaximumHeight(300)
        vh = QtWidgets.QVBoxLayout(video_holder); vh.setContentsMargins(0,0,0,0); vh.addWidget(self.view_video)
        imgs.addWidget(video_holder)

        embed_grid.addLayout(imgs, 4, 0, 1, 3)
        embed_grid.setRowMinimumHeight(4, 300)

        self.analysis_tab = AnalysisWidget()
        self.tabs.addTab(self.analysis_tab, "Steg Analysis")

        outer = QtWidgets.QVBoxLayout(self)
        outer.addWidget(self.tabs)
        outer.addWidget(self.status)

        # Signals
        self.codec_combo.currentTextChanged.connect(self.on_codec_change)
        self.box_carrier.fileDropped.connect(self.on_carrier)
        self.payload_widget.changed.connect(
            lambda: self.status.setText(f"Payload set: {self.payload_widget.payload_name()}")
        )
        self.box_stego.fileDropped.connect(self.on_stego)
        self.embed_btn.clicked.connect(self.on_embed)
        self.extract_btn.clicked.connect(self.on_extract)
        self.save_output_btn.clicked.connect(self.on_save_output_as)

        # Region signals
        self.region_enable_chk.toggled.connect(self._update_region_boxes_enabled)
        self.region_preview_btn.clicked.connect(self._preview_region)
        self.region_pick_btn.clicked.connect(self._apply_interactive_region)

        # Initial state
        self._update_region_boxes_enabled()
        self.on_codec_change(self.codec_combo.currentText())

    # ---------------- Region Helpers ----------------
    def _update_region_boxes_enabled(self):
        enable = self.region_enable_chk.isChecked() and isinstance(self.codec, Mp4Codec)
        for sb in (self.region_x, self.region_y, self.region_w, self.region_h):
            sb.setEnabled(enable)
        self.region_preview_btn.setEnabled(enable and self.carrier and self.codec.accepts(self.carrier))
        self.region_pick_btn.setEnabled(enable and self.carrier and isinstance(self.codec, Mp4Codec))

    def _current_region(self):
        if not (self.region_enable_chk.isChecked() and isinstance(self.codec, Mp4Codec)):
            return None
        x = self.region_x.value()
        y = self.region_y.value()
        w = self.region_w.value()
        h = self.region_h.value()
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)

    def _apply_interactive_region(self):
        if not (self.carrier and isinstance(self.codec, Mp4Codec)):
            self.status.setText("Load a video carrier first.")
            return
        try:
            reg = interactive_select_region(self.carrier)
            if reg is None:
                self.status.setText("Interactive selection canceled.")
                return
            x,y,w,h = reg
            self.region_x.setValue(x)
            self.region_y.setValue(y)
            self.region_w.setValue(w)
            self.region_h.setValue(h)
            self.status.setText(f"Region set to (x={x}, y={y}, w={w}, h={h})")
        except Exception as e:
            self.status.setText(f"Region pick failed: {e}")

    def _preview_region(self):
        if not (self.carrier and isinstance(self.codec, Mp4Codec)):
            self.status.setText("Load a video carrier first.")
            return
        reg = self._current_region()
        if not reg:
            self.status.setText("Enable region and set non-zero W/H before preview.")
            return
        x,y,w,h = reg
        try:
            # Save a temporary preview image beside the video
            tmp = Path(self.carrier).with_name(Path(self.carrier).stem + "__region_preview.png")
            preview_region(self.carrier, x,y,w,h, frame_index=0, window=False, save_path=tmp)
            # Show inside diff view
            im = Image.open(tmp).convert("RGB")
            self.view_diff.set_image_from_array(np.array(im))
            self.status.setText(f"Region preview saved: {tmp.name}")
        except Exception as e:
            self.status.setText(f"Preview failed: {e}")
    # ------------------------------------------------

    def on_codec_change(self, txt: str):
        self.codec_name = txt
        self.codec = CODECS[txt]

        # Show/hide region controls appropriately
        if isinstance(self.codec, ImageCodec):
            # Hide both region groups and the audio start spin for images
            self.region_group.hide()
            self.audio_region_group.hide()
            self.start_sample_spin.hide()
            self.view_video.hide()
            self.view_orig.show(); self.view_steg.show(); self.view_diff.show()

        elif isinstance(self.codec, WavCodec):
            # Show audio region (Start Sample), hide video region
            self.audio_region_group.show()
            self.region_group.hide()
            self.start_sample_spin.show()
            self.view_video.hide()
            self.view_orig.show(); self.view_steg.show(); self.view_diff.show()

        elif isinstance(self.codec, Mp4Codec):
            # Show video region, hide audio region (start sample)
            self.region_group.show()
            self.audio_region_group.hide()
            self.start_sample_spin.hide()
            # Video preview handled later when carrier is loaded

        self.status.setText(f"Carrier type set to: {txt}")
        self._update_region_boxes_enabled()

    def on_carrier(self, p: Path):
        self.carrier = p
        ext = p.suffix.lower()
        auto = EXT_TO_CODEC.get(ext)
        if auto and self.codec_combo.currentText() != auto:
            self.codec_combo.setCurrentText(auto)
        if isinstance(self.codec, ImageCodec):
            try:
                im = Image.open(p)
                rgb = im.convert("RGBA" if im.mode=="RGBA" else "RGB")
                self.view_video.hide()
                self.view_orig.show(); self.view_steg.show(); self.view_diff.show()
                self.view_orig.set_image_from_array(np.array(rgb))
            except Exception as e:
                self.view_orig.setText(f"No preview:\n{e}")
        elif isinstance(self.codec, Mp4Codec):
            try:
                self.view_orig.hide()
                self.view_steg.show(); self.view_diff.show(); self.view_video.show()
                self.view_video.load(p)
                # Update spin box limits based on video frame
                cap = cv2.VideoCapture(str(p))
                ok, frame = cap.read()
                cap.release()
                if ok:
                    H, W, _ = frame.shape
                    self.region_x.setRange(0, max(0, W-1))
                    self.region_y.setRange(0, max(0, H-1))
                    self.region_w.setRange(0, W)
                    self.region_h.setRange(0, H)
                self._update_region_boxes_enabled()
            except Exception as e:
                self.status.setText(f"No video preview: {e}")
        else:
            self.view_orig.setText(p.name)
        self._update_region_boxes_enabled()

    def on_stego(self, p: Path):
        self.stego = p
        if isinstance(self.codec, ImageCodec):
            try:
                im = Image.open(p).convert("RGB")
                self.view_steg.set_image_from_array(np.array(im))
            except Exception:
                self.view_steg.setText(p.name)
        elif isinstance(self.codec, Mp4Codec):
            self.view_steg.setText("Stego video selected (preview not implemented)")
        else:
            self.view_steg.setText(p.name)

    def on_payload(self, p: Path):
        self.payload = p
        self.status.setText(f"Payload set: {p.name}")

    def on_embed(self):
        if not self.carrier or not self.payload_widget.has_payload():
            self.status.setText("Select a carrier and provide a payload (file or text).")
            return
        if not self.codec.accepts(self.carrier):
            self.status.setText(f"{self.codec.pretty} expects a different carrier type.")
            return

        bpc = int(self.bpc_spin.value())
        key = self.key_edit.text()
        payload = self.payload_widget.payload_bytes()

        if not payload:
            self.status.setText("Payload is empty.")
            return

        region = self._current_region() if isinstance(self.codec, Mp4Codec) else None
        if region:
            x,y,w,h = region
            if w <= 0 or h <= 0:
                self.status.setText("Region width/height must be > 0.")
                return

        try:
            stem = Path(self.carrier).stem
            out_path = Path(self.carrier).with_name(f"{stem}__steg")
            # Pass region if video
            if isinstance(self.codec, Mp4Codec):
                result = self.codec.embed(self.carrier, payload, out_path, bpc, key, region=region)
                self.view_diff.setText("Video embedding complete (region mode)" if region else "Video embedding complete (full frame)")
                out_file = Path(result["out"])
            elif isinstance(self.codec, ImageCodec):
                result = self.codec.embed(self.carrier, payload, out_path, bpc, key)
                self.view_steg.set_image_from_array(result["steg"])
                self.view_orig.set_image_from_array(result["orig"])
                mask_rgb = np.stack([result["mask"]]*3, axis=2)
                self.view_diff.set_image_from_array(mask_rgb)
                out_file = Path(result.get("out_path", out_path.with_suffix(".png")))
            elif isinstance(self.codec, WavCodec):
                start_sample = int(self.start_sample_spin.value())
                result = self.codec.embed(self.carrier, payload, out_path, bpc, key, start_sample=start_sample)
                self.view_diff.setText(f"Modified samples ≈ {result['changed_pct']:.2f}%")
                out_file = Path(result["out"])
            else:
                # others
                result = self.codec.embed(self.carrier, payload, out_path, bpc, key)
                self.view_diff.setText(f"Modified samples ≈ {result['changed_pct']:.2f}%")
                out_file = Path(result["out"])

            self.last_output_path = out_file
            self.save_output_btn.setEnabled(True)
            self.status.setText(f"✅ Embedded → {out_file.name}")

        except Exception as e:
            self.status.setText(f"❌ Embed failed: {e}")

    def on_extract(self):
        if not self.stego:
            self.status.setText("Drop a stego file first."); return
        if not self.codec.accepts(self.stego):
            self.status.setText(f"{self.codec.pretty} expects a different stego file type."); return
        bpc = int(self.bpc_spin.value()); key = self.key_edit.text()
        region = self._current_region() if isinstance(self.codec, Mp4Codec) else None
        try:
            if isinstance(self.codec, Mp4Codec):
                data = self.codec.extract(self.stego, bpc, key, region=region)
            elif isinstance(self.codec, WavCodec):
                start_sample = int(self.start_sample_spin.value())
                data = self.codec.extract(self.stego, bpc, key, start_sample=start_sample)
            else:
                data = self.codec.extract(self.stego, bpc, key)
            out = Path(self.stego).with_name(Path(self.stego).stem + "__recovered.bin")
            out.write_bytes(data)
            self.status.setText(f"✅ Extracted payload → {out.name}")
        except Exception as e:
            msg = str(e)
            if "Invalid/missing header" in msg and isinstance(self.codec, WavCodec):
                self.status.setText("❌ Invalid header. Trying other bpc values...")
                self._attempt_other_bpcs(key)
            else:
                self.status.setText(f"❌ Extract failed: {e}")

    def _attempt_other_bpcs(self, key: str):
        if not self.stego or not isinstance(self.codec, WavCodec):
            return
        successes = []
        for test_bpc in range(1,9):
            try:
                start_sample = int(self.start_sample_spin.value())
                data = self.codec.extract(self.stego, test_bpc, key, start_sample=start_sample)
                successes.append((test_bpc, data))
            except Exception:
                continue
        if len(successes) == 1:
            bpc_found, data = successes[0]
            out = Path(self.stego).with_name(Path(self.stego).stem + f"__recovered_bpc{bpc_found}.bin")
            out.write_bytes(data)
            self.bpc_spin.setValue(bpc_found)
            self.status.setText(f"✅ Auto-detected bpc={bpc_found}. Extracted → {out.name}")
        elif len(successes) > 1:
            self.status.setText(f"❌ Multiple possible bpc values ({[b for b,_ in successes]}). Please recall which was used.")
        else:
            self.status.setText("❌ Could not find valid header for any bpc 1..8.")

    def on_save_output_as(self):
        if not self.last_output_path or not Path(self.last_output_path).exists():
            self.status.setText("No output to save yet."); return
        suffix = self.last_output_path.suffix.lower()
        if suffix == ".png":
            filt = "PNG Image (*.png);;All Files (*)"
        elif suffix == ".wav":
            filt = "WAV Audio (*.wav);;All Files (*)"
        elif suffix == ".mp4":
            filt = "MP4 Video (*.mp4);;All Files (*)"
        else:
            filt = "All Files (*)"
        dest, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Stego Output As…",
                        str(self.last_output_path.name), filt)
        if not dest: return
        try:
            Path(dest).write_bytes(Path(self.last_output_path).read_bytes())
            self.status.setText(f"✅ Saved a copy to: {dest}")
        except Exception as e:
            self.status.setText(f"❌ Save failed: {e}")