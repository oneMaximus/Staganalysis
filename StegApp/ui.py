from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QRunnable, QThreadPool, pyqtSignal, QObject
from pathlib import Path
from image_codec import ImageCodec
from wav_codec import WavCodec
from PIL import Image
from typing import Optional, Tuple
# UPDATED import to include region helpers
from mp4_codec import Mp4Codec, preview_region, interactive_select_region
from analysis_widget import AnalysisWidget
from stego_manager import embed as stego_embed, extract as stego_extract, capacity as stego_capacity
from wav_analysis_widget import WavAnalysisWidget   # NEW: WAV analysis tab
import traceback
import numpy as np
import cv2
# browse
from wav_analysis_widget import WavAnalysisWidget
from dropbox_widget import DropBox

CODECS = {
    "Image (PNG/BMP/TIFF)": ImageCodec(),
    "Audio (WAV PCM)"     : WavCodec(),
    "Video (MP4 H.264)"   : Mp4Codec(),
}

EXT_TO_CODEC = {
    ".mp4": "Video (MP4 H.264)",
    ".mov": "Video (MP4 H.264)",
    ".avi": "Video (MP4 H.264)",
    ".m4v": "Video (MP4 H.264)",
    ".wav": "Audio (WAV PCM)",
    ".png": "Image (PNG/BMP/TIFF)",
    ".bmp": "Image (PNG/BMP/TIFF)",
    ".tif": "Image (PNG/BMP/TIFF)",
    ".tiff":"Image (PNG/BMP/TIFF)",
    ".jpg": "Image (PNG/BMP/TIFF)",
    ".jpeg":"Image (PNG/BMP/TIFF)",
}


# -------- helper style for visible arrows ----------
class ArrowStyle(QtWidgets.QProxyStyle):
    def __init__(self, base="Fusion"):
        super().__init__(base)

    def _paint_triangle(self, painter: QtGui.QPainter, rect: QtCore.QRect, up: bool):
        size = min(rect.width(), rect.height(), 12)
        cx = rect.center().x()
        cy = rect.center().y()
        half = size // 2
        if up:
            p1 = QtCore.QPointF(cx, cy - half + 1)
            p2 = QtCore.QPointF(cx + half, cy + half - 1)
            p3 = QtCore.QPointF(cx - half, cy + half - 1)
        else:
            p1 = QtCore.QPointF(cx - half, cy - half + 1)
            p2 = QtCore.QPointF(cx + half, cy - half + 1)
            p3 = QtCore.QPointF(cx, cy + half - 1)

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#1F2430")))
        painter.drawPolygon(QtGui.QPolygonF([p1, p2, p3]))
        painter.restore()

    def drawPrimitive(self, element, option, painter, widget=None):
        if element in (QtWidgets.QStyle.PE_IndicatorArrowDown, QtWidgets.QStyle.PE_IndicatorSpinDown):
            self._paint_triangle(painter, option.rect, up=False)
            return
        if element in (QtWidgets.QStyle.PE_IndicatorArrowUp, QtWidgets.QStyle.PE_IndicatorSpinUp):
            self._paint_triangle(painter, option.rect, up=True)
            return
        return super().drawPrimitive(element, option, painter, widget)


# -------- worker threads ----------
class WorkerSignals(QObject):
    finished = pyqtSignal(object)   # result
    error = pyqtSignal(tuple)       # (exctype, value, traceback)

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit((type(e), e, traceback.format_exc()))


# -------- payload box ----------
class PayloadWidget(QtWidgets.QGroupBox):
    changed = QtCore.pyqtSignal()
    def __init__(self, title="Payload"):
        super().__init__(title)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("payloadTabs")
        tb = self.tabs.tabBar()
        tb.setUsesScrollButtons(False)
        tb.setExpanding(False)
        tb.setElideMode(QtCore.Qt.ElideNone)
        self.tabs.setMovable(False)
        self.tabs.setTabBarAutoHide(False)
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
        return (self.tabs.currentIndex() == 0 and self._file_path is not None) \
               or (self.tabs.currentIndex() == 1 and len(self.text_edit.toPlainText()) > 0)

    def payload_name(self) -> str:
        if self.tabs.currentIndex() == 0 and self._file_path:
            return self._file_path.name
        txt = self.text_edit.toPlainText()
        return f"text:{len(txt)} chars"

    def payload_bytes(self) -> bytes:
        if self.tabs.currentIndex() == 0:
            if not self._file_path:
                return b""
            try:
                return Path(self._file_path).read_bytes()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to read payload file:\n{e}")
                return b""
        enc = self.encoding.currentText()
        return self.text_edit.toPlainText().encode(enc, errors="replace")

    def _on_file(self, p: Path):
        self._file_path = p
        self.changed.emit()


# -------- selectable image view (mouse region) ----------
class SelectableImageView(QtWidgets.QLabel):
    regionChanged = QtCore.pyqtSignal(object)  # (x,y,w,h) or None

    def __init__(self, title: str):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setObjectName("previewCard")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setMinimumSize(240, 240)
        self.setToolTip(title)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMaximumHeight(300)

        self._img_arr: Optional[np.ndarray] = None
        self._pixmap_rect = QtCore.QRect()
        self._select_enabled = False
        self._rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._drag_origin = None
        self._current_sel = None  # QRect in widget coords

    def enable_selection(self, on: bool):
        self._select_enabled = on
        if not on:
            self.clear_selection()

    def clear_selection(self):
        self._rubber.hide()
        self._drag_origin = None
        self._current_sel = None
        self.regionChanged.emit(None)

    def selected_region_imgspace(self) -> Optional[Tuple[int,int,int,int]]:
        if self._img_arr is None or self._current_sel is None or self._pixmap_rect.isNull():
            return None
        r = self._current_sel.intersected(self._pixmap_rect)
        if r.isNull():
            return None
        img_h, img_w = self._img_arr.shape[:2]
        sx = img_w / self._pixmap_rect.width()
        sy = img_h / self._pixmap_rect.height()
        x = max(0, min(img_w, int((r.x() - self._pixmap_rect.x()) * sx)))
        y = max(0, min(img_h, int((r.y() - self._pixmap_rect.y()) * sy)))
        w = max(0, min(img_w - x, int(r.width() * sx)))
        h = max(0, min(img_h - y, int(r.height() * sy)))
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)

    def set_image_from_array(self, arr: np.ndarray):
        self._img_arr = arr
        if arr.ndim == 2:
            h,w = arr.shape
            qimg = QtGui.QImage(arr.data, w, h, w, QtGui.QImage.Format_Grayscale8)
        else:
            h,w,c = arr.shape
            if c == 3:
                qimg = QtGui.QImage(arr.data, w, h, 3*w, QtGui.QImage.Format_RGB888)
            else:
                qimg = QtGui.QImage(arr.data, w, h, 4*w, QtGui.QImage.Format_RGBA8888)
        scaled = qimg.scaled(self.width(), self.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        pix = QtGui.QPixmap.fromImage(scaled)
        self.setPixmap(pix)
        x = (self.width()  - scaled.width())  // 2
        y = (self.height() - scaled.height()) // 2
        self._pixmap_rect = QtCore.QRect(x, y, scaled.width(), scaled.height())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._img_arr is not None:
            self.set_image_from_array(self._img_arr)
        if self._current_sel:
            self._rubber.setGeometry(self._current_sel)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if not self._select_enabled or self._img_arr is None:
            return
        if e.button() == QtCore.Qt.LeftButton and self._pixmap_rect.contains(e.pos()):
            self._drag_origin = e.pos()
            self._current_sel = QtCore.QRect(self._drag_origin, QtCore.QSize())
            self._rubber.setGeometry(self._current_sel)
            self._rubber.show()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self._drag_origin is not None:
            self._current_sel = QtCore.QRect(self._drag_origin, e.pos()).normalized()
            self._rubber.setGeometry(self._current_sel)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if self._drag_origin is not None and e.button() == QtCore.Qt.LeftButton:
            self._current_sel = QtCore.QRect(self._drag_origin, e.pos()).normalized()
            self._rubber.setGeometry(self._current_sel)
            self._drag_origin = None
            self.regionChanged.emit(self.selected_region_imgspace())


# -------- fixed video/audio previews ----------
class FixedVideoWidget(QVideoWidget):
    def sizeHint(self):
        return QtCore.QSize(480,270)

class VideoPlayer(QtWidgets.QWidget):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("previewCard")
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.title_lbl = QtWidgets.QLabel(title); lay.addWidget(self.title_lbl)
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.video_widget = FixedVideoWidget()
        self.video_widget.setMinimumSize(480, 240)
        self.video_widget.setMaximumHeight(240)
        self.video_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        if hasattr(self.video_widget,"setAspectRatioMode"):
            self.video_widget.setAspectRatioMode(QtCore.Qt.KeepAspectRatio)
        lay.addWidget(self.video_widget)
        self.player.setVideoOutput(self.video_widget); self.player.setMuted(True)
        self.cv_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.cv_label.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.cv_label.setMinimumSize(480, 240)
        self.cv_label.setMaximumHeight(240)
        self.cv_label.setScaledContents(False)
        self.cv_label.hide()
        self.cv_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        lay.addWidget(self.cv_label)
        ctrl = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.pos_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.pos_slider.setRange(0, 0)
        self.time_lbl = QtWidgets.QLabel("00:00 / 00:00")
        ctrl.addWidget(self.btn_play); ctrl.addWidget(self.btn_pause); ctrl.addWidget(self.btn_stop)
        ctrl.addSpacing(8); ctrl.addWidget(self.pos_slider, 1); ctrl.addWidget(self.time_lbl)
        lay.addLayout(ctrl)

        self.btn_play.clicked.connect(self._play)
        self.btn_pause.clicked.connect(self._pause)
        self.btn_stop.clicked.connect(self._stop)
        self.pos_slider.sliderMoved.connect(self._set_position)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)

        self._ms = 0
        self._dur_ms = 0
        self._cap = None
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self._path = None
        self._fps_interval = 33
        if hasattr(self.player,"errorOccurred"):
            self.player.errorOccurred.connect(self._on_qt_error)
        else:
            self.player.error.connect(self._on_qt_error)
        self.player.mediaStatusChanged.connect(self._on_media_status)

    def load(self, path: Path):
        self.stop()
        self._path = Path(path)
        self.video_widget.hide()
        self._start_cv_fallback(self._path)

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
        self._fps_interval = int(max(15, 1000.0 / fps))
        self._timer.start(self._fps_interval)
        self.cv_label.show()
        count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._dur_ms = int(count * (1000.0 / (fps or 30.0)))
        self._ms = 0
        self.pos_slider.setRange(0, self._dur_ms)
        self._update_time_label(0, self._dur_ms)

    def _next_frame(self):
        if not self._cap: return
        ok, frame = self._cap.read()
        if not ok:
            if self._cap is not None:
                try: self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                except: pass
            if self._timer.isActive(): self._timer.stop()
            self._ms = 0
            self.pos_slider.blockSignals(True)
            self.pos_slider.setValue(0)
            self.pos_slider.blockSignals(False)
            self._update_time_label(0, self._dur_ms)
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = frame.shape
        qimg = QtGui.QImage(frame.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.cv_label.width(), self.cv_label.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.cv_label.setPixmap(pix)
        self._ms += self._fps_interval
        self.pos_slider.blockSignals(True)
        self.pos_slider.setValue(min(self._ms, self._dur_ms))
        self.pos_slider.blockSignals(False)
        self._update_time_label(self._ms, self._dur_ms)

    def _play(self):
        if self.cv_label.isVisible():
            if self._cap is None and self._path:
                self._start_cv_fallback(self._path); return
            if not self._timer.isActive():
                self._timer.start(self._fps_interval)
        else:
            self.player.play()

    def _pause(self):
        if self._cap is not None and self.cv_label.isVisible():
            if self._timer.isActive():
                self._timer.stop()
        else:
            self.player.pause()

    def _stop(self):
        if self._cap is not None and self.cv_label.isVisible():
            if self._timer.isActive():
                self._timer.stop()
            try: self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except: pass
            self._ms = 0
            self.pos_slider.setValue(0)
            self._update_time_label(0, self._dur_ms)
            self.cv_label.clear()
        else:
            self.player.stop()

    def _set_position(self, pos: int):
        if self._cap is not None and self.cv_label.isVisible():
            total = max(1, self._dur_ms)
            frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            target = int((pos/total) * frames)
            try: self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            except: pass
            self._ms = pos
        else:
            self.player.setPosition(pos)

    def _on_position(self, pos: int):
        if self._cap is not None and self.cv_label.isVisible():
            return
        self.pos_slider.blockSignals(True)
        self.pos_slider.setValue(pos)
        self.pos_slider.blockSignals(False)
        self._update_time_label(pos, int(self.player.duration() or 0))

    def _on_duration(self, dur: int):
        if self._cap is not None and self.cv_label.isVisible():
            self.pos_slider.setRange(0, self._dur_ms or 0)
        else:
            self.pos_slider.setRange(0, int(dur or 0))
            self._update_time_label(int(self.player.position() or 0), int(dur or 0))

    def _update_time_label(self, pos_ms: int, dur_ms: int):
        def fmt(ms):
            s = int(ms/1000); m,s = divmod(s,60); return f"{m:02d}:{s:02d}"
        self.time_lbl.setText(f"{fmt(pos_ms)} / {fmt(dur_ms)}")

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


class AudioPlayer(QtWidgets.QWidget):
    def __init__(self, title: str = "Audio Preview"):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(6,6,6,6)
        self.title_lbl = QtWidgets.QLabel(title)
        lay.addWidget(self.title_lbl)
        self.player = QMediaPlayer()
        self.player.setVolume(80)
        ctrl = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.pos_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pos_slider.setRange(0, 0)
        self.time_lbl = QtWidgets.QLabel("00:00 / 00:00")
        self.vol_lbl = QtWidgets.QLabel("Vol:")
        self.vol_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.vol_slider.setRange(0, 100); self.vol_slider.setValue(80)
        self.vol_slider.setFixedWidth(120)
        ctrl.addWidget(self.btn_play)
        ctrl.addWidget(self.btn_pause)
        ctrl.addWidget(self.btn_stop)
        ctrl.addSpacing(10)
        ctrl.addWidget(self.pos_slider, 1)
        ctrl.addWidget(self.time_lbl)
        ctrl.addSpacing(10)
        ctrl.addWidget(self.vol_lbl)
        ctrl.addWidget(self.vol_slider)
        lay.addLayout(ctrl)
        self.btn_play.clicked.connect(self.player.play)
        self.btn_pause.clicked.connect(self.player.pause)
        self.btn_stop.clicked.connect(self.player.stop)
        self.vol_slider.valueChanged.connect(self.player.setVolume)
        self.pos_slider.sliderMoved.connect(self.player.setPosition)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.setMinimumHeight(120)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def load(self, path: Path):
        self.player.setMedia(QMediaContent(QtCore.QUrl.fromLocalFile(str(path))))
        self.player.setPosition(0)
        self.time_lbl.setText("00:00 / 00:00")

    def _fmt(self, ms: int) -> str:
        s = int(ms/1000)
        m, s = divmod(s, 60)
        return f"{m:02d}:{s:02d}"

    def _on_position(self, pos: int):
        self.pos_slider.setValue(pos)
        self.time_lbl.setText(f"{self._fmt(pos)} / {self._fmt(self.player.duration() or 0)}")

    def _on_duration(self, dur: int):
        self.pos_slider.setRange(0, dur)

    def stop(self):
        try: self.player.stop()
        except: pass

    def closeEvent(self, e):
        self.stop(); super().closeEvent(e)


# ----------- Main Window -----------
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.threadpool = QThreadPool()
        self.setWindowTitle("Steg Lab — Image, WAV, MP4 LSB")
        self.codec_name = "Image (PNG/BMP/TIFF)"
        self.codec = CODECS[self.codec_name]
        self.carrier: Optional[Path] = None
        self.payload: Optional[Path] = None
        self.stego: Optional[Path] = None

        # Theme selector
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems(["System", "Light", "Dark"])
        self.theme_combo.setCurrentText("System")
        self.theme_combo.currentTextChanged.connect(self.apply_theme)

        self.view_video = VideoPlayer("Video Preview")
        self.audio_player = AudioPlayer("Audio Preview")

        self.codec_combo = QtWidgets.QComboBox(); self.codec_combo.addItems(CODECS.keys())
        self.box_carrier = DropBox("Carrier")
        self.payload_widget = PayloadWidget("Payload")
        self.box_stego   = DropBox("Stego (for Extract)")

        # Clear buttons
        self.clear_carrier_btn = QtWidgets.QPushButton("Clear Carrier")
        self.clear_payload_btn = QtWidgets.QPushButton("Clear Payload")
        self.clear_stego_btn   = QtWidgets.QPushButton("Clear Stego")

        self.bpc_spin = QtWidgets.QSpinBox(); self.bpc_spin.setRange(1,8); self.bpc_spin.setValue(1)
        self.bpc_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.bpc_spin.setMinimumHeight(30)

        self.key_edit = QtWidgets.QLineEdit(); self.key_edit.setPlaceholderText("Key (optional)")
        self.embed_btn = QtWidgets.QPushButton("Embed ▶")
        self.extract_btn = QtWidgets.QPushButton("Extract ⏏")
        self.status = QtWidgets.QLabel("Ready."); self.status.setWordWrap(True); self.status.setObjectName("status")

        # PREVIEWS (Original is selectable)
        self.view_orig = SelectableImageView("Original (drag to select embedding region)")
        self.view_steg = SelectableImageView("Embedded")
        self.view_diff = SelectableImageView("Change map / metric")

        # ---- Image Region UI ----
        self.img_region_group = QtWidgets.QGroupBox("Image Region (click & drag on 'Original')")
        ir = QtWidgets.QHBoxLayout(self.img_region_group)
        self.img_region_enable = QtWidgets.QCheckBox("Enable image region")
        self.img_region_clear  = QtWidgets.QPushButton("Clear")
        ir.addWidget(self.img_region_enable); ir.addStretch(1); ir.addWidget(self.img_region_clear)

        self.last_output_path: Optional[Path] = None
        self.save_output_btn = QtWidgets.QPushButton("Save Output As…"); self.save_output_btn.setEnabled(False)

        self.tabs = QtWidgets.QTabWidget(self)
        self.embed_tab = QtWidgets.QScrollArea(); self.embed_tab.setWidgetResizable(True)
        self._embed_page = QtWidgets.QWidget(); self.embed_tab.setWidget(self._embed_page)
        self.tabs.addTab(self.embed_tab, "Embed / Extract")
        embed_grid = QtWidgets.QGridLayout(self._embed_page)
        self.tabs.tabBar().setElideMode(QtCore.Qt.ElideNone)
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setUsesScrollButtons(True)
        embed_grid.setVerticalSpacing(12)

        form = QtWidgets.QFormLayout()
        form.addRow("Carrier Type:", self.codec_combo)
        form.addRow("LSBs per channel:", self.bpc_spin)
        form.addRow("Key:", self.key_edit)
        embed_grid.addLayout(form, 0,0,1,3)

        # Audio Region (WAV)
        self.audio_region_group = QtWidgets.QGroupBox("Audio Region (LSB area)")
        ag_lay = QtWidgets.QGridLayout(self.audio_region_group)
        ag_lay.setContentsMargins(12, 18, 12, 12)
        self.audio_region_enable_chk = QtWidgets.QCheckBox("Enable audio start (WAV only)")
        self.audio_region_enable_chk.setChecked(True)
        ag_lay.addWidget(self.audio_region_enable_chk, 0, 0, 1, 3)
        ag_lay.addWidget(QtWidgets.QLabel("Start Sample:"), 1, 0)
        self.start_sample_spin = QtWidgets.QSpinBox()
        self.start_sample_spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.start_sample_spin.setRange(0, 1_000_000_000)
        self.start_sample_spin.setValue(0)
        self.start_sample_spin.setToolTip("WAV only: frame index to start embedding/extracting (header+payload).")
        ag_lay.addWidget(self.start_sample_spin, 1, 1)
        self.audio_region_enable_chk.toggled.connect(self.start_sample_spin.setEnabled)
        self.audio_region_group.hide()
        embed_grid.addWidget(self.audio_region_group, 1, 0, 1, 3)

        # Video Region (MP4)
        self.region_group = QtWidgets.QGroupBox("Video Region (LSB area)")
        rg_lay = QtWidgets.QGridLayout(self.region_group)
        rg_lay.setContentsMargins(12, 18, 12, 12)
        self.region_enable_chk = QtWidgets.QCheckBox("Enable region (video only)")
        self.region_x = QtWidgets.QSpinBox(); self.region_x.setRange(0, 100000)
        self.region_y = QtWidgets.QSpinBox(); self.region_y.setRange(0, 100000)
        self.region_w = QtWidgets.QSpinBox(); self.region_w.setRange(0, 100000)
        self.region_h = QtWidgets.QSpinBox(); self.region_h.setRange(0, 100000)
        self.region_x.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.region_y.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.region_w.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.region_h.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
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

        # Image Region group (visible for images)
        embed_grid.addWidget(self.img_region_group, 1, 0, 1, 3)

        # Side-by-side: Carrier | Payload | Stego
        embed_grid.addWidget(self.box_carrier, 2, 0)
        embed_grid.addWidget(self.payload_widget, 2, 1)
        embed_grid.addWidget(self.box_stego,     2, 2)

        # Clear buttons
        embed_grid.addWidget(self.clear_carrier_btn, 3, 0)
        embed_grid.addWidget(self.clear_payload_btn, 3, 1)
        embed_grid.addWidget(self.clear_stego_btn,   3, 2)

        # Actions row
        actions_row = QtWidgets.QHBoxLayout()
        actions_row.addStretch(1)
        actions_row.addWidget(self.embed_btn)
        actions_row.addSpacing(12)
        actions_row.addWidget(self.extract_btn)
        actions_row.addSpacing(12)
        actions_row.addWidget(self.save_output_btn)
        actions_row.addStretch(1)
        embed_grid.addLayout(actions_row, 4, 0, 1, 3)

        # Previews row
        self.preview_stack = QtWidgets.QStackedWidget()
        self.preview_stack.addWidget(self.view_orig)     # 0 image
        self.preview_stack.addWidget(self.view_video)    # 1 video
        self.preview_stack.addWidget(self.audio_player)  # 2 audio
        self.preview_stack.setCurrentIndex(0)

        self.view_orig.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.view_steg.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.view_diff.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.view_video.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.audio_player.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        previews = QtWidgets.QHBoxLayout()
        previews.addWidget(self.preview_stack, stretch=1)
        previews.addWidget(self.view_steg, stretch=1)
        previews.addWidget(self.view_diff, stretch=1)
        embed_grid.addLayout(previews, 5, 0, 1, 3)

        self.analysis_tab = AnalysisWidget()
        self.tabs.addTab(self.analysis_tab, "Steg Analysis")
        self.wav_analysis_tab = WavAnalysisWidget()
        self.tabs.addTab(self.wav_analysis_tab, "WAV Analysis")

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

        # Image selection controls
        self.img_region_enable.toggled.connect(self.view_orig.enable_selection)
        self.img_region_clear.clicked.connect(self.view_orig.clear_selection)
        self.view_orig.regionChanged.connect(self._on_image_region_changed)

        # Video region controls
        self.region_enable_chk.toggled.connect(self._update_region_boxes_enabled)
        self.region_preview_btn.clicked.connect(self._preview_region)
        self.region_pick_btn.clicked.connect(self._apply_interactive_region)

        self.apply_theme("Light")
        self._update_region_boxes_enabled()
        self.on_codec_change(self.codec_combo.currentText())

        # Clear buttons
        self.clear_carrier_btn.clicked.connect(lambda: self.clear_box("carrier"))
        self.clear_payload_btn.clicked.connect(lambda: self.clear_box("payload"))
        self.clear_stego_btn.clicked.connect(lambda: self.clear_box("stego"))

    # ----- image region status
    def _on_image_region_changed(self, reg):
        if reg is None:
            self.status.setText("Image region cleared.")
        else:
            x,y,w,h = reg
            self.status.setText(f"Image region: x={x}, y={y}, w={w}, h={h}")

    # ---------------- Region Helpers (video)
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
            self.region_enable_chk.setChecked(True)
            self._preview_region()
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
            tmp = Path(self.carrier).with_name(Path(self.carrier).stem + "__region_preview.png")
            preview_region(self.carrier, x,y,w,h, frame_index=0, window=False, save_path=tmp)
            im = Image.open(tmp).convert("RGB")
            self.view_diff.set_image_from_array(np.array(im))
            self.status.setText(f"Region preview saved: {tmp.name}")
        except Exception as e:
            self.status.setText(f"Preview failed: {e}")

    # ---------------- Theme
    def apply_theme(self, _: str = "Light"):
        app = QtWidgets.QApplication.instance()
        if not app:
            return
        try:
            QtWidgets.QApplication.setStyle(ArrowStyle("Fusion"))
        except Exception:
            pass
        ss = """
        QWidget { background-color: #F6F7FB; color: #1F2430; font-size: 14px; }
        QGroupBox { background: #FFFFFF; border: 1px solid #D9DEEA; border-radius: 12px; margin-top: 24px; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 4px 10px; color: #1565C0; font-weight: 600; font-size: 15px; }
        QLineEdit, QSpinBox, QPlainTextEdit, QTextEdit { background: #FFFFFF; color: #1F2430; border: 1px solid #C9D4EE; border-radius: 10px; padding: 8px 10px; }
        QAbstractSpinBox { padding-right: 34px; }
        QComboBox { background: #FFFFFF; color: #1F2430; border: 1px solid #C9D4EE; border-radius: 10px; padding: 6px 10px 6px 10px; }
        QPushButton { background-color: #1976D2; color: #FFFFFF; border: none; padding: 10px 14px; border-radius: 10px; font-weight: 600; font-size: 14px; }
        QPushButton:hover { background-color: #1565C0; }
        QPushButton:pressed { background-color: #0D47A1; }
        QPushButton:disabled { background-color: #E6EAF3; color: #8A94A6; }
        QPushButton[secondary="true"] { background-color: transparent; border: 2px solid #1976D2; color: #1976D2; }
        QPushButton[secondary="true"]:hover { background-color: #E3F2FD; }
        QPushButton[secondary="true"]:pressed { background-color: #BBDEFB; }
        QPushButton[tertiary="true"] { background-color: #F5F7FA; color: #555C68; border: 1px solid #D9DEEA; }
        QPushButton[tertiary="true"]:hover { background-color: #ECEFF4; }
        QTabWidget::pane { border: 1px solid #D9DEEA; top: -1px; background: #FFFFFF; border-radius: 10px; }
        QTabBar::tab { background: #FFFFFF; color: #1F2430; padding: 10px 16px; min-width: 140px; min-height: 34px; border: 1px solid #D9DEEA; border-bottom: none; border-top-left-radius: 10px; border-top-right-radius: 10px; margin-right: 4px; font-weight: 600; }
        QTabBar::tab:selected { background: #EEF1F8; color: #0F172A; }
        QSlider::groove:horizontal { height: 8px; background: #EEF1F8; border: 1px solid #D9DEEA; border-radius: 5px; }
        QSlider::handle:horizontal { background: #1976D2; width: 18px; margin: -7px 0; border-radius: 9px; }
        QScrollBar:vertical, QScrollBar:horizontal { background: #F4F6FB; border: 1px solid #D9DEEA; border-radius: 8px; width: 14px; height: 14px; }
        QScrollBar::handle { background: #D9DEEA; border-radius: 8px; }
        QLabel#status { color: #4B5B6B; font-size: 13px; }
        QLabel#previewCard, QWidget#previewCard { background: #FFFFFF; border: 1px solid #D9DEEA; border-radius: 12px; padding: 10px; }
        """
        app.setStyleSheet(ss)

    # ---------------- codec changes / file drops
    def on_codec_change(self, txt: str):
        self.codec_name = txt
        self.codec = CODECS[txt]

        if isinstance(self.codec, ImageCodec):
            self.preview_stack.setCurrentIndex(0)
            self.region_group.hide()
            self.audio_region_group.hide()
            self.img_region_group.show()
            self.start_sample_spin.hide()

        elif isinstance(self.codec, WavCodec):
            self.preview_stack.setCurrentIndex(2)
            self.audio_region_group.show()
            self.region_group.hide()
            self.img_region_group.hide()
            self.start_sample_spin.show()

        elif isinstance(self.codec, Mp4Codec):
            self.preview_stack.setCurrentIndex(1)
            self.region_group.show()
            self.audio_region_group.hide()
            self.img_region_group.hide()
            self.start_sample_spin.hide()

        self.status.setObjectName("status")
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
                self.view_orig.set_image_from_array(np.array(rgb))
                self.preview_stack.setCurrentIndex(0)
            except Exception as e:
                self.view_orig.setText(f"No preview:\n{e}")

        elif isinstance(self.codec, Mp4Codec):
            try:
                self.view_video.load(p)
                cap = cv2.VideoCapture(str(p))
                ok, frame = cap.read()
                cap.release()
                if ok:
                    H, W, _ = frame.shape
                    self.region_x.setRange(0, max(0, W-1))
                    self.region_y.setRange(0, max(0, H-1))
                    self.region_w.setRange(0, W)
                    self.region_h.setRange(0, H)
                self.preview_stack.setCurrentIndex(1)
                self._update_region_boxes_enabled()
            except Exception as e:
                self.status.setText(f"No video preview: {e}")

        elif isinstance(self.codec, WavCodec):
            try:
                self.audio_player.load(p)
                self.preview_stack.setCurrentIndex(2)
            except Exception as e:
                self.status.setText(f"No audio preview: {e}")
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
            self.view_steg.setText("Stego video selected")
        elif isinstance(self.codec, WavCodec):
            self.preview_stack.setCurrentIndex(2)
            self.audio_player.load(p)
        else:
            self.view_steg.setText(p.name)

    # ---------------- helpers
    def _image_region_for_ops(self) -> Optional[Tuple[int,int,int,int]]:
        if not isinstance(self.codec, ImageCodec):
            return None
        if not self.img_region_enable.isChecked():
            return None
        return self.view_orig.selected_region_imgspace()

    # ---------------- embed/extract
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

        stem = Path(self.carrier).stem
        self.status.setText("Embedding, please wait...")

        img_region = self._image_region_for_ops()

        def task():
            if isinstance(self.codec, WavCodec):
                start_sample = self.start_sample_spin.value() if self.audio_region_enable_chk.isChecked() else 0
                out_base = Path(self.carrier).with_name(f"{stem}__steg")
                return self.codec.embed(self.carrier, payload, out_base, bpc, key, start_sample=start_sample)
            else:
                return stego_embed(self.carrier, payload, f"{stem}__steg", bpc, key,
                                   image_region=img_region)

        worker = Worker(task)

        def done(result):
            try:
                out_file = Path(result["out"])
                self.last_output_path = out_file
                self.save_output_btn.setEnabled(True)

                if "steg" in result and "orig" in result:
                    self.view_steg.set_image_from_array(result["steg"])
                    self.view_orig.set_image_from_array(result["orig"])
                    if "mask" in result and img_region is not None:
                        mask_rgb = np.stack([result["mask"]] * 3, axis=2)
                        self.view_diff.set_image_from_array(mask_rgb)
                else:
                    self.view_diff.setText(f"{result['metric_label']}: {result['metric_value']}")

                bytes_emb = int(result.get("bytes_embedded", 0))
                if img_region:
                    x,y,w,h = img_region
                    self.status.setText(f"✅ Embedded {bytes_emb} bytes @ region (x={x},y={y},w={w},h={h}) → {out_file}")
                else:
                    self.status.setText(f"✅ Embedded {bytes_emb} bytes → {out_file}")
            except Exception as e:
                traceback.print_exc()
                self.status.setText(f"❌ Embed post-process failed: {e}")

        def fail(err):
            exctype, value, tb = err
            self.status.setText(f"❌ Embed failed: {value}")
            print(tb)

        worker.signals.finished.connect(done)
        worker.signals.error.connect(fail)
        self.threadpool.start(worker)

    def on_extract(self):
        if not self.stego:
            self.status.setText("Drop a stego file first."); 
            return
        if not self.codec.accepts(self.stego):
            self.status.setText(f"{self.codec.pretty} expects a different stego file type."); 
            return

        bpc = int(self.bpc_spin.value())
        key = self.key_edit.text()
        img_region = self._image_region_for_ops() if isinstance(self.codec, ImageCodec) else None

        self.status.setText("Extracting, please wait...")

        def task():
            if isinstance(self.codec, WavCodec):
                start_sample = self.start_sample_spin.value() if self.audio_region_enable_chk.isChecked() else 0
                return self.codec.extract(self.stego, bpc, key, start_sample=start_sample)
            else:
                return stego_extract(self.stego, bpc, key, image_region=img_region)

        worker = Worker(task)

        def done(data):
            try:
                out = Path(self.stego).with_name(Path(self.stego).stem + "__recovered.bin")
                out.write_bytes(data)
                if img_region:
                    x,y,w,h = img_region
                    self.status.setText(f"✅ Extracted payload from region (x={x},y={y},w={w},h={h}) → {out}")
                else:
                    self.status.setText(f"✅ Extracted payload → {out}")
            except Exception as e:
                traceback.print_exc()
                self.status.setText(f"Extract post-process failed: {e}")

        def fail(err):
            exctype, value, tb = err
            self.status.setText(f"Extract failed: {value}")
            print(tb)

        worker.signals.finished.connect(done)
        worker.signals.error.connect(fail)
        self.threadpool.start(worker)

    # -------------- misc
    def clear_box(self, which: str):
        if which == "carrier":
            self.carrier = None
            self.box_carrier.label.setText("Drop a file here")
            if isinstance(self.codec, ImageCodec):
                self.view_orig.clear()
            elif isinstance(self.codec, Mp4Codec):
                self.view_video.stop()
                self.view_video.cv_label.clear()
                self.view_video.video_widget.hide()
                self.view_video.cv_label.hide()
                self.preview_stack.setCurrentIndex(0)
            elif isinstance(self.codec, WavCodec):
                self.audio_player.stop()
                self.audio_player.time_lbl.setText("00:00 / 00:00")
                self.audio_player.hide()
                self.preview_stack.setCurrentIndex(0)
        elif which == "payload":
            self.payload = None
            self.payload_widget.drop.label.setText("Drop a file here")
            self.payload_widget.text_edit.clear()
        elif which == "stego":
            self.stego = None
            self.box_stego.label.setText("Drop a file here")
            self.view_steg.clear()
            self.view_diff.clear()
        self.status.setText(f"Cleared {which}.")

    def check_capacity(self):
        if not self.carrier or not self.payload_widget.has_payload():
            return
        try:
            bpc = int(self.bpc_spin.value())
            need = len(self.payload_widget.payload_bytes())
            if isinstance(self.codec, WavCodec):
                start_sample = self.start_sample_spin.value() if self.audio_region_enable_chk.isChecked() else 0
                cap = self.codec.capacity_bytes(self.carrier, bpc, start_sample=start_sample)
            else:
                cap = stego_capacity(self.carrier, bpc)
            if need > cap:
                self.status.setText(f"⚠️ Payload {need} B exceeds capacity {cap} B at {bpc} bpc.")
            else:
                self.status.setText(f"Capacity OK: need {need} B / have {cap} B.")
        except Exception as e:
            self.status.setText(f"Capacity check failed: {e}")

    def on_save_output_as(self):
        if not self.last_output_path or not Path(self.last_output_path).exists():
            self.status.setText("No output to save yet."); return
        suffix = self.last_output_path.suffix.lower()
        if suffix == ".png":
            filt = "PNG Image (*.png);;All Files (*)"
        elif suffix == ".wav":
            filt = "WAV Audio (*.wav);;All Files (*)"
        elif suffix in (".mp4", ".avi"):
            filt = "Video (*.mp4 *.avi);;All Files (*)"
        elif suffix in (".tif", ".tiff"):
            filt = "TIFF Image (*.tif *.tiff);;All Files (*)"
        elif suffix == ".bmp":
            filt = "BMP Image (*.bmp);;All Files (*)"
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
