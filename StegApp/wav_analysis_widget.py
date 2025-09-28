# wav_analysis_widget.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import wave
import numpy as np

from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ===================== WAV LOADING (Analysis Only) =====================

def load_wav(path: Path) -> Tuple[int, np.ndarray]:
    """
    Load a PCM WAV (8/16/24/32-bit integer) into an int16 numpy array for analysis.
    Returns (sample_rate, samples[channels, N]) channel-first.

    NOTE:
      - This is an ANALYSIS helper only. For >16-bit PCM we down-shift into a 16-bit
        dynamic range so plotting & bit-plane extraction remain simple and fast.
      - Actual embedding/extraction (wav_codec) still works at the real bit depth.
    """
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fr = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
        data = (data - 128) << 8
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2")
    elif sampwidth == 3:
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        sign = (a[:, 2] & 0x80) != 0
        ext = np.where(sign, 0xFF, 0x00).astype(np.uint8)
        b = np.column_stack([a, ext])
        data32 = b.view("<i4")[:, 0]
        data = (data32 >> 8).astype(np.int16)
    elif sampwidth == 4:
        data32 = np.frombuffer(raw, dtype="<i4")
        data = (data32 >> 16).astype(np.int16)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth * 8} bits")

    if nch > 1:
        data = data.reshape(-1, nch).T  # (channels, frames)
    else:
        data = data.reshape(1, -1)

    return fr, data


# ===================== Drop Label Widget =====================

class DropLabel(QtWidgets.QLabel):
    fileDropped = QtCore.pyqtSignal(Path)

    def __init__(self, text: str):
        super().__init__(text)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setFrameShape(QtWidgets.QFrame.Box)
        self.setMinimumHeight(40)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                if u.toLocalFile().lower().endswith(".wav"):
                    e.acceptProposedAction()
                    return

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if not urls:
            return
        p = Path(urls[0].toLocalFile())
        if p.suffix.lower() == ".wav":
            self.setText(p.name)
            self.fileDropped.emit(p)


# ===================== Main Analysis Widget =====================

class WavAnalysisWidget(QtWidgets.QWidget):
    """
    WAV Steganalysis / Visualization Widget with Difference Mode (FULL-FILE VIEW)

    Features:
      - Primary (reference) and optional Compare (stego) WAV drop targets
      - Overlay waveforms (blue = primary, orange = compare)
      - Dual bit-plane visualization (rows: primary, compare)
      - Bit-plane difference (XOR) map as dedicated subplot
      - LSB statistics (primary only)
      - Difference metrics: changed bits %, mean absolute sample difference, SNR
      - Amplitude histogram overlay
      - Optional spectrogram (primary subset)
      - Automatic waveform decimation if very large

    Bit-plane range is limited to 1..8 (1 = LSB).
    """

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(False)  # We use dedicated drop labels instead

        # ---- Drop Areas ----
        self.drop_primary = DropLabel("Drop PRIMARY (cover) WAV here")
        self.drop_compare = DropLabel("Drop COMPARE (stego/suspect) WAV here")

        self.drop_primary.fileDropped.connect(self._on_primary_dropped)
        self.drop_compare.fileDropped.connect(self._on_compare_dropped)

        # ---- Controls ----
        self.channel_box = QtWidgets.QComboBox()
        self.channel_box.addItems(["Left / Mono", "Right", "Mix (L+R)/2"])
        self.channel_box.setToolTip("Select which channel to analyze.\nRight falls back to Left if mono.\nMix = (L+R)/2 integer average.")

        self.bit_spin = QtWidgets.QSpinBox()
        self.bit_spin.setRange(1, 8)
        self.bit_spin.setValue(1)
        self.bit_spin.setPrefix("Bit ")
        self.bit_spin.setToolTip("Bit-plane (1 = LSB). Only lowest 8 planes shown.")

        self.spectro_chk = QtWidgets.QCheckBox("Spectrogram")
        self.spectro_chk.setToolTip("Show a small inset spectrogram (primary WAV only).")

        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.setToolTip("Force redraw (usually auto).")

        self.lsb_stats_lbl = QtWidgets.QLabel("LSB stats: --")
        self.lsb_stats_lbl.setMinimumWidth(240)

        self.diff_metrics_lbl = QtWidgets.QLabel("Diff: --")
        self.diff_metrics_lbl.setWordWrap(True)
        self.diff_metrics_lbl.setStyleSheet("QLabel { color: #444; }")

        ctrl_layout = QtWidgets.QHBoxLayout()
        for w in [
            QtWidgets.QLabel("Channel:"), self.channel_box,
            QtWidgets.QLabel("Bit-plane:"), self.bit_spin,
            self.lsb_stats_lbl,
            self.spectro_chk,
            self.refresh_btn
        ]:
            ctrl_layout.addWidget(w)
        ctrl_layout.addStretch(1)

        # ---- Matplotlib Figure ----
        # 4 rows now:
        # 1: Waveform overlay
        # 2: Bit-planes (primary/compare)
        # 3: Amplitude histogram
        # 4: Bit-plane DIFF (XOR)
        self.fig = Figure(figsize=(10, 8), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax_wave = self.fig.add_subplot(411)
        self.ax_planes = self.fig.add_subplot(412)
        self.ax_hist = self.fig.add_subplot(413)
        self.ax_diff = self.fig.add_subplot(414)
        self.ax_spectro = None  # inset on waveform

        # ---- State ----
        self.sample_rate_primary: Optional[int] = None
        self.samples_primary: Optional[np.ndarray] = None

        self.sample_rate_compare: Optional[int] = None
        self.samples_compare: Optional[np.ndarray] = None

        # ---- Signals ----
        for sig_widget in (self.channel_box, self.bit_spin, self.spectro_chk):
            sig_widget.currentIndexChanged.connect(self.refresh) if hasattr(sig_widget, "currentIndexChanged") else None
        self.bit_spin.valueChanged.connect(self.refresh)
        self.spectro_chk.stateChanged.connect(self.refresh)
        self.refresh_btn.clicked.connect(self.refresh)

        # ---- Layout ----
        top_box = QtWidgets.QHBoxLayout()
        top_box.addWidget(self.drop_primary)
        top_box.addWidget(self.drop_compare)

        outer = QtWidgets.QVBoxLayout(self)
        outer.addLayout(top_box)
        outer.addLayout(ctrl_layout)
        outer.addWidget(self.canvas)
        outer.addWidget(self.diff_metrics_lbl)

    # ===================== File Loading =====================

    def _on_primary_dropped(self, path: Path):
        try:
            sr, data = load_wav(path)
            self.sample_rate_primary = sr
            self.samples_primary = data
            self.drop_primary.setText(f"{path.name}\n{data.shape[1]:,} samples @ {sr} Hz | {data.shape[0]} ch")
        except Exception as ex:
            QtWidgets.QMessageBox.warning(self, "Load Error (Primary)", str(ex))
            return
        self.refresh()

    def _on_compare_dropped(self, path: Path):
        try:
            sr, data = load_wav(path)
            self.sample_rate_compare = sr
            self.samples_compare = data
            self.drop_compare.setText(f"{path.name}\n{data.shape[1]:,} samples @ {sr} Hz | {data.shape[0]} ch")
        except Exception as ex:
            QtWidgets.QMessageBox.warning(self, "Load Error (Compare)", str(ex))
            return
        self.refresh()

    # ===================== Helpers =====================

    def _select_channel(self, samples: np.ndarray, idx_mode: int) -> np.ndarray:
        """
        Return 1D channel data based on selection.
        idx_mode: 0=Left/Mono, 1=Right, 2=Mix
        """
        ch_count, _ = samples.shape
        if ch_count == 1:
            return samples[0]
        if idx_mode == 0:     # Left
            return samples[0]
        elif idx_mode == 1:   # Right
            return samples[1] if ch_count > 1 else samples[0]
        else:                 # Mix
            if ch_count == 1:
                return samples[0]
            return ((samples[0].astype(np.int32) +
                     samples[1].astype(np.int32)) // 2).astype(np.int16)

    def _bit_plane(self, data: np.ndarray, bit_index: int) -> np.ndarray:
        return (data.astype(np.int32) >> bit_index) & 1

    def _compute_lsb_stats(self, plane: np.ndarray) -> str:
        ones = int(plane.sum())
        zeros = plane.size - ones
        pct1 = ones * 100.0 / plane.size
        return f"LSB stats: 0={zeros:,} 1={ones:,} (1={pct1:5.2f}%)"

    def _safe_decimate(self, arr: np.ndarray, max_points: int = 2_000_000) -> np.ndarray:
        if arr.size <= max_points:
            return arr
        step = arr.size // max_points + 1
        return arr[::step]

    # ===================== Rendering =====================

    def refresh(self):
        self.ax_wave.clear()
        self.ax_planes.clear()
        self.ax_hist.clear()
        self.ax_diff.clear()
        self.ax_spectro = None

        if self.samples_primary is None:
            self.canvas.draw_idle()
            self.lsb_stats_lbl.setText("LSB stats: --")
            self.diff_metrics_lbl.setText("Diff: --")
            return

        ch_mode = self.channel_box.currentIndex()
        bit_plane_index = self.bit_spin.value() - 1  # 0..7

        # Primary channel data
        primary_chan = self._select_channel(self.samples_primary, ch_mode)
        primary_len = primary_chan.size
        sr_primary = self.sample_rate_primary or 0

        # Compare channel (optional)
        compare_chan = None
        sr_compare = None
        if self.samples_compare is not None:
            # Only proceed if sample rate matches (simple case)
            sr_compare = self.sample_rate_compare or 0
            if sr_compare == sr_primary and sr_primary > 0:
                compare_chan = self._select_channel(self.samples_compare, ch_mode)
            else:
                # Different sample rates: we could resample; for now just ignore compare
                compare_chan = None

        # ---------------- Waveform (Row 1) ----------------
        wave_primary = self._safe_decimate(primary_chan)
        if sr_primary > 0:
            time_factor = primary_len / wave_primary.size
            t_primary = (np.arange(wave_primary.size) * time_factor) / sr_primary
            self.ax_wave.plot(t_primary, wave_primary, lw=0.6, color="#005f99", label="Primary")
            self.ax_wave.set_xlabel("Time (s)")
        else:
            self.ax_wave.plot(wave_primary, lw=0.6, color="#005f99", label="Primary")
            self.ax_wave.set_xlabel("Sample index")

        if compare_chan is not None:
            wave_compare = self._safe_decimate(compare_chan)
            # Align lengths for plotting
            if wave_compare.size != wave_primary.size:
                # best-effort simple decimation alignment (not sample-accurate if huge difference)
                pass
            if sr_primary > 0:
                self.ax_wave.plot(t_primary, wave_compare[:wave_primary.size], lw=0.6, color="#ff8800", alpha=0.65, label="Compare")
            else:
                self.ax_wave.plot(wave_compare[:wave_primary.size], lw=0.6, color="#ff8800", alpha=0.65, label="Compare")

        self.ax_wave.set_title(f"Waveform (Primary length: {primary_len:,} samples{' + Compare' if compare_chan is not None else ''})")
        self.ax_wave.set_ylabel("Amplitude")
        if compare_chan is not None:
            self.ax_wave.legend(loc="upper right", fontsize=8)

        # Spectrogram inset (primary only)
        if self.spectro_chk.isChecked() and sr_primary > 0:
            self.ax_spectro = self.ax_wave.inset_axes([0.67, 0.05, 0.30, 0.40])
            spec_seg = primary_chan[:8192] if primary_len > 8192 else primary_chan
            self.ax_spectro.specgram(spec_seg.astype(np.float32),
                                     Fs=sr_primary,
                                     NFFT=1024,
                                     noverlap=512,
                                     cmap="magma")
            self.ax_spectro.set_xticks([])
            self.ax_spectro.set_yticks([])
            self.ax_spectro.set_title("Spec", fontsize=8)

        # ---------------- Bit-planes (Row 2) ----------------
        plane_primary_full = self._bit_plane(primary_chan, bit_plane_index).astype(np.uint8)
        self.lsb_stats_lbl.setText(self._compute_lsb_stats(plane_primary_full))

        if compare_chan is not None:
            plane_compare_full = self._bit_plane(compare_chan, bit_plane_index).astype(np.uint8)
            length_common = min(plane_primary_full.size, plane_compare_full.size)
            plane_img = np.vstack([
                plane_primary_full[:length_common].reshape(1, -1),
                plane_compare_full[:length_common].reshape(1, -1)
            ])
            self.ax_planes.imshow(plane_img, aspect="auto", cmap="gray", interpolation="nearest")
            self.ax_planes.set_yticks([0, 1])
            self.ax_planes.set_yticklabels(["Primary", "Compare"])
            self.ax_planes.set_title(f"Bit-plane {bit_plane_index + 1} (rows: Primary / Compare)")
        else:
            plane_img = plane_primary_full.reshape(1, -1)
            self.ax_planes.imshow(plane_img, aspect="auto", cmap="gray", interpolation="nearest")
            self.ax_planes.set_yticks([])
            self.ax_planes.set_title(f"Bit-plane {bit_plane_index + 1} (Primary)")

        # ---------------- Histogram (Row 3) ----------------
        hist_p, bins_p = np.histogram(primary_chan, bins=256, range=(-32768, 32767))
        centers_p = (bins_p[:-1] + bins_p[1:]) / 2
        self.ax_hist.bar(centers_p, hist_p, width=(bins_p[1] - bins_p[0]), color="#888888", label="Primary")

        if compare_chan is not None:
            hist_c, bins_c = np.histogram(compare_chan, bins=256, range=(-32768, 32767))
            centers_c = (bins_c[:-1] + bins_c[1:]) / 2
            self.ax_hist.plot(centers_c, hist_c, color="#ff8800", linewidth=1.0, label="Compare")
            self.ax_hist.legend(loc="upper right", fontsize=8)

        self.ax_hist.set_title("Amplitude Histogram (int16 domain)")
        self.ax_hist.set_xlabel("Amplitude")
        self.ax_hist.set_ylabel("Count")
        self.ax_hist.ticklabel_format(style="plain", axis="y")

        # ---------------- Bit-plane Diff (Row 4) ----------------
        diff_metrics_text = "Diff: (no compare file loaded)"
        if compare_chan is not None:
            plane_compare_full = self._bit_plane(compare_chan, bit_plane_index).astype(np.uint8)
            length_common = min(plane_primary_full.size, plane_compare_full.size)
            diff_plane = (plane_primary_full[:length_common] ^ plane_compare_full[:length_common]).astype(np.uint8)

            # Diff visualization
            self.ax_diff.imshow(diff_plane.reshape(1, -1), aspect="auto",
                                cmap="gray", interpolation="nearest")
            self.ax_diff.set_yticks([])
            title_suffix = ""
            if plane_primary_full.size != plane_compare_full.size:
                title_suffix = f" (trimmed {plane_primary_full.size:,}->{length_common:,})"
            self.ax_diff.set_title(f"Bit-plane XOR (changed bits white){title_suffix}")

            changed = int(diff_plane.sum())
            total = diff_plane.size
            changed_pct = (changed / total * 100.0) if total else 0.0

            # Sample-domain difference metrics (same trimmed length)
            primary_trim = primary_chan[:length_common].astype(np.int32)
            compare_trim = compare_chan[:length_common].astype(np.int32)
            abs_diff = np.abs(primary_trim - compare_trim)
            mean_abs = float(abs_diff.mean())

            # SNR estimate: signal power / noise power (dB)
            noise_power = float((abs_diff ** 2).mean())
            signal_power = float((primary_trim.astype(np.float64) ** 2).mean())
            if noise_power > 0 and signal_power > 0:
                snr_db = 10.0 * np.log10(signal_power / noise_power)
            else:
                snr_db = float("inf")

            diff_metrics_text = (
                f"Diff: samples_compared={length_common:,} | "
                f"changed_bits={changed:,} ({changed_pct:5.2f}%) | "
                f"MASD={mean_abs:.2f} | SNR~{snr_db:.2f} dB"
            )

            if plane_primary_full.size != plane_compare_full.size:
                diff_metrics_text += " | NOTE: length mismatch (trimmed)"

        else:
            self.ax_diff.set_title("Bit-plane XOR (load a Compare WAV to view differences)")
            self.ax_diff.set_xticks([])

        self.diff_metrics_lbl.setText(diff_metrics_text)

        # ---------------- Final Draw ----------------
        self.canvas.draw_idle()