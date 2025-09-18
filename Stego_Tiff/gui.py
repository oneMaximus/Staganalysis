# gui.py — full GUI with image previews + diff map + start-pixel click

import os
import traceback
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Drag & drop is optional; if not installed, buttons still work
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    BaseTk = TkinterDnD.Tk
    DND_AVAILABLE = True
except Exception:
    BaseTk = tk.Tk
    DND_AVAILABLE = False

from PIL import Image, ImageTk

from stegano import SteganographyEngine
from utilities import load_tiff_bytes, mask_to_image

APP_TITLE = "TIFF LSB Steganography"
CANVAS_W, CANVAS_H = 420, 420


class App(BaseTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1100x760")

        self.engine = SteganographyEngine()

        self.cover_path: Optional[str] = None
        self.payload_path: Optional[str] = None
        self.stego_path: Optional[str] = None

        self.k_bits: int = 1
        self.key_value: int = 0
        self.start_pixel: int = 0

        # preview state (PhotoImage must be kept on self)
        self._cover_preview_img = None
        self._stego_preview_img = None
        self._diff_preview_img = None

        # mapping from canvas click -> image pixel
        self._cover_canvas_origin = (0, 0)   # where image is drawn on canvas
        self._cover_scale = 1.0              # image->canvas scale
        self._cover_size = None              # (w, h)
        self._cover_channels = None

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)

        ttk.Label(top, text="LSB bits (1..8):").pack(side="left")
        self.k_var = tk.IntVar(value=1)
        sp = ttk.Spinbox(top, from_=1, to=8, width=5, textvariable=self.k_var, command=self._on_params_changed)
        sp.pack(side="left", padx=6)

        ttk.Label(top, text="Key (integer):").pack(side="left", padx=(14, 0))
        self.key_var = tk.StringVar(value="0")
        key_entry = ttk.Entry(top, width=14, textvariable=self.key_var)
        key_entry.pack(side="left", padx=6)
        key_entry.bind("<KeyRelease>", lambda e: self._on_params_changed())

        self.start_label = ttk.Label(top, text="Start pixel: 0")
        self.start_label.pack(side="left", padx=(16, 0))

        ttk.Button(top, text="Select Cover", command=self.select_cover).pack(side="left", padx=6)
        ttk.Button(top, text="Select Payload", command=self.select_payload).pack(side="left", padx=6)
        ttk.Button(top, text="Encode ▶", command=self.encode_action).pack(side="left", padx=12)
        ttk.Button(top, text="Select Stego", command=self.select_stego).pack(side="left", padx=6)
        ttk.Button(top, text="Decode ◀", command=self.decode_action).pack(side="left", padx=6)

        self.capacity_var = tk.StringVar(value="Capacity: -")
        ttk.Label(top, textvariable=self.capacity_var).pack(side="right")

        # Middle: three canvases
        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, padx=10, pady=6)

        left = ttk.Frame(mid); left.pack(side="left", fill="both", expand=True)
        center = ttk.Frame(mid); center.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(mid); right.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="Cover").pack()
        self.canvas_cover = tk.Canvas(left, width=CANVAS_W, height=CANVAS_H,
                                      bg="#202020", highlightthickness=1, highlightbackground="#444")
        self.canvas_cover.pack(pady=6)
        self.canvas_cover.bind("<Button-1>", self.on_cover_click)
        if DND_AVAILABLE:
            self.canvas_cover.drop_target_register(DND_FILES)
            self.canvas_cover.dnd_bind("<<Drop>>", self._on_drop_cover)

        ttk.Label(center, text="Stego").pack()
        self.canvas_stego = tk.Canvas(center, width=CANVAS_W, height=CANVAS_H,
                                      bg="#202020", highlightthickness=1, highlightbackground="#444")
        self.canvas_stego.pack(pady=6)
        if DND_AVAILABLE:
            self.canvas_stego.drop_target_register(DND_FILES)
            self.canvas_stego.dnd_bind("<<Drop>>", self._on_drop_stego)

        ttk.Label(right, text="Difference Map (LSB changes)").pack()
        self.canvas_diff = tk.Canvas(right, width=CANVAS_W, height=CANVAS_H,
                                     bg="#202020", highlightthickness=1, highlightbackground="#444")
        self.canvas_diff.pack(pady=6)

        # Bottom: log
        bottom = ttk.Frame(self)
        bottom.pack(fill="both", expand=False, padx=10, pady=(0, 10))
        ttk.Label(bottom, text="Log:").pack(anchor="w")
        self.log = tk.Text(bottom, height=10)
        self.log.pack(fill="x")

    # --------------- Drag & drop ---------------
    def _on_drop_cover(self, event):
        path = event.data.strip("{}")
        if os.path.isfile(path):
            self.cover_path = path
            self._load_cover_preview()
            self._update_capacity()

    def _on_drop_stego(self, event):
        path = event.data.strip("{}")
        if os.path.isfile(path):
            self.stego_path = path
            self._load_stego_preview()

    # --------------- File pickers ---------------
    def select_cover(self):
        p = filedialog.askopenfilename(
            title="Select Cover (TIFF or image)",
            filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp")]
        )
        if p:
            self.cover_path = p
            self._load_cover_preview()
            self._update_capacity()

    def select_payload(self):
        p = filedialog.askopenfilename(title="Select Payload (any file)")
        if p:
            self.payload_path = p
            self._log(f"Payload selected: {p}")
            self._update_capacity()

    def select_stego(self):
        p = filedialog.askopenfilename(
            title="Select Stego (TIFF or image)",
            filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp")]
        )
        if p:
            self.stego_path = p
            self._load_stego_preview()

    # --------------- Preview helpers ---------------
    def _fit_to_canvas(self, img: Image.Image):
        iw, ih = img.size
        scale = min(CANVAS_W / iw, CANVAS_H / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        pad_x = (CANVAS_W - nw) // 2
        pad_y = (CANVAS_H - nh) // 2
        disp = img.resize((nw, nh), Image.NEAREST)
        return disp, scale, (pad_x, pad_y)

    def _load_cover_preview(self):
        if not self.cover_path:
            return
        try:
            img = Image.open(self.cover_path)
            self._cover_size = img.size
            disp, scale, origin = self._fit_to_canvas(img)
            self._cover_scale = scale
            self._cover_canvas_origin = origin
            self._cover_preview_img = ImageTk.PhotoImage(disp)
            self.canvas_cover.delete("all")
            self.canvas_cover.create_image(origin[0], origin[1],
                                           image=self._cover_preview_img, anchor="nw")
            # record channels for start-pixel math
            _, _, ch, _ = load_tiff_bytes(self.cover_path)
            self._cover_channels = ch
            self._log(f"Cover loaded: {self.cover_path} ({img.size[0]}x{img.size[1]})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load cover: {e}")

    def _load_stego_preview(self):
        if not self.stego_path:
            return
        try:
            img = Image.open(self.stego_path)
            disp, _, origin = self._fit_to_canvas(img)
            self._stego_preview_img = ImageTk.PhotoImage(disp)
            self.canvas_stego.delete("all")
            self.canvas_stego.create_image(origin[0], origin[1],
                                           image=self._stego_preview_img, anchor="nw")
            self._log(f"Stego loaded: {self.stego_path} ({img.size[0]}x{img.size[1]})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load stego: {e}")

    def _show_diff_image(self, pil_img: Image.Image):
        disp, _, origin = self._fit_to_canvas(pil_img)
        self._diff_preview_img = ImageTk.PhotoImage(disp)
        self.canvas_diff.delete("all")
        self.canvas_diff.create_image(origin[0], origin[1],
                                      image=self._diff_preview_img, anchor="nw")

    # --------------- Param updates ---------------
    def _on_params_changed(self):
        try:
            self.k_bits = int(self.k_var.get())
        except Exception:
            self.k_bits = 1
        try:
            self.key_value = int(self.key_var.get() or "0")
        except Exception:
            self.key_value = 0
        self._update_capacity()

    def _update_capacity(self):
        if not self.cover_path:
            self.capacity_var.set("Capacity: -")
            return
        try:
            report = self.engine.get_capacity_report(
                self.cover_path, max(1, min(8, int(self.k_var.get()))), self.start_pixel
            )
            self.capacity_var.set(
                f"Capacity: up to {report['max_payload_bytes']} bytes (k={self.k_var.get()}, start={self.start_pixel})"
            )
        except Exception as e:
            self.capacity_var.set(f"Capacity: error: {e}")

    # --------------- Mouse: set start pixel ---------------
    def on_cover_click(self, event):
        if not self.cover_path or self._cover_size is None:
            return
        # Map canvas coords -> image pixel coords
        x_img = int((event.x - self._cover_canvas_origin[0]) / self._cover_scale)
        y_img = int((event.y - self._cover_canvas_origin[1]) / self._cover_scale)
        w, h = self._cover_size
        if 0 <= x_img < w and 0 <= y_img < h:
            self.start_pixel = y_img * w + x_img
            self.start_label.config(text=f"Start pixel: {self.start_pixel} (x={x_img}, y={y_img})")
            # Draw a little crosshair
            self._load_cover_preview()
            cx = int(self._cover_canvas_origin[0] + x_img * self._cover_scale)
            cy = int(self._cover_canvas_origin[1] + y_img * self._cover_scale)
            self.canvas_cover.create_line(cx - 6, cy, cx + 6, cy, fill="yellow")
            self.canvas_cover.create_line(cx, cy - 6, cx, cy + 6, fill="yellow")
            self._update_capacity()

    # --------------- Actions ---------------
    def encode_action(self):
        if not self.cover_path:
            messagebox.showwarning("Missing Cover", "Please select a cover image first.")
            return
        if not self.payload_path:
            messagebox.showwarning("Missing Payload", "Please select a payload file.")
            return
        out = filedialog.asksaveasfilename(
            title="Save stego as",
            defaultextension=".tif",
            filetypes=[("TIFF Image", "*.tif *.tiff")]
        )
        if not out:
            return
        try:
            report = self.engine.embed(
                cover_path=self.cover_path,
                payload_path=self.payload_path,
                out_path=out,
                k_bits=int(self.k_var.get()),
                key=int(self.key_var.get() or "0"),
                start_pixel=self.start_pixel
            )
            self._log(f"Encoded OK. Used {report['cover_bytes_used']} cover bytes.")
            self.stego_path = out
            self._load_stego_preview()

            # Show difference map
            mask_flat = report.get("diff_mask_flat")
            if mask_flat is not None:
                w, h = report["size"]
                diff_img = mask_to_image(mask_flat, (w, h))
                self._show_diff_image(diff_img)
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Encode Error", str(e))

    def decode_action(self):
        if not self.stego_path:
            messagebox.showwarning("Missing Stego", "Please select a stego image first.")
            return
        try:
            res = self.engine.decode(
                stego_path=self.stego_path,
                k_bits=int(self.k_var.get()),
                key=int(self.key_var.get() or "0"),
                start_pixel=self.start_pixel
            )
            meta = res["meta"]
            self._log(f"Decoded OK. Payload length={meta.length} bytes, ext=.{meta.ext}, k={meta.k_bits}")
            out = filedialog.asksaveasfilename(
                title="Save extracted payload as",
                defaultextension="." + (meta.ext or "bin")
            )
            if out:
                with open(out, "wb") as f:
                    f.write(res["payload_bytes"])
                self._log(f"Extracted payload saved: {out}")
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Decode Error", str(e))

    # --------------- Log ---------------
    def _log(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")


if __name__ == "__main__":
    app = App()
    app.mainloop()
