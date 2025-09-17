"""
gui.py - Tkinter GUI for MP3 steganography

Features:
- Drag and drop / file browse for cover MP3 and payload
- Select number of LSBs (1-8)
- Encode to stego MP3
- Decode from stego MP3
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from stego_core import encode_mp3, decode_mp3

class StegoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MP3 Steganography Tool")

        # --- Cover MP3 selection ---
        self.cover_label = tk.Label(root, text="Cover MP3: (drag & drop or browse)")
        self.cover_label.pack()
        self.cover_entry = tk.Entry(root, width=50)
        self.cover_entry.pack()
        tk.Button(root, text="Browse", command=self.browse_cover).pack()

        # --- Payload selection ---
        self.payload_label = tk.Label(root, text="Payload file (binary/text):")
        self.payload_label.pack()
        self.payload_entry = tk.Entry(root, width=50)
        self.payload_entry.pack()
        tk.Button(root, text="Browse", command=self.browse_payload).pack()

        # --- LSB selection ---
        self.lsb_label = tk.Label(root, text="Number of LSBs:")
        self.lsb_label.pack()
        self.lsb_slider = tk.Scale(root, from_=1, to=8, orient=tk.HORIZONTAL)
        self.lsb_slider.set(1)
        self.lsb_slider.pack()

        # --- Encode/Decode buttons ---
        tk.Button(root, text="Encode", command=self.encode).pack(pady=5)
        tk.Button(root, text="Decode", command=self.decode).pack(pady=5)

    # ========== Browse handlers ==========
    def browse_cover(self):
        path = filedialog.askopenfilename(filetypes=[("MP3 files", "*.mp3")])
        if path:
            self.cover_entry.delete(0, tk.END)
            self.cover_entry.insert(0, path)

    def browse_payload(self):
        path = filedialog.askopenfilename()
        if path:
            self.payload_entry.delete(0, tk.END)
            self.payload_entry.insert(0, path)

    # ========== Encode ==========
    def encode(self):
        cover_path = self.cover_entry.get()
        payload_path = self.payload_entry.get()
        k = self.lsb_slider.get()

        if not cover_path or not payload_path:
            messagebox.showerror("Error", "Select both cover MP3 and payload")
            return

        try:
            with open(cover_path, "rb") as f:
                cover = f.read()
            with open(payload_path, "rb") as f:
                payload = f.read()

            out_path = filedialog.asksaveasfilename(
                defaultextension=".mp3",
                filetypes=[("MP3 files", "*.mp3")],
                title="Save Stego MP3 As"
            )
            if not out_path:
                return

            stego = encode_mp3(cover, payload, k, fname=payload_path)
            with open(out_path, "wb") as f:
                f.write(stego)

            messagebox.showinfo("Success", f"Stego MP3 saved at:\n{out_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Encoding failed: {e}")

    # ========== Decode ==========
    def decode(self):
        stego_path = self.cover_entry.get()
        if not stego_path:
            messagebox.showerror("Error", "Select a stego MP3 file")
            return

        try:
            with open(stego_path, "rb") as f:
                stego = f.read()

            payload, fname = decode_mp3(stego)

            out_path = filedialog.asksaveasfilename(
                defaultextension=".bin",
                initialfile=fname or "payload.bin",
                filetypes=[("Binary files", "*.bin"), ("All files", "*.*")]
            )
            if not out_path:
                return

            with open(out_path, "wb") as f:
                f.write(payload)

            messagebox.showinfo("Success", f"Payload extracted and saved at:\n{out_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Decoding failed: {e}")

# ==============================
if __name__ == "__main__":
    root = tk.Tk()
    app = StegoApp(root)
    root.mainloop()
