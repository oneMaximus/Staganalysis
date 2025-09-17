# TIFF LSB Steganography (Python, Tkinter)

This project implements **image LSB steganography** for **TIFF** images (and other common formats via conversion), with:

- Selectable **1–8 LSBs**
- **File explorer buttons** (drag & drop optional with `tkinterdnd2`)
- **Start location selection** (via mouse click)
- **Integer key** controls a permutation of cover bytes (must match for encode & decode)
- **Capacity check** shown before embedding
- **Difference map** visualization of LSB changes
- **Header** includes metadata (magic, version, k, ext, payload size, SHA-256 checksum)

---

📝 How to Use

Click Select Cover → choose sample_cover.tif.
Capacity info appears at the top.

(Optional) Click on the Cover preview to set a start location.

Click Select Payload → choose sample_payload.txt (or any file).

Set LSB bits (1–8) and enter an integer Key (e.g., 12345).

Click Encode ▶ → save as stego.tif.

The Stego preview and Difference Map will appear.

To recover: click Select Stego, re-enter the same k and Key, set the same start location, then Decode ◀ and save the extracted file.

1. Encode:
    Cover: sample_cover.tif
    Payload: sample_payload.txt
    LSBs: 3
    Key: 12345
    Save as: stego.tif

2. Decode:
    Stego: stego.tif
    LSBs: 3
    Key: 12345
    Save extracted payload as recovered_payload.txt