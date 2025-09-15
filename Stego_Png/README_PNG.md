# PNG Steganography Tool

A Python application for hiding and extracting messages or files inside PNG images using **Least Significant Bit (LSB)** steganography.  
This project is built in **phases**, starting with text embedding and gradually expanding to more complex payloads and analysis.

> ⚠️ For educational purposes only. LSB steganography is fragile and not secure. If confidentiality is required, encrypt your data before embedding.

---

## 🚀 Phases

### Phase 1 — Embed Text Files into PNGs
- Hide small text messages or `.txt` files in a PNG.
- Extract payloads without the original image (blind mode).
- Optional XOR key obfuscation (lightweight, not encryption).

### Phase 2 — Embed Image Files into PNGs
- Support arbitrary binary/image files as payloads.
- Optional compression before embedding.

### Phase 3 — Embed Video Files into PNGs
- Split large payloads across multiple carrier PNGs.
- Use a manifest to reconstruct original files.

### Phase 4 — Steg Analysis
- Tools to detect likely stego images.
- Includes LSB histograms, chi-square tests, and risk scoring.

### Phase 5 — Decrypt PNG Files
- Replace XOR with strong encryption (AES-GCM).
- Add passphrase support for secure payload protection.

### Phase 6 — Web Front-End
- Develop a simple **web interface** on top of the Python backend.
- Provide **Encode** workflow:
  1. Upload a PNG cover image.
  2. Enter a message or upload a file.
  3. (Optional) Enter a passphrase for encryption (Phase 5).
  4. Click **Encode** → download stego PNG.
- Provide **Decode** workflow:
  1. Upload a stego PNG.
  2. (Optional) Enter passphrase if encryption was used.
  3. Click **Decode** → reveal message or download hidden file.
- Show **capacity information** (how much data can fit).
- Integrate **steg analysis results** (from Phase 4) to give a “risk score” on the image.
- Include clear disclaimers:
  - PNG only (lossless format).
  - Payload may be corrupted by resizing or recompression.
  - Without encryption, this is not secure.
- Tech stack ideas:
  - **Flask** or **FastAPI** (backend).
  - **HTML/CSS/JavaScript** for frontend UI.
  - Optional: React/Vue for richer UX.

---

## 📦 Installation

Requirements:
- Python 3.10+
- [Pillow](https://pypi.org/project/Pillow/)
- [click](https://pypi.org/project/click/)
- [numpy](https://pypi.org/project/numpy/)

Install dependencies:

```bash
pip install Pillow click numpy matplotlib
