# MP3 Steganography (LSB-Based)

This project implements **audio steganography** for **MP3 files** using **Least Significant Bit (LSB)** encoding.  
It allows you to **hide any binary payload** (text, images, executables, etc.) inside an MP3 file without affecting playback quality.  
You can choose **1–8 LSBs** to balance **capacity** vs **stealthiness**.

---

## Features
- Embed any binary file inside MP3 audio  
- Extract hidden payload safely with CRC32 integrity check  
- Choose **1–8 LSBs** for embedding (via GUI or CLI)  
- Simple GUI with browse buttons & slider  
- Example files provided in `samples/`

---

## 📂 File Structure
Stego_MP3/
├── init.py
├── utils.py # helper functions (bits, CRC)
├── mp3_parser.py # MP3 frame parsing + ancillary byte offsets
├── stego_core.py # encode/decode logic
├── encode.py # CLI tool: hide payload in MP3
├── decode.py # CLI tool: extract payload
├── gui.py # Tkinter GUI
├── samples/
│ ├── cover.mp3
│ ├── secret.bin
│ └── stego_output.mp3
└── README.md
