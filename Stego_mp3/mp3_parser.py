"""
mp3_parser.py - Parse MP3 frames and extract ancillary byte positions.
"""

def find_frames(mp3_bytes: bytes) -> list[dict]:
    """
    Very simplified MP3 frame parser.
    Returns a list of frames with their ancillary byte offsets.
    """
    frames = []
    i = 0
    while i < len(mp3_bytes) - 4:
        # MP3 frame sync = 0xFFF (first 11 bits all 1)
        if mp3_bytes[i] == 0xFF and (mp3_bytes[i+1] & 0xE0) == 0xE0:
            # Simplified: assume fixed frame length (CBR, no VBR)
            # Typical MP3 frame length ~ 1040 bytes at 128 kbps, 44.1kHz
            frame_len = 1040
            frame_end = i + frame_len
            if frame_end > len(mp3_bytes):
                break

            # For now, we assume last ~10 bytes are ancillary
            anc_start = frame_end - 10
            anc_positions = list(range(anc_start, frame_end))

            frames.append({"start": i, "end": frame_end, "ancillary": anc_positions})
            i = frame_end
        else:
            i += 1
    return frames
