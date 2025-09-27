# stego_manager.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from image_codec import ImageCodec
from wav_codec import WavCodec
from mp4_codec import Mp4Codec

# Register codecs here once
CODEC_LIST = [ImageCodec(), WavCodec(), Mp4Codec()]
CODECS_BY_ID = {c.codec_id: c for c in CODEC_LIST}

def pick_codec_for(path: Path):
    """Return the first codec that accepts the path, else None."""
    for c in CODEC_LIST:
        try:
            if c.accepts(path):
                return c
        except Exception:
            pass
    return None

def capacity(carrier: Path, bpc: int) -> int:
    """Unified capacity lookup; raises if no codec accepts the file."""
    codec = pick_codec_for(carrier)
    if not codec:
        raise ValueError(f"No codec accepts: {carrier}")
    return int(codec.capacity_bytes(carrier, bpc))

def embed(
    carrier: Path,
    payload: bytes,
    out_stem: Optional[str],
    bpc: int,
    key: str
) -> Dict:
    """
    Returns a normalized dict:
      {
        "out": Path,                     # output file path
        "bytes_embedded": int,           # number of payload bytes that were embedded
        "metric_label": str,             # short label for UI metric
        "metric_value": str              # printable metric (e.g., '12.3%', '18342 frames', etc.)
      }
    """



    codec = pick_codec_for(carrier)
    if not codec:
        raise ValueError(f"No codec accepts: {carrier}")

    # suggest output path
    stem = (out_stem if out_stem else Path(carrier).stem) + "__steg"
    out_path = Path(carrier).with_name(stem)

    # let the codec do real work
    result = codec.embed(carrier, payload, out_path, bpc, key)

    # normalize
    out_file = Path(result.get("out", out_path))
    bytes_embedded = int(result.get("bytes_embedded", 0))

    # Prefer codec-provided metric, or map legacy ones
    if "metric_label" in result and "metric_value" in result:
        metric_label = str(result["metric_label"])
        metric_value = str(result["metric_value"])
    else:
        # Back-compat: ImageCodec returns steg/orig/mask; WavCodec returns changed_pct; Mp4 used to return appended
        if "changed_pct" in result:  # WAV
            metric_label = "Samples changed"
            metric_value = f"{float(result['changed_pct']):.2f}%"
        elif "appended" in result:   # older MP4
            metric_label = "Payload appended"
            metric_value = f"{int(result['appended'])} bytes"
            bytes_embedded = max(bytes_embedded, int(result["appended"]))
        else:
            metric_label = "Embed status"
            metric_value = "OK"

    return {
        "out": out_path,
        "bytes_embedded": len(payload),
        "metric_label": "Payload size",
        "metric_value": f"{len(payload)} bytes"
    }


def extract(stego: Path, bpc: int, key: str) -> bytes:
    codec = pick_codec_for(stego)
    if not codec:
        raise ValueError(f"No codec accepts: {stego}")
    return codec.extract(stego, bpc, key)
