from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

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
    Returns a normalized dict and also passes through codec-specific fields
    so the UI can display previews/metrics. Canonical keys:
      - out: Path to output file
      - bytes_embedded: int
      - metric_label: short label for UI metric
      - metric_value: printable metric (e.g., "12.3%", "18342 frames")
    Any additional keys from the codec (e.g., steg/orig/mask/out_path/changed_pct)
    are preserved.
    """
    codec = pick_codec_for(carrier)
    if not codec:
        raise ValueError(f"No codec accepts: {carrier}")

    # suggest output path
    stem = (out_stem if out_stem else Path(carrier).stem) + "__steg"
    out_path = Path(carrier).with_name(stem)

    # let the codec do real work
    result = codec.embed(carrier, payload, out_path, bpc, key)

    # Determine final output file path (codec may return either "out" or "out_path")
    out_file = Path(result.get("out", result.get("out_path", out_path)))
    bytes_embedded = len(payload)

    # Base normalized fields
    ret = {
        "out": out_file,
        "bytes_embedded": bytes_embedded,
        "metric_label": "Payload size",
        "metric_value": f"{bytes_embedded} bytes",
    }

    # Preserve codec-specific fields so UI can use them (image previews, masks, etc.)
    ret.update(result)
    return ret

def extract(stego: Path, bpc: int, key: str) -> bytes:
    codec = pick_codec_for(stego)
    if not codec:
        raise ValueError(f"No codec accepts: {stego}")
    return codec.extract(stego, bpc, key)
