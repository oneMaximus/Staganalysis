# base_codec.py
from pathlib import Path
from typing import Optional

class BaseCodec:
    codec_id: str
    pretty: str

    def capacity_bytes(self, carrier: Path, bpc: int) -> int:
        """Return how many payload bytes can be stored in this carrier at given bpc."""
        raise NotImplementedError

    def embed(self, carrier: Path, payload: bytes, out_path: Path, bpc: int, key: str) -> dict:
        """Embed the payload in the carrier, save result at out_path, return metadata dict."""
        raise NotImplementedError

    def extract(self, stego: Path, bpc: int, key: str) -> bytes:
        """Extract the hidden payload from a stego file."""
        raise NotImplementedError

    def accepts(self, path: Path) -> bool:
        """Return True if this codec can handle the given file type (by suffix/headers)."""
        raise NotImplementedError
