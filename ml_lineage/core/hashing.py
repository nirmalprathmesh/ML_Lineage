"""
ml_lineage.core.hashing
=======================
Cryptographic hashing utilities for content-addressable artifact verification.
"""

import hashlib
import os
from pathlib import Path


def compute_file_md5(filepath: str | Path, chunk_size: int = 1048576) -> str:
    """Calculate the MD5 hexadecimal digest of a local file in chunks.

    Args:
        filepath: Path to the target file.
        chunk_size: Byte size per chunk (default: 1 MB / 1048576 bytes).

    Returns:
        32-character hexadecimal MD5 hash string.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    hasher = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
