"""
Shared data for the Lua analyzer.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path


def detect_file_encoding(file_path: Path) -> str:
    """Detect file encoding: UTF-8 BOM -> UTF-8 -> latin-1 fallback."""
    raw = file_path.read_bytes()
    # UTF-8 BOM
    if raw[:3] == b'\xef\xbb\xbf':
        return 'utf-8-sig'
    # try UTF-8
    try:
        raw.decode('utf-8')
        return 'utf-8'
    except UnicodeDecodeError:
        pass
    # fallback to latin-1 (maps bytes 0-255 directly, always succeeds)
    return 'latin-1'


@dataclass
class Finding:
    """Represents a single issue found during analysis."""
    pattern_name: str
    severity: str  # GREEN, YELLOW, RED, DEBUG
    line_num: int
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    source_line: str = ""

    # aliases for compatibility
    @property
    def description(self) -> str:
        return self.message

    @property
    def line_content(self) -> str:
        return self.source_line
