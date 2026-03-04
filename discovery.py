"""
This finds mod folders and their Lua scripts.

Handles the standard MO2 structure:
  <path>/gamedata/scripts/*.lua
  <path>/gamedata/scripts/*.script
"""

from pathlib import Path
from typing import Dict, List


def _get_scripts_if_exists(path: Path) -> List[Path]:
    """Helper to return scripts if directory exists."""
    if path.exists():
        return find_scripts(path)
    return []

def discover_mods(root_path: Path) -> Dict[str, List[Path]]:
    """
    Discover all mods and their script files.
    Returns dict mapping mod name -> list of script file paths.
    """
    mods: Dict[str, List[Path]] = {}
    root_path = Path(root_path)

    # check if this IS a gamedata folder directly
    if root_path.name == "gamedata":
        scripts = _get_scripts_if_exists(root_path / "scripts")
        if scripts:
            mods["(root)"] = scripts
        return mods

    # check if this folder contains gamedata directly (single mod)
    scripts = _get_scripts_if_exists(root_path / "gamedata" / "scripts")
    if scripts:
        mods[root_path.name] = scripts

    # scan subdirectories for mod folders
    for item in root_path.iterdir():
        if not item.is_dir():
            continue

        # skip common non-mod folders
        if item.name.startswith('.') or item.name in ('__pycache__', 'backup', 'backups'):
            continue

        scripts = _get_scripts_if_exists(item / "gamedata" / "scripts")
        if scripts:
            mods[item.name] = scripts
        else:
            # maybe nested structure, check one level deeper
            for subitem in item.iterdir():
                if subitem.is_dir():
                    scripts = _get_scripts_if_exists(subitem / "gamedata" / "scripts")
                    if scripts:
                        mods[f"{item.name}/{subitem.name}"] = scripts

    return mods


def find_scripts(scripts_dir: Path) -> List[Path]:
    """Find all Lua script files in a directory."""
    scripts = set()

    for ext in ("**/*.lua", "**/*.script"):
        scripts.update(scripts_dir.glob(ext))

    return sorted(scripts)


def get_mod_info(mod_path: Path) -> Dict[str, str]:
    """
    Try to extract mod info from common metadata files.
    Returns dict with name, version, author if found.
    """
    info = {"name": mod_path.name}

    # check for meta.ini (MO2 style)
    meta_ini = mod_path / "meta.ini"
    if meta_ini.exists():
        try:
            content = meta_ini.read_text(encoding='utf-8', errors='ignore')
            for line in content.splitlines():
                if line.startswith("name="):
                    info["name"] = line.split("=", 1)[1].strip()
                elif line.startswith("version="):
                    info["version"] = line.split("=", 1)[1].strip()
                elif line.startswith("author="):
                    info["author"] = line.split("=", 1)[1].strip()
        except (OSError, IOError):
            pass

    # check for modinfo.txt
    modinfo = mod_path / "modinfo.txt"
    if modinfo.exists():
        try:
            content = modinfo.read_text(encoding='utf-8', errors='ignore')
            lines = content.splitlines()
            if lines:
                info["name"] = lines[0].strip()
        except (OSError, IOError):
            pass

    return info


def discover_direct(path: Path) -> Dict[str, List[Path]]:
    """
    Discover scripts directly without gamedata/scripts structure.
    
    - If path is a .script/.lua file, return just that file
    - If path is a directory, find all scripts in it (recursively)
    
    Returns dict mapping mod name -> list of script file paths
    """
    path = Path(path)
    mods = {}
    
    # single file
    if path.is_file():
        if path.suffix in ('.script', '.lua'):
            mods["(direct)"] = [path]
        return mods
    
    # directory - find all scripts using set
    if path.is_dir():
        scripts = set()
        for ext in ("*.lua", "*.script"):
            scripts.update(path.glob(ext))
        for ext in ("**/*.lua", "**/*.script"):
            scripts.update(path.glob(ext))
        
        if scripts:
            mods["(direct)"] = sorted(scripts)
    
    return mods
