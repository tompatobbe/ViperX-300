"""
pipeline_artifacts.py
=====================
Reproducible artifact management for ViperX-300 SysID pipeline.

Every saved artifact gets a deterministic name derived from:
  (input file stem, pipeline name, pipeline version, config hash)

and an adjacent .json sidecar with full provenance.  The same run twice
produces the same path, so callers can detect a cache hit before computing.

Directory layout
----------------
outputs/
  {input_stem}/
    {input_stem}__{pipeline}-v{version}__cfg-{hash8}.npy    <- numpy arrays
    {input_stem}__{pipeline}-v{version}__cfg-{hash8}.urdf   <- text artifacts
    {input_stem}__{pipeline}-v{version}__cfg-{hash8}.json   <- metadata sidecar
  legacy/          <- migrate_legacy() moves old npy/ files here
logs/
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Layout defaults (all paths are relative to CWD unless overridden)
# ---------------------------------------------------------------------------

DEFAULT_OUTPUTS_ROOT = Path("outputs")
DEFAULT_LEGACY_DIR   = Path("npy")
SEP = "__"          # field separator in artifact filenames


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def config_hash(config: dict[str, Any], length: int = 8) -> str:
    """
    Stable SHA-256 fingerprint of a config dict, hex-truncated to `length` chars.

    Keys are sorted before serialisation so insertion order is irrelevant.
    Callers must be consistent about types (e.g. always pass fs as float, never
    int) to guarantee stable hashes across calls.
    """
    canonical = json.dumps(config, sort_keys=True, default=_json_default)
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]


def artifact_path(
    csv_path:         str | Path,
    pipeline_name:    str,
    pipeline_version: str,
    config:           dict[str, Any],
    outputs_root:     str | Path | None = None,
    suffix:           str = ".npy",
) -> tuple[Path, Path]:
    """
    Return the deterministic ``(artifact_path, json_path)`` pair for an experiment.

    The paths depend only on (input stem, pipeline_name, pipeline_version,
    config).  Timestamps are *not* part of the path — they live in the sidecar.
    Two identical runs therefore map to the same file, enabling cache detection.

    Parameters
    ----------
    csv_path:
        Path to the input file (only the stem is used in the filename).
        Despite the name, any input file type works — .csv, .npy, etc.
    pipeline_name:
        Short identifier for the script, e.g. ``"sysid_feasible"``.
    pipeline_version:
        Semantic version string, e.g. ``"1.1"``.  Bump this whenever the
        algorithm changes in a way that makes old and new outputs incomparable.
    config:
        Dict of all parameters that affect the output (sample rate, filter
        cutoffs, optimiser settings, template path, …).
    outputs_root:
        Root directory for outputs.  Defaults to ``outputs/``.
    suffix:
        File extension for the artifact, e.g. ``".npy"`` or ``".urdf"``.
    """
    csv_path = Path(csv_path)
    root     = Path(outputs_root) if outputs_root else DEFAULT_OUTPUTS_ROOT

    stem         = csv_path.stem
    version_slug = pipeline_version.replace(".", "-")   # "1.1" → "1-1"
    cfg_hash     = config_hash(config)
    type_dir     = suffix.lstrip(".")                   # ".npy" → "npy", ".urdf" → "urdf"

    basename = SEP.join([stem, f"{pipeline_name}-v{version_slug}", f"cfg-{cfg_hash}"])
    subdir   = root / type_dir

    return subdir / f"{basename}{suffix}", subdir / f"{basename}.json"


def save_artifact(
    phi:              np.ndarray,
    csv_path:         str | Path,
    pipeline_name:    str,
    pipeline_version: str,
    config:           dict[str, Any],
    outputs_root:     str | Path | None = None,
    allow_overwrite:  bool = False,
) -> tuple[Path, Path]:
    """
    Atomically write *phi* and a provenance sidecar to the deterministic path.

    Uses a write-to-temp-then-rename strategy so a crash or KeyboardInterrupt
    never leaves a partial file at the destination.

    Raises
    ------
    FileExistsError
        If the artifact already exists and ``allow_overwrite`` is False.

    Returns
    -------
    (npy_path, json_path)
    """
    npy_path, json_path = artifact_path(
        csv_path, pipeline_name, pipeline_version, config, outputs_root
    )

    if npy_path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"\nArtifact already exists:\n  {npy_path}\n"
            "Pass --force to recompute, or load the existing result with:\n"
            f"  np.load('{npy_path}')\n"
            f"  # metadata: {json_path}"
        )

    meta = _build_metadata(phi, csv_path, npy_path, pipeline_name, pipeline_version, config)

    npy_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_npy(phi, npy_path)
    _atomic_json(meta, json_path)

    return npy_path, json_path


def load_artifact(
    csv_path:         str | Path,
    pipeline_name:    str,
    pipeline_version: str,
    config:           dict[str, Any],
    outputs_root:     str | Path | None = None,
) -> np.ndarray | None:
    """
    Return the cached array if it exists, else ``None``.

    Typical usage::

        phi = load_artifact(csv, PIPELINE_NAME, PIPELINE_VERSION, cfg)
        if phi is not None:
            print("Cache hit — skipping computation.")
        else:
            phi = run_identification(csv, ...)
            save_artifact(phi, csv, PIPELINE_NAME, PIPELINE_VERSION, cfg)
    """
    npy_path, _ = artifact_path(
        csv_path, pipeline_name, pipeline_version, config, outputs_root
    )
    return np.load(npy_path) if npy_path.exists() else None


def save_text_artifact(
    content:          str,
    input_path:       str | Path,
    pipeline_name:    str,
    pipeline_version: str,
    config:           dict[str, Any],
    suffix:           str = ".urdf",
    outputs_root:     str | Path | None = None,
    allow_overwrite:  bool = False,
) -> tuple[Path, Path]:
    """
    Atomically write a text artifact (URDF, SVG, CSV, …) and a provenance sidecar.

    Identical to :func:`save_artifact` but for string content.  The path is
    derived with the given *suffix* so ``.npy`` and ``.urdf`` outputs produced
    from the same input never collide even if every other parameter matches.

    Returns
    -------
    (artifact_path, json_path)
    """
    art_path, json_path = artifact_path(
        input_path, pipeline_name, pipeline_version, config, outputs_root,
        suffix=suffix,
    )

    if art_path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"\nArtifact already exists:\n  {art_path}\n"
            "Pass --force to recompute, or load the existing file directly.\n"
            f"  # metadata: {json_path}"
        )

    meta = _build_text_metadata(
        content, input_path, art_path, pipeline_name, pipeline_version, config
    )

    art_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(content, art_path)
    _atomic_json(meta, json_path)

    return art_path, json_path


def load_text_artifact(
    input_path:       str | Path,
    pipeline_name:    str,
    pipeline_version: str,
    config:           dict[str, Any],
    suffix:           str = ".urdf",
    outputs_root:     str | Path | None = None,
) -> str | None:
    """Return the cached text content if it exists, else ``None``."""
    art_path, _ = artifact_path(
        input_path, pipeline_name, pipeline_version, config, outputs_root,
        suffix=suffix,
    )
    return art_path.read_text(encoding="utf-8") if art_path.exists() else None


def list_artifacts(
    csv_path:     str | Path | None = None,
    outputs_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Return a list of metadata dicts from all .json sidecars in *outputs_root*,
    optionally filtered to a single input file stem.

    Filtering reads the ``input_file`` field from each sidecar rather than
    relying on directory names, so it works across all type subdirectories
    (``npy/``, ``urdf/``, ``legacy/``, …).
    """
    root = Path(outputs_root) if outputs_root else DEFAULT_OUTPUTS_ROOT
    if not root.exists():
        return []

    filter_stem = Path(csv_path).stem if csv_path is not None else None

    results: list[dict[str, Any]] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for jf in sorted(d.glob("*.json")):
            try:
                meta = json.loads(jf.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if filter_stem is None or Path(meta.get("input_file", "")).stem == filter_stem:
                results.append(meta)
    return results


def migrate_legacy(
    legacy_dir:   str | Path = DEFAULT_LEGACY_DIR,
    outputs_root: str | Path = DEFAULT_OUTPUTS_ROOT,
) -> int:
    """
    Copy files from the old ``npy/`` directory into ``outputs/legacy/`` and
    create stub metadata sidecars for them.

    Safe to call repeatedly — files that already have a destination are skipped.
    The originals in *legacy_dir* are left intact.

    Returns the number of newly migrated files.
    """
    legacy_dir   = Path(legacy_dir)
    outputs_root = Path(outputs_root)

    if not legacy_dir.exists():
        return 0

    dest_dir = outputs_root / "legacy"
    dest_dir.mkdir(parents=True, exist_ok=True)

    migrated = 0
    for npy_src in sorted(legacy_dir.glob("*.npy")):
        dest_npy  = dest_dir / npy_src.name
        dest_json = dest_npy.with_suffix(".json")

        if dest_npy.exists():
            continue    # already migrated in a prior call

        shutil.copy2(npy_src, dest_npy)

        meta: dict[str, Any] = {
            "input_file":       "unknown (legacy)",
            "output_file":      str(dest_npy),
            "pipeline_name":    "unknown",
            "pipeline_version": "legacy",
            "config":           {},
            "config_hash":      "legacy",
            "created_at":       _mtime_iso(npy_src),
            "git_commit":       None,
            "source_checksum":  None,
            "output_shape":     list(np.load(dest_npy, allow_pickle=False).shape),
            "migrated_from":    str(npy_src),
            "note":             "Migrated from legacy npy/ directory; metadata is incomplete.",
        }
        _atomic_json(meta, dest_json)
        migrated += 1

    return migrated


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _file_sha256(path: Path) -> str | None:
    """SHA-256 checksum of *path*, prefixed with ``sha256:``.  Returns None if
    the file does not exist."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _mtime_iso(path: Path) -> str:
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def _build_metadata(
    phi:              np.ndarray,
    csv_path:         Path | str,
    npy_path:         Path,
    pipeline_name:    str,
    pipeline_version: str,
    config:           dict[str, Any],
) -> dict[str, Any]:
    csv_path = Path(csv_path)
    return {
        "input_file":       str(csv_path),
        "output_file":      str(npy_path),
        "pipeline_name":    pipeline_name,
        "pipeline_version": pipeline_version,
        "config":           config,
        "config_hash":      config_hash(config),
        "created_at":       datetime.now(tz=timezone.utc).isoformat(),
        "git_commit":       _git_commit(),
        "source_checksum":  _file_sha256(csv_path),
        "output_shape":     list(phi.shape),
        "output_dtype":     str(phi.dtype),
        "hostname":         socket.gethostname(),
        "python_version":   platform.python_version(),
        "numpy_version":    np.__version__,
    }


def _atomic_npy(arr: np.ndarray, dest: Path) -> None:
    """Write *arr* to a temp file then rename atomically to *dest*."""
    buf = io.BytesIO()
    np.save(buf, arr)
    raw = buf.getvalue()

    fd, tmp_str = tempfile.mkstemp(dir=dest.parent)
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_json(data: dict[str, Any], dest: Path) -> None:
    """Write *data* as indented JSON to a temp file then rename atomically."""
    text = json.dumps(data, indent=2, default=_json_default)

    fd, tmp_str = tempfile.mkstemp(dir=dest.parent, suffix=".json")
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_text(content: str, dest: Path) -> None:
    """Write a UTF-8 string to a temp file then rename atomically to *dest*."""
    fd, tmp_str = tempfile.mkstemp(dir=dest.parent)
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _build_text_metadata(
    content:          str,
    input_path:       Path | str,
    art_path:         Path,
    pipeline_name:    str,
    pipeline_version: str,
    config:           dict[str, Any],
) -> dict[str, Any]:
    input_path = Path(input_path)
    return {
        "input_file":       str(input_path),
        "output_file":      str(art_path),
        "pipeline_name":    pipeline_name,
        "pipeline_version": pipeline_version,
        "config":           config,
        "config_hash":      config_hash(config),
        "created_at":       datetime.now(tz=timezone.utc).isoformat(),
        "git_commit":       _git_commit(),
        "source_checksum":  _file_sha256(input_path),
        "content_length":   len(content),
        "hostname":         socket.gethostname(),
        "python_version":   platform.python_version(),
        "numpy_version":    np.__version__,
    }
