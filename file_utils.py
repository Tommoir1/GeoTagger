"""File-system helpers shared by GeoTagger workflows."""

from __future__ import annotations

import filecmp
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


GENERATED_OUTPUT_DIRECTORY_NAMES = frozenset(
    {
        "transects_json",
        "transects_output",
        "transects_output_batch",
        "extracted_images",
        "geotagged",
    }
)
GENERATED_OUTPUT_DIRECTORY_PREFIXES = (
    "extracted_images_",
    "geotagged_",
)


@dataclass(frozen=True)
class CopyResult:
    """Result of a collision-safe file copy."""

    destination_path: str
    copied: bool
    renamed_for_collision: bool


def is_generated_output_directory(directory_name: str) -> bool:
    """Return whether a directory name is one GeoTagger creates."""
    normalized_name = str(directory_name).strip().casefold()
    return normalized_name in GENERATED_OUTPUT_DIRECTORY_NAMES or normalized_name.startswith(
        GENERATED_OUTPUT_DIRECTORY_PREFIXES
    )


def _files_identical(first_path: str, second_path: str) -> bool:
    try:
        return filecmp.cmp(first_path, second_path, shallow=False)
    except OSError:
        return False


def _source_identity(source_path: str, source_root: Optional[str]) -> str:
    absolute_source = os.path.abspath(source_path)
    if source_root:
        absolute_root = os.path.abspath(source_root)
        try:
            if os.path.commonpath([absolute_source, absolute_root]) == absolute_root:
                return os.path.relpath(absolute_source, absolute_root)
        except (OSError, ValueError):
            pass
    return absolute_source


def copy_file_safely(
    source_path: str,
    destination_directory: str,
    *,
    source_root: Optional[str] = None,
) -> CopyResult:
    """
    Copy a file without silently overwriting a different file.

    The original basename is retained when available. If another file already
    occupies that name, a deterministic suffix derived from the source path is
    added. Re-running an extraction skips an already-identical copy.
    """
    source = os.path.abspath(os.fspath(source_path))
    destination_dir = os.path.abspath(os.fspath(destination_directory))

    if not os.path.isfile(source):
        raise FileNotFoundError(f"Source file does not exist: {source}")

    os.makedirs(destination_dir, exist_ok=True)
    direct_destination = os.path.join(destination_dir, os.path.basename(source))

    if not os.path.exists(direct_destination):
        shutil.copy2(source, direct_destination)
        return CopyResult(direct_destination, copied=True, renamed_for_collision=False)

    try:
        if os.path.samefile(source, direct_destination):
            return CopyResult(direct_destination, copied=False, renamed_for_collision=False)
    except OSError:
        pass

    if _files_identical(source, direct_destination):
        return CopyResult(direct_destination, copied=False, renamed_for_collision=False)

    source_identity = _source_identity(source, source_root).replace("\\", "/")
    identity_hash = hashlib.sha256(source_identity.encode("utf-8")).hexdigest()[:10]
    source_name = Path(source).name
    stem = Path(source_name).stem
    suffix = Path(source_name).suffix
    collision_destination = os.path.join(destination_dir, f"{stem}__{identity_hash}{suffix}")

    version = 2
    while os.path.exists(collision_destination):
        if _files_identical(source, collision_destination):
            return CopyResult(collision_destination, copied=False, renamed_for_collision=True)
        collision_destination = os.path.join(
            destination_dir,
            f"{stem}__{identity_hash}_{version}{suffix}",
        )
        version += 1

    shutil.copy2(source, collision_destination)
    return CopyResult(collision_destination, copied=True, renamed_for_collision=True)
