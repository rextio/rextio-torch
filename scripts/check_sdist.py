#!/usr/bin/env python3
"""Audit repository fixtures shipped in a rextio-torch source distribution."""

from __future__ import annotations

import argparse
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SCOPED_DIRS = frozenset({"benchmarks", "docs", "scripts", "tests"})
SCOPED_FILES = frozenset({"CHANGELOG.md", "MANIFEST.in"})


def _repository_files() -> tuple[Path, ...]:
    """Return tracked and newly-added files that define the sdist test surface."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "benchmarks",
            "docs",
            "scripts/check_sdist.py",
            "scripts/linux-smoke.sh",
            "tests",
            "CHANGELOG.md",
            "MANIFEST.in",
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    paths = tuple(
        sorted(
            Path(raw.decode("utf-8"))
            for raw in result.stdout.split(b"\0")
            if raw
        )
    )
    missing = [str(path) for path in paths if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"repository fixture list contains missing files: {missing}")
    if not paths:
        raise RuntimeError("repository fixture list is empty")
    return paths


def _safe_member_path(name: str) -> PurePosixPath:
    """Return a normalized tar member path or reject unsafe traversal."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError(f"unsafe sdist member path: {name!r}")
    return path


def audit_sdist(archive: Path, extract_to: Path | None = None) -> Path | None:
    """Verify scoped members and optionally extract the audited archive."""
    expected = _repository_files()
    with tarfile.open(archive, mode="r:gz") as tar:
        members = tar.getmembers()
        safe_paths = {member.name: _safe_member_path(member.name) for member in members}
        top_levels = {path.parts[0] for path in safe_paths.values()}
        if len(top_levels) != 1:
            raise RuntimeError(f"sdist must have exactly one top-level directory: {top_levels}")
        top = next(iter(top_levels))
        by_name = {member.name: member for member in members}

        expected_names = {f"{top}/{path.as_posix()}" for path in expected}
        for relative in expected:
            name = f"{top}/{relative.as_posix()}"
            member = by_name.get(name)
            if member is None or not member.isfile():
                raise RuntimeError(f"required regular sdist member is missing: {relative}")
            stream = tar.extractfile(member)
            if stream is None or stream.read() != (ROOT / relative).read_bytes():
                raise RuntimeError(f"sdist member differs from repository bytes: {relative}")

        shipped_scoped: set[str] = set()
        for member in members:
            if not member.isfile():
                continue
            path = safe_paths[member.name]
            if len(path.parts) < 2 or path.parts[0] != top:
                continue
            relative = PurePosixPath(*path.parts[1:])
            if relative.as_posix() in SCOPED_FILES or relative.parts[0] in SCOPED_DIRS:
                shipped_scoped.add(member.name)
        unexpected = sorted(shipped_scoped - expected_names)
        if unexpected:
            raise RuntimeError(f"unexpected untracked scoped sdist members: {unexpected}")

        smoke = by_name[f"{top}/scripts/linux-smoke.sh"]
        if smoke.mode & 0o111 == 0:
            raise RuntimeError("scripts/linux-smoke.sh lost its executable mode in the sdist")

        extracted_root: Path | None = None
        if extract_to is not None:
            if extract_to.exists() and any(extract_to.iterdir()):
                raise RuntimeError(f"extraction directory is not empty: {extract_to}")
            extract_to.mkdir(parents=True, exist_ok=True)
            tar.extractall(extract_to, filter="data")
            extracted_root = extract_to / top

    print(f"sdist repository fixtures: OK ({len(expected)} byte-equal files)")
    if extracted_root is not None:
        print(f"extracted root: {extracted_root}")
    return extracted_root


def main(argv: list[str] | None = None) -> int:
    """Run the command-line artifact audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args(argv)
    audit_sdist(args.archive, args.extract_to)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
