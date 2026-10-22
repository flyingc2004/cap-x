"""Materialize the published Franka Mini depth/marker batch without changing its Git checkout.

The source checkout remains at its current commit. Large LFS objects and the
materialized copy both live on the data disk. Videos are omitted for calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


PREFIX = "play_piano/mini-depth-marker/episodes/"
POINTER = re.compile(rb"\Aversion https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n?\Z")


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--revision", default="origin/master")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_repo.resolve()
    output = args.output_dir.resolve()
    revision = git(source, "rev-parse", args.revision).decode().strip()
    paths = git(source, "ls-tree", "-r", "--name-only", revision, "--", PREFIX).decode().splitlines()
    paths = [p for p in paths if p.startswith(PREFIX) and "/video/" not in p]
    seeds = sorted({int(p.split("/")[3]) for p in paths})
    if len(seeds) != 30:
        raise RuntimeError(f"expected 30 Mini depth/marker episodes, found {seeds}")
    output.mkdir(parents=True, exist_ok=True)
    existing = downloaded = pending = 0
    missing = []
    for path in paths:
        relative = Path(path).relative_to(PREFIX)
        destination = output / "episodes" / relative
        if not destination.is_relative_to(output):
            raise RuntimeError(f"unsafe remote path: {path}")
        blob = git(source, "show", f"{revision}:{path}")
        pointer = POINTER.fullmatch(blob)
        if pointer:
            sha, size = pointer.group(1).decode(), int(pointer.group(2))
            origin = source / ".git" / "lfs" / "objects" / sha[:2] / sha[2:4] / sha
            if not origin.is_file() or origin.stat().st_size != size:
                missing.append(path)
                pending += 1
                continue
            expected = sha
        else:
            origin = None
            expected = hashlib.sha256(blob).hexdigest()
            size = len(blob)
        if destination.exists():
            if destination.stat().st_size != size or digest(destination) != expected:
                raise RuntimeError(f"existing file differs; preserved: {destination}")
            existing += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".incomplete")
        if temporary.exists():
            temporary.unlink()
        if origin is None:
            temporary.write_bytes(blob)
        else:
            shutil.copyfile(origin, temporary)
        if temporary.stat().st_size != size or digest(temporary) != expected:
            raise RuntimeError(f"checksum mismatch: {temporary}")
        temporary.rename(destination)
        downloaded += 1
    report = {
        "source_url": git(source, "remote", "get-url", "origin").decode().strip(),
        "source_commit": revision,
        "seeds": seeds,
        "video_omitted": True,
        "verified_existing_files": existing,
        "materialized_files": downloaded,
        "pending_lfs_paths": missing,
    }
    (output / "materialization.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "pending_lfs_paths"} | {"pending_count": pending}))
    if pending:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
