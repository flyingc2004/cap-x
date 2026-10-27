#!/usr/bin/env python3
"""Independent offline DIGIT representation experiment. Defaults to CPU."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from capx.memory.trajectory.data import DEFAULT_SOURCES, prepare, read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["suite", "prepare", "train", "evaluate", "encode", "search"])
    p.add_argument("--run", type=Path, required=True, help="Fresh data-disk directory for prepare; existing run for other stages")
    p.add_argument("--device", default="cpu", help="Explicit cpu or cuda:N; never automatically selected")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--episode", help="Public cache episode id for encode")
    p.add_argument("--end-step", type=int)
    p.add_argument("--output", type=Path)
    p.add_argument("--index", type=Path)
    p.add_argument("--query", type=Path)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()
    suite = args.stage == "suite"
    def status(stage, state="running", **fields):
        write_json(args.run / "status.json", dict(stage=stage, state=state, pid=os.getpid(),
            updated_unix_seconds=time.time(), **fields))

    if args.stage in ("prepare", "suite"):
        args.run.mkdir(parents=True, exist_ok=False)
        if suite:
            stream=(args.run/"run.log").open("x",encoding="utf-8",buffering=1)
            sys.stdout=sys.stderr=stream
            def handle_error(exc_type,exc,tb):
                traceback.print_exception(exc_type,exc,tb,file=stream)
                previous=read_json(args.run/"status.json") if (args.run/"status.json").exists() else {}
                status(previous.get("stage","initialization"),"failed",error=f"{exc_type.__name__}: {exc}",exit_code=1)
            sys.excepthook=handle_error
        status("prepare")
        source_dir = args.run / "source"
        source_dir.mkdir()
        files = sorted((REPO / "capx/memory/trajectory").glob("*.py")) + [Path(__file__)]
        for name in ("audit_digit_trajectory_neighbors.py",):
            path = Path(__file__).with_name(name)
            if path.exists(): files.append(path)
        source_hashes = {}
        for path in files:
            relative = path.relative_to(REPO)
            target = source_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            source_hashes[str(relative)] = sha256(path)
        git = lambda *a: subprocess.check_output(["git", "-C", str(REPO), *a], text=True).strip()
        write_json(args.run / "provenance.json", dict(repo=str(REPO), commit=git("rev-parse","HEAD"),
            dirty=bool(git("status","--porcelain")), sources=source_hashes, inputs=[str(p) for p in DEFAULT_SOURCES],
            python=sys.executable, device_requested=args.device, threads=args.threads,
            created_unix_seconds=time.time(), proxy_network_used=False))
        prepare(args.run / "cache")
        if not suite:
            status("prepare","complete",exit_code=0)
            return
    # Public evaluation and encoding must use exactly the snapshotted implementation.
    provenance = read_json(args.run / "provenance.json")
    for relative, expected in provenance["sources"].items():
        if sha256(REPO / relative) != expected:
            raise RuntimeError(f"Implementation changed after prepare: {relative}; use a new run")
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.stage == "train" or suite:
        from capx.memory.trajectory.train import train_suite
        status("train")
        train_suite(args.run / "cache", args.run, args.device)
        if not suite: status("train","complete",exit_code=0)
    if args.stage == "evaluate" or suite:
        from capx.memory.trajectory.evaluate import evaluate_suite
        status("evaluate")
        evaluate_suite(args.run, args.device)
        if not suite: status("evaluate","complete",exit_code=0)
    if suite:
        status("private_audit")
        subprocess.run([sys.executable,str(Path(__file__).with_name("audit_digit_trajectory_neighbors.py")),
                        "--run",str(args.run)],check=True,stdout=sys.stdout,stderr=sys.stderr)
        status("complete","complete",exit_code=0)
        print("SUITE_COMPLETE: six frozen encoders, public evaluation and separate private audit",flush=True)
    elif args.stage == "encode":
        from capx.memory.trajectory.evaluate import encode_cached_window
        if not args.checkpoint or not args.episode or args.end_step is None or not args.output:
            p.error("encode requires --checkpoint --episode --end-step --output")
        if args.output.exists(): raise FileExistsError(args.output)
        write_json(args.output, encode_cached_window(args.run / "cache", args.checkpoint, args.episode, args.end_step, args.device))
    elif args.stage == "search":
        from capx.memory.trajectory.model import search_index
        if not args.index or not args.query: p.error("search requires --index --query")
        import json
        print(json.dumps(search_index(args.index, read_json(args.query), k=args.k), indent=2))


if __name__ == "__main__":
    main()
