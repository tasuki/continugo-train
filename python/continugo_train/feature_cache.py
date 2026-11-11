"""Lossless, indexed per-SGF training-feature sidecars.

SGFs remain authoritative. Caches contain existing CGFEAT02 packets, independently
zlib-compressed per eligible ply; no model, split, or sampler state is stored.
Build explicitly with python -m continugo_train.feature_cache --data DIR ... .
"""

import argparse
import hashlib
import json
import os
import struct
import tempfile
import time
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from .features import (
    COARSE_PLANES,
    MOVE_FEATURES,
    SCHEMA_VERSION,
    TARGET_VERSION,
    FeatureClient,
    decode_sample,
    feature_binary,
    inspect_game,
)

MAGIC = b"CGCACHE1"
PREFIX = struct.Struct("<8sQ32s")
MAX_INDEX = 32 * 1024 * 1024
MAX_PACKET = 1_000_000_000
MODES = ("off", "prefer", "require")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sidecar_path(sgf):
    sgf = Path(sgf)
    return sgf.parent / "features" / (sgf.stem + ".cfeat")


@lru_cache(maxsize=16)
def _binary_hash(path, size, mtime_ns):
    return file_hash(path)


def extractor_hash(binary=None):
    path = feature_binary(binary)
    stat = path.stat()
    return _binary_hash(str(path), stat.st_size, stat.st_mtime_ns)


def semantics(density, binary=None):
    return {
        "format": 1,
        "codec": "zlib",
        "schema": SCHEMA_VERSION,
        "targets": TARGET_VERSION,
        "coarse": list(COARSE_PLANES),
        "moves": list(MOVE_FEATURES),
        "samples_per_diameter": density,
        "extractor_sha256": extractor_hash(binary),
        "domain": "training-target-cells",
    }


class CacheReader:
    def __init__(self, sgf, *, density=4, binary=None, file_sha256=None):
        self.path = sidecar_path(sgf)
        self.stream = self.path.open("rb")
        try:
            size = self.path.stat().st_size
            prefix = self.stream.read(PREFIX.size)
            if len(prefix) != PREFIX.size:
                raise ValueError("truncated cache prefix")
            magic, offset, checksum = PREFIX.unpack(prefix)
            if magic != MAGIC or not PREFIX.size <= offset < size:
                raise ValueError("invalid cache format/index offset")
            if size - offset > MAX_INDEX:
                raise ValueError("oversized cache index")
            self.stream.seek(offset)
            raw = self.stream.read()
            if hashlib.sha256(raw).digest() != checksum:
                raise ValueError("cache index checksum mismatch")
            self.metadata = json.loads(raw)
            if self.metadata["semantics"] != semantics(density, binary):
                raise ValueError("incompatible cache extractor/schema/density")
            actual = file_hash(sgf)
            if actual != self.metadata["sgf_sha256"] or (
                file_sha256 is not None and actual != file_sha256
            ):
                raise ValueError("SGF changed: stale feature cache")
            self.records = {}
            end = PREFIX.size
            for record in self.metadata["records"]:
                ply, start, length, raw_length, checksum = record
                if (
                    not isinstance(ply, int)
                    or ply < 0
                    or ply in self.records
                    or start != end
                    or length <= 0
                    or start + length > offset
                    or not 0 < raw_length <= MAX_PACKET
                    or not isinstance(checksum, str)
                    or len(checksum) != 64
                ):
                    raise ValueError("invalid cache record index")
                self.records[ply] = record
                end = start + length
            if end != offset or sorted(self.records) != self.metadata["game"]["eligible_plies"]:
                raise ValueError("incomplete cache position index")
        except BaseException:
            self.stream.close()
            raise

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def packet(self, ply):
        if ply not in self.records:
            raise ValueError(f"ply {ply} missing from cache")
        _, offset, length, raw_length, checksum = self.records[ply]
        self.stream.seek(offset)
        compressed = self.stream.read(length)
        decoder = zlib.decompressobj()
        try:
            raw = decoder.decompress(compressed, raw_length + 1)
        except zlib.error as error:
            raise ValueError("corrupt compressed cache record") from error
        if (
            len(compressed) != length
            or len(raw) != raw_length
            or not decoder.eof
            or decoder.unused_data
            or decoder.unconsumed_tail
            or hashlib.sha256(raw).hexdigest() != checksum
        ):
            raise ValueError("cache record length/checksum mismatch")
        return raw

    def load(self, ply):
        sample = decode_sample(self.packet(ply))
        game = self.metadata["game"]
        if (
            sample.ply != ply
            or sample.full_domain
            or sample.samples_per_diameter != self.metadata["semantics"]["samples_per_diameter"]
            or sample.board_size != game["board_size"]
            or sample.resolution != game["resolution"]
            or sample.diameter != game["diameter"]
        ):
            raise ValueError("cache packet does not match game/index")
        return sample


class SampleReader:
    """Bounded per-worker open-file LRU; Rust is started only for live extraction."""

    def __init__(self, binary=None, *, cache="off", profile=False):
        if cache not in MODES:
            raise ValueError(f"feature cache mode must be one of {MODES}")
        self.binary, self.cache, self.profile = binary, cache, profile
        self.readers = OrderedDict()
        self.live = None
        self.last_profile = None

    @property
    def process(self):
        return self.live.process

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()
        if self.live is not None:
            self.live.close()

    def load(self, sgf, ply, *, density=4, file_sha256=None):
        self.last_profile = None
        key = (str(Path(sgf).resolve()), density, file_sha256)
        if self.cache != "off":
            if key in self.readers:
                reader = self.readers.pop(key)
            else:
                try:
                    reader = CacheReader(
                        sgf, density=density, binary=self.binary, file_sha256=file_sha256
                    )
                except FileNotFoundError:
                    if self.cache == "require":
                        raise
                    reader = None
                if len(self.readers) >= 8:
                    self.readers.popitem(last=False)[1].close()
            if reader is not None:
                self.readers[key] = reader
                return reader.load(ply)
        if self.live is None:
            self.live = FeatureClient(self.binary, profile=self.profile)
        sample = self.live.load(sgf, ply, density=density, file_sha256=file_sha256)
        self.last_profile = self.live.last_profile
        return sample


def build_cache(sgf, *, density=4, binary=None, rebuild=False):
    """Single writer, atomic publication. Never writes to the SGF itself."""
    sgf = Path(sgf).resolve()
    destination = sidecar_path(sgf)
    started = time.perf_counter()
    if destination.exists() and not rebuild:
        with CacheReader(sgf, density=density, binary=binary) as reader:
            # A resumed build validates every existing compressed record, not just its header.
            for ply in reader.records:
                reader.load(ply)
            return {
                "sgf": str(sgf),
                "cache": str(destination),
                "status": "verified-existing",
                "positions": len(reader.records),
                "bytes": destination.stat().st_size,
                "seconds": time.perf_counter() - started,
            }
    info = inspect_game(sgf, density, binary)
    if not info["labeled"] or not info["eligible_plies"]:
        raise ValueError(f"no eligible labeled positions: {sgf}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.with_suffix(destination.suffix + ".lock")
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    temporary = None
    try:
        if destination.exists() and not rebuild:
            raise FileExistsError(destination)
        fd, name = tempfile.mkstemp(
            prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(name)
        records = []
        raw_total = 0
        with os.fdopen(fd, "w+b") as output, FeatureClient(binary, retain_packet=True) as client:
            output.write(bytes(PREFIX.size))
            for ply in info["eligible_plies"]:
                client.load(sgf, ply, density=density, file_sha256=info["file_sha256"])
                packet = client.last_packet
                compressed = zlib.compress(packet, level=3)
                records.append(
                    [
                        ply,
                        output.tell(),
                        len(compressed),
                        len(packet),
                        hashlib.sha256(packet).hexdigest(),
                    ]
                )
                raw_total += len(packet)
                output.write(compressed)
            metadata = {
                "semantics": semantics(density, binary),
                "sgf_sha256": info["file_sha256"],
                "game": info,
                "records": records,
            }
            raw_index = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()
            if len(raw_index) > MAX_INDEX:
                raise ValueError("oversized cache index")
            offset = output.tell()
            output.write(raw_index)
            output.seek(0)
            output.write(PREFIX.pack(MAGIC, offset, hashlib.sha256(raw_index).digest()))
            output.flush()
            os.fsync(output.fileno())
        if file_hash(sgf) != info["file_sha256"]:
            raise ValueError("SGF changed while building cache")
        os.replace(temporary, destination)
        temporary = None
        return {
            "sgf": str(sgf),
            "cache": str(destination),
            "status": "built",
            "positions": len(records),
            "bytes": destination.stat().st_size,
            "raw_bytes": raw_total,
            "sgf_sha256": info["file_sha256"],
            "seconds": time.perf_counter() - started,
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True, type=Path)
    parser.add_argument("--feature-binary", type=Path)
    parser.add_argument("--density", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.density < 1:
        parser.error("workers and density must be positive")
    paths = set()
    for path in args.data:
        if path.is_dir():
            paths.update(
                p.resolve() for p in path.rglob("*") if p.is_file() and p.suffix.lower() == ".sgf"
            )
        elif path.is_file() and path.suffix.lower() == ".sgf":
            paths.add(path.resolve())
        else:
            parser.error(f"not an SGF or directory: {path}")
    if not paths:
        parser.error("no SGFs found")

    def build(path):
        return build_cache(
            path, density=args.density, binary=args.feature_binary, rebuild=args.rebuild
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(build, sorted(paths)):
            print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
