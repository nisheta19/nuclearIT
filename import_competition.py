"""Copy the three organizer CSVs byte-for-byte; never execute archive content."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def import_archive(source: Path, destination: Path):
    names = ("train.csv", "test.csv", "sample_submit.csv")
    with ZipFile(source) as archive:
        payloads = {name: archive.read(name) for name in names}
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        target = destination / name
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite different raw data: {target}")
    for name, payload in payloads.items():
        (destination / name).write_bytes(payload)
    manifest = {"archive": str(source.resolve()), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "files": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}}
    (destination / "archive_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/raw/competition"))
    args = parser.parse_args()
    print(json.dumps(import_archive(args.archive, args.output), ensure_ascii=False, indent=2))
