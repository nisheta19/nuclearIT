"""Package only explicit project sources, approved organizer data and completed runs."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def build(destination):
    root = Path(__file__).resolve().parents[1]
    files = [root / name for name in ("README.md", "PRODUCT.md", "DESIGN.md", "requirements.txt", "requirements-dev.txt", "requirements.lock.txt", "pyproject.toml", "import_competition.py", "submission_check.py", "start_demo.ps1", ".gitignore", "data/source_manifest.json", ".impeccable/design.json", "artifacts/browser_verification.json")]
    files.append(root / '.gitattributes')
    for folder, pattern in [("vacv", "*.py"), ("tests", "*.py"), ("scripts", "*.py"), ("configs", "*.json"), ("docs", "*.md"), ("data/raw/competition", "*.csv"), ("data/source_archives", "*.zip"), ("data/review/target_versions", "*")]:
        files.extend((root / folder).glob(pattern))
    for folder in ("web", "artifacts/baseline-original", "artifacts/nested-v2", "artifacts/kernel-v3", "artifacts/kernel-v3-review", "artifacts/advanced-v4", "artifacts/advanced-v4-review", "artifacts/medical-review-v4"):
        files.extend(p for p in (root / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(root))
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        return {"archive": str(destination.resolve()), "files": len(archive.namelist()), "bytes": destination.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="dist/vacv-workbench-v4.1.zip")
    print(build(parser.parse_args().output))
