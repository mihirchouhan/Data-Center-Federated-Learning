"""Build a source-only ZIP; explicit allowlist excludes data and all run outputs."""
import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {"main.py", "pipeline.py", "README.md", "requirements.txt", ".gitignore"}
SOURCE_DIRS = {"src", "tests", "tools", "configs"}


def package(destination):
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(ROOT)
        if "__pycache__" in relative.parts:
            continue
        if (len(relative.parts) == 1 and relative.name in TOP_FILES
                or relative.parts[0] in SOURCE_DIRS and path.suffix in (".py", ".json")):
            files.append(path)
    required = {"main.py", "pipeline.py", "README.md", "requirements.txt", "configs/default.json", "configs/m100.json"}
    included = {p.relative_to(ROOT).as_posix() for p in files}
    if not required.issubset(included):
        raise RuntimeError(f"Project is incomplete: {sorted(required - included)}")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"{ROOT.name}/{path.relative_to(ROOT).as_posix()}")
    print(f"Created {destination} ({destination.stat().st_size:,} bytes, {len(files)} source files; no datasets or outputs)")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT.parent / f"{ROOT.name}.zip")
    package(parser.parse_args().destination)
