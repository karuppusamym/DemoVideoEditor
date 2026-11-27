"""Build the x64 Windows release ZIP from a Git commit."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from zipfile import ZIP_DEFLATED, ZipFile

PYTHON_URL = "https://www.python.org/ftp/python/3.13.17/python-3.13.17-embed-amd64.zip"
PYTHON_SHA256 = "9e35319bffedab8b36ad44d8f377cb4f1733bcec257a0b6d04d94f01486ddb38"
PIP_VERSION = "26.3"
ROOT = Path(__file__).resolve().parents[1]


def build(ref: str, output: Path) -> None:
    if sys.platform != "win32" or platform.machine().lower() != "amd64" or sys.version_info[:2] != (3, 13):
        raise SystemExit("Use x64 Windows and Python 3.13 to build this ZIP.")
    archive = subprocess.check_output(
        ["git", "archive", "--format=zip", ref, "clipeditor", "portable", "docs", "requirements.txt",
         "LICENSE", "README.md", "CONTRIBUTING.md", "COMPARISON.md", "CODE_OF_CONDUCT.md", "SECURITY.md"],
        cwd=ROOT,
    )
    with TemporaryDirectory(prefix="clipeditor-release-") as temp:
        build_dir = Path(temp)
        bundle = build_dir / "ClipEditor"
        with ZipFile(io.BytesIO(archive)) as sources:
            sources.extractall(bundle)
        runtime = bundle / "runtime"
        with urlopen(PYTHON_URL, timeout=60) as response:
            python_zip = response.read()
        if hashlib.sha256(python_zip).hexdigest() != PYTHON_SHA256:
            raise RuntimeError("The Python runtime checksum did not match python.org.")
        with ZipFile(io.BytesIO(python_zip)) as python_files:
            python_files.extractall(runtime)
        (runtime / "python313._pth").write_text("python313.zip\n.\n..\n", encoding="ascii")

        pip = [sys.executable, "-m", "pip", "--isolated"]
        index = ["--index-url", "https://pypi.org/simple/", "--only-binary=:all:", "--disable-pip-version-check"]
        subprocess.run([*pip, "download", *index, "--no-deps", "--dest", str(bundle), f"pip=={PIP_VERSION}"],
                       check=True)
        report = build_dir / "report.json"
        subprocess.run([*pip, "install", *index, "--dry-run", "--ignore-installed", "--report", str(report),
                        "-r", str(bundle / "requirements.txt")], check=True)
        packages = json.loads(report.read_text(encoding="utf-8"))["install"]
        requirements = [f'{item["metadata"]["name"]}=={item["metadata"]["version"]} --hash=sha256:'
                        f'{item["download_info"]["archive_info"]["hashes"]["sha256"]}' for item in packages]
        (bundle / "requirements.lock").write_text("\n".join(sorted(requirements, key=str.casefold)) + "\n",
                                                  encoding="ascii")
        for name in ("Launch ClipEditor.bat", "bootstrap.py"):
            (bundle / "portable" / name).rename(bundle / name)
        shutil.rmtree(bundle / "portable")

        output.parent.mkdir(parents=True, exist_ok=True)
        pending = output.with_suffix(".tmp")
        try:
            with ZipFile(pending, "w", compression=ZIP_DEFLATED, compresslevel=6) as release:
                for file in sorted(bundle.rglob("*")):
                    if file.is_file():
                        release.write(file, file.relative_to(build_dir).as_posix())
            pending.replace(output)
        finally:
            pending.unlink(missing_ok=True)
    with output.open("rb") as archive_file:
        checksum = hashlib.file_digest(archive_file, "sha256").hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{checksum}  {output.name}\n", encoding="ascii")
    print(f"Built {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD", help="Git commit or tag to build")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "ClipEditor-Windows-x64.zip")
    args = parser.parse_args()
    build(args.ref, args.output.resolve())
