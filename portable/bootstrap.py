"""Install the release dependencies beside the app on first launch."""
from __future__ import annotations

import msvcrt
from pathlib import Path
import site
import subprocess
import sys
from tempfile import TemporaryDirectory


def install_dependencies(root: Path, packages: Path) -> None:
    with (root / ".setup.lock").open("a+b") as lock:
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("ClipEditor setup is already running. Try again when it finishes.") from error
        if packages.is_dir():
            return
        print("Installing ClipEditor's dependencies from PyPI. Internet access is needed for this first step.",
              flush=True)
        pip_wheel = next(root.glob("pip-*.whl"))
        with TemporaryDirectory(prefix=".setup-", dir=root) as temp:
            pending = Path(temp) / "packages"
            run_pip = ("import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); "
                       "runpy.run_module('pip', run_name='__main__', alter_sys=True)")
            subprocess.run([sys.executable, "-c", run_pip, str(pip_wheel), "--isolated", "install",
                            "--index-url", "https://pypi.org/simple/", "--only-binary=:all:",
                            "--require-hashes", "--no-deps", "--disable-pip-version-check",
                            "--target", str(pending), "-r", str(root / "requirements.lock")], check=True)
            pending.rename(packages)


def main() -> int:
    root = Path(__file__).resolve().parent
    packages = root / "packages"
    if not packages.is_dir():
        try:
            install_dependencies(root, packages)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            print(f"ClipEditor setup failed: {error}", file=sys.stderr)
            return 1
    site.addsitedir(str(packages))
    from clipeditor.__main__ import main as run_editor

    return run_editor()


if __name__ == "__main__":
    sys.exit(main())
