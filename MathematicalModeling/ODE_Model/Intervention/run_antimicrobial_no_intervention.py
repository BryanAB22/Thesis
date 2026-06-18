"""Run the MATLAB antimicrobial + nitric-oxide experiment from Python."""

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_FILES = (
    "model.m",
    "modelDriver.m",
    "load_pars_Init_Copeland_Edited.m",
    "run_antimicrobial_no_intervention.m",
)


def matlab_executable_from_path(path: Path) -> Path | None:
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        return None
    for relative in (
        Path("bin") / "matlab",
        Path("Contents") / "MacOS" / "MATLAB",
        Path("Contents") / "MacOS" / "matlab",
    ):
        candidate = path / relative
        if candidate.is_file():
            return candidate.resolve()
    return None


def find_matlab_executable() -> Path:
    matlab_on_path = shutil.which("matlab")
    if matlab_on_path:
        return Path(matlab_on_path).resolve()

    candidates = sorted(
        glob.glob(str(Path.home() / "Desktop" / "MATLAB_R*.app"))
        + glob.glob("/Applications/MATLAB_R*.app"),
        reverse=True,
    )
    for app in candidates:
        executable = matlab_executable_from_path(Path(app))
        if executable:
            return executable

    raise FileNotFoundError(
        "MATLAB was not found. Add it to PATH or pass --matlab with "
        "the MATLAB executable or application path."
    )


def validate_folder(folder: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {folder}:\n  - " + "\n  - ".join(missing)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "matlab_folder",
        nargs="?",
        default=None,
        help="Folder containing the MATLAB files; defaults to this script's folder.",
    )
    parser.add_argument(
        "--matlab",
        default=None,
        help="Path to the MATLAB executable or MATLAB .app bundle.",
    )
    args = parser.parse_args()

    folder = (
        Path(__file__).resolve().parent
        if args.matlab_folder is None
        else Path(args.matlab_folder).expanduser().resolve()
    )

    try:
        validate_folder(folder)
        if args.matlab:
            executable = matlab_executable_from_path(
                Path(args.matlab).expanduser().resolve()
            )
            if executable is None:
                raise FileNotFoundError("The --matlab path is not runnable.")
        else:
            executable = find_matlab_executable()

        escaped_folder = str(folder).replace("'", "''")
        matlab_command = (
            f"cd('{escaped_folder}'); "
            "run('run_antimicrobial_no_intervention.m');"
        )

        print("Running antimicrobial + nitric-oxide intervention experiment...")
        completed = subprocess.run(
            [str(executable), "-batch", matlab_command],
            cwd=folder,
            check=False,
        )
        return completed.returncode

    except (FileNotFoundError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
