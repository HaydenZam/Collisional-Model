"""Shared helpers for the study scripts: repo on sys.path, output folders, tee'd output."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def out_dir(study):
    """runs/<study>/ (git-ignored), created if needed."""
    d = REPO / "runs" / study
    d.mkdir(parents=True, exist_ok=True)
    return d


class Tee:
    """Context manager copying everything printed to a log file as well."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self._file = open(self.path, "w")
        self._stdout = sys.stdout
        sys.stdout = self
        return self

    def write(self, s):
        self._stdout.write(s)
        self._file.write(s)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def __exit__(self, *exc):
        sys.stdout = self._stdout
        self._file.close()
