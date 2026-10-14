"""Obtain the real MODERATE/Fluvius CSV with content verification.

The URL selects real measurements, not the synthetic datasets in the same
record. An existing valid file is reused. A mismatching local file is kept
and rejected; downloads replace the destination only after SHA-256 succeeds.
"""

from pathlib import Path
import hashlib
import urllib.request
import tempfile
import os

ROOT = Path(__file__).resolve().parent
URL = "https://zenodo.org/records/20525554/files/fluvius_wide_format.csv?download=1"
SHA256 = "bf490309adedd05549f90c97f6b44802f812eedf0f21bef4d0f5bafd3229c7b9"


def digest(path):
    """Compute SHA-256 in 1 MiB blocks without loading the whole CSV into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    """Verify or download the dataset beside this script.

    Use a temporary file in the destination directory so the final rename is
    atomic on that filesystem. Network/checksum errors propagate to the caller;
    the finally block removes a partial download, never a pre-existing CSV.
    """
    destination = ROOT / "fluvius_wide_format.csv"
    if destination.exists():
        if digest(destination) != SHA256:
            raise RuntimeError(
                "Existing CSV has unexpected SHA-256; preserve it and inspect before replacing."
            )
        print("Verified existing real dataset:", destination)
        return
    fd, name = tempfile.mkstemp(prefix="fluvius_", suffix=".part", dir=ROOT)
    os.close(fd)
    path = Path(name)
    try:
        with urllib.request.urlopen(URL, timeout=120) as response, path.open(
            "wb"
        ) as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        if digest(path) != SHA256:
            raise RuntimeError("Downloaded CSV failed SHA-256 verification")
        path.replace(destination)
    finally:
        path.unlink(missing_ok=True)
    print("Downloaded and verified:", destination)


if __name__ == "__main__":
    main()
