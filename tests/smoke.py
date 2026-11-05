"""End-to-end checks; --gpu requires a working GPU and tests both CUDA modes."""
import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build/bf16_stream"


def run(args, ok=True):
    result = subprocess.run([str(BIN), *map(str, args)], text=True, capture_output=True)
    assert (result.returncode == 0) == ok, result.stderr
    return result


with tempfile.TemporaryDirectory(dir=ROOT) as directory:
    path = Path(directory) / "test.bf16"
    for m, k, n, blocks in [(32, 128, 7, 1), (32, 128, 7, 3), (17, 2048, 3, 5)]:
        path.unlink(missing_ok=True)
        base = ["--file", path, "--m", m, "--k", k, "--n", n, "--blocks", blocks,
                "--threads", 2, "--repeats", 1, "--full-check"]
        run([*base, "--prepare"])
        run([*base, "--prepare"], ok=False)  # Existing data must never be overwritten.
        modes = ["cpu", "io"] + (["sync", "async"] if "--gpu" in sys.argv else [])
        for mode in modes:
            result = run([*base, "--mode", mode])
            row, = csv.DictReader(io.StringIO(result.stdout))
            assert int(row["bytes"]) == m * k * 2 * blocks
            assert int(row["checked"]) == (0 if mode == "io" else m * n * blocks)
            assert float(row["max_abs_error"]) == 0
            assert float(row["total_ms"]) > 0
        with path.open("r+b") as f:
            f.write(bytes(m * k * 2))
        bad = run([*base, "--mode", "cpu"], ok=False)
        assert "validation failed" in bad.stderr
        with path.open("r+b") as f:
            f.truncate(10)
        assert "size/type mismatch" in run([*base, "--mode", "cpu"], ok=False).stderr
    run(["--m", 3, "--k", 3], ok=False)
    run(["--mode", "typo"], ok=False)
    run(["--blocks", "-1"], ok=False)
    run(["--cache-mib", "-1"], ok=False)
    run(["--cache-mib", "4096"], ok=False)
    run(["--profile", "--mode", "cpu", "--repeats", "1"], ok=False)

print("PASS: full reference checks, block reuse/tails, corruption, truncation, CLI validation")
print("GPU modes verified" if "--gpu" in sys.argv else "GPU modes not run; use python3 tests/smoke.py --gpu")
