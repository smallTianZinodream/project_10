"""Vendor ONNX Runtime WASM used by Transformers.js 3.7.2."""
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2/dist/"
FILES = [
    "ort-wasm-simd-threaded.jsep.mjs",
    "ort-wasm-simd-threaded.jsep.wasm",
]


def main() -> None:
    dest_dir = ROOT / "vendor" / "ort"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = dest_dir / name
        if dest.exists() and dest.stat().st_size > 1_000:
            print(f"skip {dest} ({dest.stat().st_size} bytes)")
            continue
        url = BASE + name
        print(f"get  {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f"save {dest} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
