"""Download YOLOv9 ONNX weights from hf-mirror for same-origin loading."""
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MIRROR = "https://hf-mirror.com"
REPO = "Xenova/yolov9-c_all"
FILES = [
    "config.json",
    "preprocessor_config.json",
    "onnx/model_quantized.onnx",
]
SAMPLES = {
    "street.jpg": f"{MIRROR}/datasets/Xenova/transformers.js-docs/resolve/main/city-streets.jpg",
    "crowd.jpg": f"{MIRROR}/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg",
}


def fetch(url: str, dest: Path, min_size: int = 0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > min_size:
        print(f"skip {dest} ({dest.stat().st_size} bytes)")
        return
    print(f"get  {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = resp.read()
    dest.write_bytes(data)
    print(f"save {dest} ({len(data)} bytes)")


def main() -> None:
    for rel in FILES:
        fetch(
            f"{MIRROR}/{REPO}/resolve/main/{rel}",
            ROOT / "models" / REPO / rel,
            min_size=1_000 if rel.endswith(".onnx") else 0,
        )
    for name, url in SAMPLES.items():
        fetch(url, ROOT / "samples" / name, min_size=10_000)


if __name__ == "__main__":
    main()
