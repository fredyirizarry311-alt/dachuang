import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    metadata = json.loads((ROOT / "project-version.json").read_text(encoding="utf-8"))
    assert metadata["version"] == version, "VERSION and project-version.json disagree"
    assert metadata["vision_classes"] == 5
    assert metadata["dataset_images"] == 56141

    classes = [
        line.strip()
        for line in (ROOT / "vision" / "classes.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = ["normal_ship", "fire", "cabin_smoke", "external_smoke", "water_ingress"]
    assert classes == expected, f"class order mismatch: {classes}"

    required = [
        "README.md",
        "CHANGELOG.md",
        "docs/datasets.md",
        "docs/version-status.md",
        "vision/train_mobilenetv4.py",
        "vision/predict_image.py",
        "vision/predict_video.py",
        "jetson/build_int8.py",
        "jetson/infer_image.py",
    ]
    missing = [item for item in required if not (ROOT / item).exists()]
    assert not missing, f"missing release files: {missing}"

    print(f"release validation passed: v{version}, {len(classes)} classes, 56,141 images documented")


if __name__ == "__main__":
    main()
