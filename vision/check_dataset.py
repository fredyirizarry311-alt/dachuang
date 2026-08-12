from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    root = Path("dataset")
    if not root.exists():
        print("dataset folder not found.")
        return

    total = 0
    for split in ["train", "val", "test"]:
        split_dir = root / split
        print(f"\n[{split}]")
        if not split_dir.exists():
            print("  missing")
            continue
        for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            count = len([p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
            total += count
            print(f"  {class_dir.name:16s} {count:5d}")
    print(f"\nTotal images: {total}")


if __name__ == "__main__":
    main()
