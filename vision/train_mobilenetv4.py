import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


class FilteredImageFolder(Dataset):
    def __init__(self, root, classes, transform=None, max_per_class=0):
        self.root = Path(root)
        self.classes = list(classes)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.transform = transform
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            added = 0
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, self.class_to_idx[class_name]))
                    added += 1
                    if max_per_class and added >= max_per_class:
                        break
        if not self.samples:
            raise FileNotFoundError(f"No valid images found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV4 for ship risk classification.")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Dataset root with train/val/test folders.")
    parser.add_argument("--model-name", type=str, default="mobilenetv4_conv_small.e2400_r224_in1k")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--report-dir", type=str, default="reports")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights.")
    parser.add_argument("--max-train-per-class", type=int, default=0, help="Use only N train images per class for quick testing.")
    parser.add_argument("--max-val-per-class", type=int, default=0, help="Use only N val images per class for quick testing.")
    return parser.parse_args()


def build_transforms(img_size):
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_tf, eval_tf


def count_images(folder):
    if not folder.exists():
        return 0
    return len([p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])


def active_classes(data_dir):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    classes = []
    for class_dir in sorted([p for p in train_dir.iterdir() if p.is_dir()]):
        train_count = count_images(class_dir)
        val_count = count_images(val_dir / class_dir.name)
        if train_count > 0 and val_count > 0:
            classes.append(class_dir.name)
    return classes


def validate_dataset(data_dir):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("dataset must contain train/ and val/ folders.")
    class_dirs = [p for p in train_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise FileNotFoundError("dataset/train has no class folders.")
    inactive = []
    for split in ["train", "val"]:
        for cls in sorted(p.name for p in class_dirs):
            folder = data_dir / split / cls
            count = count_images(folder)
            if count == 0:
                inactive.append(str(folder))
    if inactive:
        print("Warning: these folders have no images and will be ignored until you add pictures:")
        for item in inactive:
            print(f"  - {item}")

    classes = active_classes(data_dir)
    if len(classes) < 2:
        raise ValueError("Need at least two classes with images in both train/ and val/.")
    print(f"Active classes for this run: {classes}")


def create_model(model_name, num_classes, pretrained=True):
    try:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception as exc:
        print(f"Could not create {model_name}: {exc}")
        if pretrained:
            try:
                print(f"Retrying {model_name} without pretrained weights.")
                model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
                return model, model_name
            except Exception as retry_exc:
                print(f"Retry failed: {retry_exc}")
        fallback = "mobilenetv3_small_100"
        print(f"Using fallback model: {fallback}")
        try:
            model = timm.create_model(fallback, pretrained=pretrained, num_classes=num_classes)
        except Exception:
            print(f"Using fallback model without pretrained weights: {fallback}")
            model = timm.create_model(fallback, pretrained=False, num_classes=num_classes)
        model_name = fallback
    return model, model_name


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(loader, leave=False)
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loop.set_description("train" if train else "val")
        loop.set_postfix(loss=loss.item())

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="evaluate", leave=False):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())
    return y_true, y_pred


def save_confusion_matrix(cm, class_names, output_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    validate_dataset(args.data_dir)

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.img_size)
    classes = active_classes(args.data_dir)
    train_set = FilteredImageFolder(
        Path(args.data_dir) / "train",
        classes=classes,
        transform=train_tf,
        max_per_class=args.max_train_per_class,
    )
    val_set = FilteredImageFolder(
        Path(args.data_dir) / "val",
        classes=classes,
        transform=eval_tf,
        max_per_class=args.max_val_per_class,
    )
    test_path = Path(args.data_dir) / "test"
    test_set = FilteredImageFolder(test_path, classes=classes, transform=eval_tf, max_per_class=args.max_val_per_class) if test_path.exists() else None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if test_set else None

    class_names = classes
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {class_names}")

    model, actual_model_name = create_model(args.model_name, num_classes, pretrained=not args.no_pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_acc = 0.0
    history = []
    best_path = output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_name": actual_model_name,
                    "class_names": class_names,
                    "img_size": args.img_size,
                    "val_acc": best_acc,
                },
                best_path,
            )
            print(f"Saved best model: {best_path} val_acc={best_acc:.4f}")

    with open(report_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    checkpoint = torch.load(best_path, map_location=device)
    model, _ = create_model(checkpoint["model_name"], len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    eval_loader = test_loader if test_loader else val_loader
    split_name = "test" if test_loader else "val"
    y_true, y_pred = evaluate(model, eval_loader, device)
    labels = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    with open(report_dir / f"{split_name}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    save_confusion_matrix(cm, class_names, report_dir / f"{split_name}_confusion_matrix.png")

    print("\nFinal report:")
    print(report)
    print(f"Best model: {best_path}")
    print(f"Reports saved to: {report_dir}")


if __name__ == "__main__":
    main()
