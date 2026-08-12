import argparse
import json
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms


CN_NAMES = {
    "normal_ship": "正常船舶",
    "fire": "火灾",
    "cabin_smoke": "舱内烟雾",
    "external_smoke": "船舶外部烟雾",
    "water_ingress": "进水积水异常",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one image and produce a semantic token.")
    parser.add_argument("--model", type=str, default="models/best_model.pth")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--time", type=str, default="00:00:00")
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.model, map_location="cpu")
    class_names = checkpoint["class_names"]
    img_size = checkpoint.get("img_size", 224)
    model_name = checkpoint.get("model_name", "mobilenetv4_conv_small.e2400_r224_in1k")

    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    image = Image.open(args.image).convert("RGB")
    x = tf(image).unsqueeze(0)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1)[0]
    confidence, idx = torch.max(prob, dim=0)
    event = class_names[idx.item()]
    confidence = float(confidence.item())

    importance = confidence
    if event != "normal_ship":
        importance = min(1.0, confidence + 0.08)

    token = {
        "time": args.time,
        "modality": "vision",
        "type": "ship_risk",
        "event": event,
        "event_cn": CN_NAMES.get(event, event),
        "confidence": round(confidence, 4),
        "importance": round(importance, 4),
        "source": str(Path(args.image).name),
    }

    text = json.dumps(token, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
