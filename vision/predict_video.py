import argparse
import json
from pathlib import Path

import cv2
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
    parser = argparse.ArgumentParser(description="Sample video frames and produce semantic tokens.")
    parser.add_argument("--model", type=str, default="models/best_model.pth")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--seconds", type=float, default=1.0, help="Sample one frame every N seconds.")
    parser.add_argument("--threshold", type=float, default=0.50, help="Only output abnormal tokens above this confidence.")
    parser.add_argument("--output", type=str, default="reports/video_tokens.json")
    return parser.parse_args()


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(int(fps * args.seconds), 1)
    frame_idx = 0
    tokens = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        x = tf(image).unsqueeze(0)
        with torch.no_grad():
            prob = torch.softmax(model(x), dim=1)[0]
        confidence, idx = torch.max(prob, dim=0)
        event = class_names[idx.item()]
        confidence = float(confidence.item())
        t = frame_idx / fps

        if event != "normal_ship" and confidence >= args.threshold:
            tokens.append(
                {
                    "time": format_time(t),
                    "modality": "vision",
                    "type": "ship_risk",
                    "event": event,
                    "event_cn": CN_NAMES.get(event, event),
                    "confidence": round(confidence, 4),
                    "importance": round(min(1.0, confidence + 0.08), 4),
                    "frame": frame_idx,
                }
            )
        frame_idx += 1

    cap.release()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(tokens, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(tokens)} tokens to {output}")


if __name__ == "__main__":
    main()
