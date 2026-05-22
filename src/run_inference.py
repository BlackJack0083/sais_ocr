#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageFile, ImageOps
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from ultralytics import YOLO


ImageFile.LOAD_TRUNCATED_IMAGES = True

INPUT_DIR = Path(os.getenv("INPUT_DIR", "/saisdata"))
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", "/saisresult/prediction.json"))
DETECTOR_WEIGHTS = Path(os.getenv("DETECTOR_WEIGHTS", "/app/models/detector_best.pt"))
CLASSIFIER_WEIGHTS = Path(os.getenv("CLASSIFIER_WEIGHTS", "/app/models/classifier_best.pt"))
REQUEST_USE_GPU = os.getenv("USE_GPU", "1") not in {"0", "false", "False", "no", "NO"}
DETECT_CONF = float(os.getenv("DETECT_CONF", "0.12"))
DETECT_IOU = float(os.getenv("DETECT_IOU", "0.45"))
DETECT_IMGSZ = int(os.getenv("DETECT_IMGSZ", "1280"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "4096"))
CLASSIFY_BATCH_SIZE = int(os.getenv("CLASSIFY_BATCH_SIZE", "128"))
BOX_EXPAND_RATIO = float(os.getenv("BOX_EXPAND_RATIO", "0.00"))
CLASSIFY_MIN_PROB = float(os.getenv("CLASSIFY_MIN_PROB", "0.20"))
CLASSIFY_MIN_MARGIN = float(os.getenv("CLASSIFY_MIN_MARGIN", "0.00"))
IMAGE_MEAN = (0.85233593, 0.85246795, 0.8517555)
IMAGE_STD = (0.31232414, 0.3122127, 0.31273854)


def choose_device() -> str:
    if REQUEST_USE_GPU and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def find_images() -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

    if INPUT_DIR.exists():
        return sorted(path for path in INPUT_DIR.rglob("*") if path.suffix.lower() in suffixes)

    fallback_root = Path("/saisdata")
    if fallback_root.exists():
        return sorted(path for path in fallback_root.rglob("*") if path.suffix.lower() in suffixes)

    return []


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def infer(self, embeddings: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        return cosine * self.s


class EfficientNetArcFace(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.embedding(features)
        return F.normalize(embeddings)


@dataclass
class Detection:
    bbox: list[int]
    crop: Image.Image


def pad_to_square(image: Image.Image, fill: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    pad_left = (side - width) // 2
    pad_top = (side - height) // 2
    pad_right = side - width - pad_left
    pad_bottom = side - height - pad_top
    return ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=fill)


class AncientCharRecognizer:
    def __init__(self, detector_weights: Path, classifier_weights: Path, device: str) -> None:
        if not detector_weights.exists():
            raise FileNotFoundError(f"Detector weights not found: {detector_weights}")
        if not classifier_weights.exists():
            raise FileNotFoundError(f"Classifier weights not found: {classifier_weights}")

        self.device = torch.device(device)
        self.detector = YOLO(str(detector_weights))

        checkpoint = torch.load(classifier_weights, map_location="cpu", weights_only=False)
        label_map: dict[str, int] = checkpoint["label_map"]
        embedding_dim = int(checkpoint["args"]["embedding_dim"])
        arcface_s = float(checkpoint["args"]["arcface_s"])
        arcface_m = float(checkpoint["args"]["arcface_m"])
        num_classes = len(label_map)

        self.classifier = EfficientNetArcFace(embedding_dim=embedding_dim).to(self.device)
        self.arcface = ArcMarginProduct(
            in_features=embedding_dim,
            out_features=num_classes,
            s=arcface_s,
            m=arcface_m,
        ).to(self.device)
        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.arcface.load_state_dict(checkpoint["arcface_state_dict"])
        self.classifier.eval()
        self.arcface.eval()

        self.idx_to_text = self._build_idx_to_text(label_map)
        self.transform = transforms.Compose(
            [
                transforms.Lambda(pad_to_square),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            ]
        )

    @staticmethod
    def _build_idx_to_text(label_map: dict[str, int]) -> dict[int, str]:
        print("Classifier label mode: direct Unicode labels")
        return {int(index): class_name for class_name, index in label_map.items()}

    def _predict_texts(self, crops: Iterable[Image.Image]) -> list[str]:
        tensors = [self.transform(crop.convert("RGB")) for crop in crops]
        if not tensors:
            return []

        results: list[str] = []
        with torch.inference_mode():
            for start in range(0, len(tensors), CLASSIFY_BATCH_SIZE):
                batch = torch.stack(tensors[start : start + CLASSIFY_BATCH_SIZE]).to(self.device)
                embeddings = self.classifier(batch)
                logits = self.arcface.infer(embeddings)
                probs = logits.softmax(dim=1)
                top2_probs, top2_indices = probs.topk(k=min(2, probs.shape[1]), dim=1)
                pred_indices = top2_indices[:, 0].tolist()
                pred_probs = top2_probs[:, 0].tolist()
                if top2_probs.shape[1] > 1:
                    pred_margins = (top2_probs[:, 0] - top2_probs[:, 1]).tolist()
                else:
                    pred_margins = pred_probs

                for pred_index, pred_prob, pred_margin in zip(pred_indices, pred_probs, pred_margins):
                    if pred_prob < CLASSIFY_MIN_PROB or pred_margin < CLASSIFY_MIN_MARGIN:
                        results.append("")
                        continue
                    results.append(self.idx_to_text.get(int(pred_index), ""))
        return results

    def _expand_and_clip_box(self, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> list[int]:
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        expand_x = box_w * BOX_EXPAND_RATIO
        expand_y = box_h * BOX_EXPAND_RATIO

        left = max(0, int(round(x1 - expand_x)))
        top = max(0, int(round(y1 - expand_y)))
        right = min(width, int(round(x2 + expand_x)))
        bottom = min(height, int(round(y2 + expand_y)))
        return [left, top, max(0, right - left), max(0, bottom - top)]

    def detect_and_recognize(self, image_path: Path) -> list[dict]:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            image_width, image_height = rgb_image.size

            prediction = self.detector.predict(
                source=str(image_path),
                conf=DETECT_CONF,
                iou=DETECT_IOU,
                imgsz=DETECT_IMGSZ,
                max_det=MAX_DETECTIONS,
                device=0 if self.device.type == "cuda" else "cpu",
                verbose=False,
            )[0]

            detections: list[Detection] = []
            boxes = prediction.boxes.xyxy.tolist() if prediction.boxes is not None else []
            for box in boxes:
                x1, y1, x2, y2 = box
                bbox = self._expand_and_clip_box(x1, y1, x2, y2, image_width, image_height)
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    continue
                crop = rgb_image.crop((x, y, x + w, y + h))
                detections.append(Detection(bbox=bbox, crop=crop))

        texts = self._predict_texts([item.crop for item in detections])
        results: list[dict] = []
        for detection, text in zip(detections, texts):
            if not text:
                continue
            results.append({"bbox": detection.bbox, "text": text})

        results.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        return results


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    image_paths = find_images()
    print(f"Input directory: {INPUT_DIR}")
    print(f"Images found: {len(image_paths)}")
    print(f"Detector weights: {DETECTOR_WEIGHTS}")
    print(f"Classifier weights: {CLASSIFIER_WEIGHTS}")
    print(f"BOX_EXPAND_RATIO={BOX_EXPAND_RATIO}")
    print(f"CLASSIFY_MIN_PROB={CLASSIFY_MIN_PROB}")
    print(f"CLASSIFY_MIN_MARGIN={CLASSIFY_MIN_MARGIN}")

    if not image_paths:
        OUTPUT_FILE.write_text("{}", encoding="utf-8")
        print(f"No images found. Saved empty result to {OUTPUT_FILE}")
        return

    device = choose_device()
    print(f"Use GPU requested: {REQUEST_USE_GPU}")
    print(f"Use device actual: {device}")

    recognizer = AncientCharRecognizer(
        detector_weights=DETECTOR_WEIGHTS,
        classifier_weights=CLASSIFIER_WEIGHTS,
        device=device,
    )

    predictions: dict[str, list[dict]] = {}
    for index, image_path in enumerate(image_paths, start=1):
        if index == 1 or index % 50 == 0:
            print(f"[{index}/{len(image_paths)}] {image_path.name}")
        try:
            predictions[image_path.stem] = recognizer.detect_and_recognize(image_path)
        except Exception as exc:
            print(f"Warning: failed to process {image_path}: {exc}")
            traceback.print_exc()
            predictions[image_path.stem] = []

    OUTPUT_FILE.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
