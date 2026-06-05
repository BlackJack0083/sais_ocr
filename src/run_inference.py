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

DEFAULT_INPUT_DIR = "/saisdata/50/eval/images"
INPUT_DIR = Path(os.getenv("INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", "/saisresult/prediction.json"))
DETECTOR_WEIGHTS = Path(os.getenv("DETECTOR_WEIGHTS", "/app/models/detector_best.pt"))
CLASSIFIER_WEIGHTS = Path(os.getenv("CLASSIFIER_WEIGHTS", "/app/models/classifier_best.pt"))
REQUEST_USE_GPU = os.getenv("USE_GPU", "1") not in {"0", "false", "False", "no", "NO"}
DETECT_CONF = float(os.getenv("DETECT_CONF", "0.08"))
DETECT_IOU = float(os.getenv("DETECT_IOU", "0.40"))
DETECT_IMGSZ = int(os.getenv("DETECT_IMGSZ", "1536"))
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "4096"))
CLASSIFY_BATCH_SIZE = int(os.getenv("CLASSIFY_BATCH_SIZE", "128"))
BOX_EXPAND_RATIO = float(os.getenv("BOX_EXPAND_RATIO", "0.00"))
CLASSIFY_MIN_PROB = float(os.getenv("CLASSIFY_MIN_PROB", "0.38"))
CLASSIFY_MIN_MARGIN = float(os.getenv("CLASSIFY_MIN_MARGIN", "0.07"))
CLASSIFY_MIN_COS = float(os.getenv("CLASSIFY_MIN_COS", "0.0"))
SLICE_SIZE = int(os.getenv("SLICE_SIZE", "0"))
SLICE_OVERLAP = int(os.getenv("SLICE_OVERLAP", "0"))
SLICE_MERGE_IOU = float(os.getenv("SLICE_MERGE_IOU", "0.50"))
DET_TTA_MODE = os.getenv("DET_TTA_MODE", "scale1536")
IMAGE_MEAN = (0.85233593, 0.85246795, 0.8517555)
IMAGE_STD = (0.31232414, 0.3122127, 0.31273854)


def choose_device() -> str:
    if REQUEST_USE_GPU and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def find_images() -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

    candidate_dirs = [
        INPUT_DIR,
        Path("/saisdata/50/eval/images"),
        Path("/saisdata/eval/images"),
        Path("/saisdata"),
    ]
    seen: set[Path] = set()
    for candidate in candidate_dirs:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return sorted(path for path in candidate.rglob("*") if path.suffix.lower() in suffixes)

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


@dataclass
class DetectionBox:
    bbox: list[int]
    score: float


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


def iou_xywh(a: list[int], b: list[int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(boxes: list[DetectionBox], iou_threshold: float) -> list[DetectionBox]:
    if not boxes:
        return []
    remaining = sorted(boxes, key=lambda item: item.score, reverse=True)
    kept: list[DetectionBox] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        filtered: list[DetectionBox] = []
        for candidate in remaining:
            if iou_xywh(current.bbox, candidate.bbox) < iou_threshold:
                filtered.append(candidate)
        remaining = filtered
    return kept


def sliding_starts(length: int, window: int, overlap: int) -> list[int]:
    if window <= 0 or length <= window:
        return [0]
    stride = max(1, window - overlap)
    starts = list(range(0, max(length - window, 0) + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


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
        checkpoint_args = checkpoint.get("args", {})
        embedding_dim = int(checkpoint_args["embedding_dim"])
        arcface_s = float(checkpoint_args["arcface_s"])
        arcface_m = float(checkpoint_args["arcface_m"])
        self.classifier_img_size = int(checkpoint_args.get("img_size", 128))
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
                transforms.Resize((self.classifier_img_size, self.classifier_img_size)),
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
                cosines = logits / self.arcface.s
                probs = logits.softmax(dim=1)
                top2_probs, top2_indices = probs.topk(k=min(2, probs.shape[1]), dim=1)
                top1_cosines = cosines.gather(1, top2_indices[:, :1]).squeeze(1).tolist()
                pred_indices = top2_indices[:, 0].tolist()
                pred_probs = top2_probs[:, 0].tolist()
                if top2_probs.shape[1] > 1:
                    pred_margins = (top2_probs[:, 0] - top2_probs[:, 1]).tolist()
                else:
                    pred_margins = pred_probs

                for pred_index, pred_prob, pred_margin, pred_cos in zip(
                    pred_indices, pred_probs, pred_margins, top1_cosines
                ):
                    if (
                        pred_prob < CLASSIFY_MIN_PROB
                        or pred_margin < CLASSIFY_MIN_MARGIN
                        or pred_cos < CLASSIFY_MIN_COS
                    ):
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

    def _detector_device(self) -> int | str:
        return 0 if self.device.type == "cuda" else "cpu"

    def _predict_raw_boxes(self, image: Image.Image, imgsz: int) -> list[DetectionBox]:
        prediction = self.detector.predict(
            source=image,
            conf=DETECT_CONF,
            iou=DETECT_IOU,
            imgsz=imgsz,
            max_det=MAX_DETECTIONS,
            device=self._detector_device(),
            verbose=False,
        )[0]
        if prediction.boxes is None:
            return []
        coords = prediction.boxes.xyxy.tolist()
        scores = prediction.boxes.conf.tolist()
        boxes: list[DetectionBox] = []
        for (x1, y1, x2, y2), score in zip(coords, scores):
            boxes.append(
                DetectionBox(
                    bbox=[
                        int(round(x1)),
                        int(round(y1)),
                        max(0, int(round(x2)) - int(round(x1))),
                        max(0, int(round(y2)) - int(round(y1))),
                    ],
                    score=float(score),
                )
            )
        return boxes

    def _predict_full_image_boxes(self, rgb_image: Image.Image) -> list[DetectionBox]:
        image_width, _ = rgb_image.size
        boxes: list[DetectionBox] = []

        def add_view(view: Image.Image, imgsz: int, flipped: bool) -> None:
            for box in self._predict_raw_boxes(view, imgsz=imgsz):
                x, y, w, h = box.bbox
                if flipped:
                    x = image_width - (x + w)
                boxes.append(DetectionBox(bbox=[x, y, w, h], score=box.score))

        # Deduplicate views so DETECT_IMGSZ=1536 with DET_TTA_MODE=scale1536
        # does not run the exact same pass twice and inflate FP.
        planned_views: list[tuple[int, bool]] = [(DETECT_IMGSZ, False)]
        if DET_TTA_MODE in {"hflip", "hflip+scale1536"}:
            planned_views.append((DETECT_IMGSZ, True))
        if DET_TTA_MODE in {"scale1536", "hflip+scale1536"}:
            planned_views.append((1536, False))
        if DET_TTA_MODE == "hflip+scale1536":
            planned_views.append((1536, True))

        seen_views: set[tuple[int, bool]] = set()
        flipped_image = rgb_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        for imgsz, flipped in planned_views:
            key = (imgsz, flipped)
            if key in seen_views:
                continue
            seen_views.add(key)
            add_view(flipped_image if flipped else rgb_image, imgsz=imgsz, flipped=flipped)
        return boxes

    def _predict_sliced_boxes(self, rgb_image: Image.Image) -> list[DetectionBox]:
        if SLICE_SIZE <= 0:
            return []
        width, height = rgb_image.size
        boxes: list[DetectionBox] = []
        for top in sliding_starts(height, SLICE_SIZE, SLICE_OVERLAP):
            for left in sliding_starts(width, SLICE_SIZE, SLICE_OVERLAP):
                right = min(width, left + SLICE_SIZE)
                bottom = min(height, top + SLICE_SIZE)
                tile = rgb_image.crop((left, top, right, bottom))
                for box in self._predict_raw_boxes(tile, imgsz=SLICE_SIZE):
                    x, y, w, h = box.bbox
                    boxes.append(DetectionBox(bbox=[x + left, y + top, w, h], score=box.score))
        return boxes

    def _merge_boxes(self, boxes: list[DetectionBox], width: int, height: int) -> list[list[int]]:
        merged = nms_boxes(boxes, iou_threshold=SLICE_MERGE_IOU)
        clipped: list[list[int]] = []
        for box in merged:
            x, y, w, h = box.bbox
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            clipped.append([x1, y1, x2 - x1, y2 - y1])
        return clipped

    def detect_and_recognize(self, image_path: Path) -> list[dict]:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            image_width, image_height = rgb_image.size
            detections: list[Detection] = []
            merged_boxes = self._merge_boxes(
                self._predict_full_image_boxes(rgb_image) + self._predict_sliced_boxes(rgb_image),
                width=image_width,
                height=image_height,
            )
            for x1, y1, w, h in merged_boxes:
                x2 = x1 + w
                y2 = y1 + h
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
    print(f"CLASSIFY_MIN_COS={CLASSIFY_MIN_COS}")
    print(f"SLICE_SIZE={SLICE_SIZE}")
    print(f"SLICE_OVERLAP={SLICE_OVERLAP}")
    print(f"SLICE_MERGE_IOU={SLICE_MERGE_IOU}")
    print(f"DET_TTA_MODE={DET_TTA_MODE}")

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
