#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageFile, ImageOps
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from ultralytics import YOLO


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_MEAN = (0.85233593, 0.85246795, 0.8517555)
IMAGE_STD = (0.31232414, 0.3122127, 0.31273854)


@dataclass
class Item:
    bbox: list[int]
    text: str


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end F1 on the validation images.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--classifier-weights", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True, help="Raw competition root with XML files.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--det-conf", type=float, default=0.20)
    parser.add_argument("--det-iou", type=float, default=0.50)
    parser.add_argument("--det-imgsz", type=int, default=1280)
    parser.add_argument("--cls-batch-size", type=int, default=128)
    parser.add_argument("--cls-min-prob", type=float, default=0.0)
    parser.add_argument("--cls-min-margin", type=float, default=0.0)
    parser.add_argument("--box-expand-ratio", type=float, default=0.02)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


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


def parse_position(raw: str) -> list[int] | None:
    raw = raw.strip()
    if not raw:
        return None
    if ";" in raw:
        points = []
        for item in raw.split(";"):
            item = item.strip()
            if not item:
                continue
            parts = [p.strip() for p in item.split(",") if p.strip()]
            if len(parts) != 2:
                return None
            x, y = map(float, parts)
            points.append((x, y))
        if not points:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) != 4:
            return None
        x1, y1, x2, y2 = map(float, parts)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
    return [int(x1), int(y1), int(x2), int(y2)]


def xyxy_to_xywh(box: list[int]) -> list[int]:
    x1, y1, x2, y2 = box
    return [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]


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


def load_ground_truth(xml_path: Path) -> list[Item]:
    root = ET.fromstring(xml_path.read_text(encoding="utf-16"))
    results: list[Item] = []
    for elem in root.iter("char"):
        text = (elem.text or "").strip()
        if len(text) != 1:
            continue
        bbox = parse_position(elem.attrib.get("position", ""))
        if bbox is None:
            continue
        results.append(Item(bbox=xyxy_to_xywh(bbox), text=text))
    return results


class Recognizer:
    def __init__(
        self,
        detector_weights: Path,
        classifier_weights: Path,
        device: str,
        det_conf: float,
        det_iou: float,
        det_imgsz: int,
        cls_batch_size: int,
        cls_min_prob: float,
        cls_min_margin: float,
        box_expand_ratio: float,
    ) -> None:
        self.detector = YOLO(str(detector_weights))
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(classifier_weights, map_location="cpu", weights_only=False)
        label_map: dict[str, int] = ckpt["label_map"]
        self.idx_to_text = {int(idx): text for text, idx in label_map.items()}
        self.model = EfficientNetArcFace(int(ckpt["args"]["embedding_dim"])).to(self.device)
        self.arcface = ArcMarginProduct(
            in_features=int(ckpt["args"]["embedding_dim"]),
            out_features=len(label_map),
            s=float(ckpt["args"]["arcface_s"]),
            m=float(ckpt["args"]["arcface_m"]),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.arcface.load_state_dict(ckpt["arcface_state_dict"])
        self.model.eval()
        self.arcface.eval()
        self.det_conf = det_conf
        self.det_iou = det_iou
        self.det_imgsz = det_imgsz
        self.cls_batch_size = cls_batch_size
        self.cls_min_prob = cls_min_prob
        self.cls_min_margin = cls_min_margin
        self.box_expand_ratio = box_expand_ratio
        self.transform = transforms.Compose(
            [
                transforms.Lambda(pad_to_square),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            ]
        )

    def _expand_box(self, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> list[int]:
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        expand_x = box_w * self.box_expand_ratio
        expand_y = box_h * self.box_expand_ratio
        left = max(0, int(round(x1 - expand_x)))
        top = max(0, int(round(y1 - expand_y)))
        right = min(width, int(round(x2 + expand_x)))
        bottom = min(height, int(round(y2 + expand_y)))
        return [left, top, max(0, right - left), max(0, bottom - top)]

    def predict(self, image_path: Path) -> list[Item]:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            image_width, image_height = rgb.size
            prediction = self.detector.predict(
                source=str(image_path),
                conf=self.det_conf,
                iou=self.det_iou,
                imgsz=self.det_imgsz,
                device=0 if self.device.type == "cuda" else "cpu",
                verbose=False,
            )[0]
            boxes = prediction.boxes.xyxy.tolist() if prediction.boxes is not None else []

            xywh_boxes: list[list[int]] = []
            crops = []
            for x1, y1, x2, y2 in boxes:
                bbox = self._expand_box(x1, y1, x2, y2, image_width, image_height)
                x1i, y1i, w, h = bbox
                if w <= 0 or h <= 0:
                    continue
                xywh_boxes.append(bbox)
                crops.append(self.transform(rgb.crop((x1i, y1i, x1i + w, y1i + h))))

        texts: list[str] = []
        if crops:
            with torch.inference_mode():
                for start in range(0, len(crops), self.cls_batch_size):
                    batch = torch.stack(crops[start : start + self.cls_batch_size]).to(self.device)
                    logits = self.arcface.infer(self.model(batch))
                    probs = logits.softmax(dim=1)
                    top2_probs, top2_indices = probs.topk(k=min(2, probs.shape[1]), dim=1)
                    pred_indices = top2_indices[:, 0].tolist()
                    pred_probs = top2_probs[:, 0].tolist()
                    if top2_probs.shape[1] > 1:
                        pred_margins = (top2_probs[:, 0] - top2_probs[:, 1]).tolist()
                    else:
                        pred_margins = pred_probs

                    for pred_index, pred_prob, pred_margin in zip(pred_indices, pred_probs, pred_margins):
                        if pred_prob < self.cls_min_prob or pred_margin < self.cls_min_margin:
                            texts.append("")
                            continue
                        texts.append(self.idx_to_text[int(pred_index)])

        return [Item(bbox=bbox, text=text) for bbox, text in zip(xywh_boxes, texts) if text]


def match_counts(preds: list[Item], gts: list[Item], iou_threshold: float) -> tuple[int, int, int]:
    candidates: list[tuple[float, int, int]] = []
    for pred_index, pred in enumerate(preds):
        for gt_index, gt in enumerate(gts):
            if pred.text != gt.text:
                continue
            iou = iou_xywh(pred.bbox, gt.bbox)
            if iou >= iou_threshold:
                candidates.append((iou, pred_index, gt_index))
    candidates.sort(reverse=True)

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    tp = 0
    for _, pred_index, gt_index in candidates:
        if pred_index in matched_preds or gt_index in matched_gts:
            continue
        matched_preds.add(pred_index)
        matched_gts.add(gt_index)
        tp += 1
    fp = len(preds) - tp
    fn = len(gts) - tp
    return tp, fp, fn


def main() -> None:
    args = parse_args()
    recognizer = Recognizer(
        detector_weights=args.detector_weights,
        classifier_weights=args.classifier_weights,
        device=args.device,
        det_conf=args.det_conf,
        det_iou=args.det_iou,
        det_imgsz=args.det_imgsz,
        cls_batch_size=args.cls_batch_size,
        cls_min_prob=args.cls_min_prob,
        cls_min_margin=args.cls_min_margin,
        box_expand_ratio=args.box_expand_ratio,
    )

    total_tp = total_fp = total_fn = 0
    total_gt = total_pred = 0
    image_paths = sorted(path for path in args.images_dir.iterdir() if path.is_file())

    for index, image_path in enumerate(image_paths, start=1):
        xml_path = (args.source_root / image_path.name).with_suffix(".xml")
        if not xml_path.exists():
            resolved = image_path.resolve()
            xml_path = resolved.with_suffix(".xml")
        if not xml_path.exists():
            raise FileNotFoundError(f"Missing XML for {image_path}")

        preds = recognizer.predict(image_path.resolve())
        gts = load_ground_truth(xml_path)
        tp, fp, fn = match_counts(preds, gts, args.iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += len(gts)
        total_pred += len(preds)
        if index == 1 or index % 50 == 0:
            print(f"[{index}/{len(image_paths)}] {image_path.name} tp={tp} fp={fp} fn={fn}")

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    result = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_threshold": args.iou_threshold,
        "num_images": len(image_paths),
        "total_gt_chars": total_gt,
        "total_pred_chars": total_pred,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
