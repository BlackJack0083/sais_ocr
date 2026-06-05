#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ET
from collections import Counter
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


@dataclass
class DetectionBox:
    bbox: list[int]
    score: float


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
    parser = argparse.ArgumentParser(description="Break down end-to-end errors into detector and classifier components.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--classifier-weights", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--det-conf", type=float, default=0.08)
    parser.add_argument("--det-iou", type=float, default=0.40)
    parser.add_argument("--det-imgsz", type=int, default=1280)
    parser.add_argument("--det-tta-mode", type=str, default="scale1536")
    parser.add_argument("--cls-batch-size", type=int, default=128)
    parser.add_argument("--cls-min-prob", type=float, default=0.38)
    parser.add_argument("--cls-min-margin", type=float, default=0.07)
    parser.add_argument("--cls-min-cos", type=float, default=0.0)
    parser.add_argument("--box-expand-ratio", type=float, default=0.0)
    parser.add_argument("--slice-size", type=int, default=0)
    parser.add_argument("--slice-overlap", type=int, default=0)
    parser.add_argument("--slice-merge-iou", type=float, default=0.50)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=30)
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
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.detector = YOLO(str(args.detector_weights))
        ckpt = torch.load(args.classifier_weights, map_location="cpu", weights_only=False)
        label_map: dict[str, int] = ckpt["label_map"]
        checkpoint_args = ckpt.get("args", {})
        self.idx_to_text = {int(idx): text for text, idx in label_map.items()}
        self.model = EfficientNetArcFace(int(checkpoint_args["embedding_dim"])).to(self.device)
        self.arcface = ArcMarginProduct(
            in_features=int(checkpoint_args["embedding_dim"]),
            out_features=len(label_map),
            s=float(checkpoint_args["arcface_s"]),
            m=float(checkpoint_args["arcface_m"]),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.arcface.load_state_dict(ckpt["arcface_state_dict"])
        self.model.eval()
        self.arcface.eval()
        self.classifier_img_size = int(checkpoint_args.get("img_size", 128))
        self.transform = transforms.Compose(
            [
                transforms.Lambda(pad_to_square),
                transforms.Resize((self.classifier_img_size, self.classifier_img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            ]
        )

    def _detector_device(self) -> int | str:
        return 0 if self.device.type == "cuda" else "cpu"

    def _expand_box(self, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> list[int]:
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        expand_x = box_w * self.args.box_expand_ratio
        expand_y = box_h * self.args.box_expand_ratio
        left = max(0, int(round(x1 - expand_x)))
        top = max(0, int(round(y1 - expand_y)))
        right = min(width, int(round(x2 + expand_x)))
        bottom = min(height, int(round(y2 + expand_y)))
        return [left, top, max(0, right - left), max(0, bottom - top)]

    def _predict_raw_boxes(self, image: Image.Image, imgsz: int) -> list[DetectionBox]:
        prediction = self.detector.predict(
            source=image,
            conf=self.args.det_conf,
            iou=self.args.det_iou,
            imgsz=imgsz,
            device=self._detector_device(),
            verbose=False,
        )[0]
        if prediction.boxes is None:
            return []
        coords = prediction.boxes.xyxy.tolist()
        scores = prediction.boxes.conf.tolist()
        return [
            DetectionBox(
                bbox=xyxy_to_xywh([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]),
                score=float(score),
            )
            for (x1, y1, x2, y2), score in zip(coords, scores)
        ]

    def _predict_full_image_boxes(self, rgb: Image.Image) -> list[DetectionBox]:
        image_width, _ = rgb.size
        all_boxes: list[DetectionBox] = []

        def add_view_boxes(view: Image.Image, imgsz: int, flipped: bool) -> None:
            raw_boxes = self._predict_raw_boxes(view, imgsz=imgsz)
            for box in raw_boxes:
                x, y, w, h = box.bbox
                if flipped:
                    x = image_width - (x + w)
                all_boxes.append(DetectionBox(bbox=[x, y, w, h], score=box.score))

        add_view_boxes(rgb, imgsz=self.args.det_imgsz, flipped=False)
        if self.args.det_tta_mode in {"hflip", "hflip+scale1536"}:
            add_view_boxes(rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT), imgsz=self.args.det_imgsz, flipped=True)
        if self.args.det_tta_mode in {"scale1536", "hflip+scale1536"}:
            add_view_boxes(rgb, imgsz=1536, flipped=False)
        if self.args.det_tta_mode == "hflip+scale1536":
            add_view_boxes(rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT), imgsz=1536, flipped=True)
        return all_boxes

    def _predict_sliced_boxes(self, rgb: Image.Image) -> list[DetectionBox]:
        width, height = rgb.size
        if self.args.slice_size <= 0:
            return []
        boxes: list[DetectionBox] = []
        x_starts = sliding_starts(width, self.args.slice_size, self.args.slice_overlap)
        y_starts = sliding_starts(height, self.args.slice_size, self.args.slice_overlap)
        for top in y_starts:
            for left in x_starts:
                right = min(width, left + self.args.slice_size)
                bottom = min(height, top + self.args.slice_size)
                tile = rgb.crop((left, top, right, bottom))
                for box in self._predict_raw_boxes(tile, imgsz=self.args.slice_size):
                    x, y, w, h = box.bbox
                    boxes.append(DetectionBox(bbox=[x + left, y + top, w, h], score=box.score))
        return boxes

    def _merge_boxes(self, boxes: list[DetectionBox], width: int, height: int) -> list[list[int]]:
        merged = nms_boxes(boxes, iou_threshold=self.args.slice_merge_iou)
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

    def predict(self, image_path: Path) -> tuple[list[Item], list[list[int]]]:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            image_width, image_height = rgb.size
            merged_boxes = self._merge_boxes(
                self._predict_full_image_boxes(rgb) + self._predict_sliced_boxes(rgb),
                width=image_width,
                height=image_height,
            )

            xywh_boxes: list[list[int]] = []
            crops = []
            for x, y, w, h in merged_boxes:
                expanded = self._expand_box(x, y, x + w, y + h, image_width, image_height)
                x1i, y1i, ew, eh = expanded
                if ew <= 0 or eh <= 0:
                    continue
                xywh_boxes.append(expanded)
                crops.append(self.transform(rgb.crop((x1i, y1i, x1i + ew, y1i + eh))))

        texts: list[str] = []
        if crops:
            with torch.inference_mode():
                for start in range(0, len(crops), self.args.cls_batch_size):
                    batch = torch.stack(crops[start : start + self.args.cls_batch_size]).to(self.device)
                    logits = self.arcface.infer(self.model(batch))
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
                            pred_prob < self.args.cls_min_prob
                            or pred_margin < self.args.cls_min_margin
                            or pred_cos < self.args.cls_min_cos
                        ):
                            texts.append("")
                            continue
                        texts.append(self.idx_to_text[int(pred_index)])

        preds = [Item(bbox=bbox, text=text) for bbox, text in zip(xywh_boxes, texts) if text]
        return preds, xywh_boxes


def main() -> None:
    args = parse_args()
    recognizer = Recognizer(args)
    image_paths = sorted(path for path in args.images_dir.iterdir() if path.is_file())
    started_at = time.perf_counter()

    summary = Counter()
    per_image: list[dict] = []

    for index, image_path in enumerate(image_paths, start=1):
        xml_path = (args.source_root / image_path.name).with_suffix(".xml")
        if not xml_path.exists():
            xml_path = image_path.resolve().with_suffix(".xml")
        if not xml_path.exists():
            raise FileNotFoundError(f"Missing XML for {image_path}")

        preds, raw_boxes = recognizer.predict(image_path.resolve())
        gts = load_ground_truth(xml_path)

        det_matched_gt = set()
        cls_correct_pred = set()
        gt_with_any_box = set()
        gt_with_correct_text = set()

        for pi, pred in enumerate(preds):
            for gi, gt in enumerate(gts):
                iou = iou_xywh(pred.bbox, gt.bbox)
                if iou >= args.iou_threshold:
                    gt_with_any_box.add(gi)
                    if pred.text == gt.text:
                        cls_correct_pred.add(pi)
                        gt_with_correct_text.add(gi)

        for bi, box in enumerate(raw_boxes):
            for gi, gt in enumerate(gts):
                if iou_xywh(box, gt.bbox) >= args.iou_threshold:
                    det_matched_gt.add(gi)
                    break

        detector_miss = len(gts) - len(det_matched_gt)
        classifier_wrong = len(det_matched_gt - gt_with_correct_text)
        background_fp = len(preds) - len(cls_correct_pred)

        summary["images"] += 1
        summary["gt_chars"] += len(gts)
        summary["pred_chars"] += len(preds)
        summary["detector_matched_gt"] += len(det_matched_gt)
        summary["detector_miss_gt"] += detector_miss
        summary["classifier_recovered_gt"] += len(gt_with_correct_text)
        summary["classifier_wrong_on_detected_gt"] += classifier_wrong
        summary["fp_total"] += background_fp

        per_image.append(
            {
                "image": image_path.name,
                "gt_count": len(gts),
                "pred_count": len(preds),
                "det_raw_boxes": len(raw_boxes),
                "detector_matched_gt": len(det_matched_gt),
                "detector_miss_gt": detector_miss,
                "classifier_correct_gt": len(gt_with_correct_text),
                "classifier_wrong_on_detected_gt": classifier_wrong,
                "background_fp": background_fp,
            }
        )

        if index == 1 or index % 100 == 0:
            print(f"[{index}/{len(image_paths)}] {image_path.name} det_miss={detector_miss} cls_wrong={classifier_wrong} bg_fp={background_fp}")

    hard_detector = sorted(per_image, key=lambda x: (x["detector_miss_gt"], x["gt_count"]), reverse=True)[: args.top_k]
    hard_classifier = sorted(per_image, key=lambda x: (x["classifier_wrong_on_detected_gt"], x["gt_count"]), reverse=True)[: args.top_k]
    hard_fp = sorted(per_image, key=lambda x: (x["background_fp"], x["pred_count"]), reverse=True)[: args.top_k]

    output = {
        "summary": dict(summary),
        "elapsed_sec": time.perf_counter() - started_at,
        "top_detector_miss_images": hard_detector,
        "top_classifier_error_images": hard_classifier,
        "top_background_fp_images": hard_fp,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
