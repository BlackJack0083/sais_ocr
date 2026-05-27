#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageFile, ImageOps
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from ultralytics import YOLO

EDGECRAFTER_ROOT = Path("/mnt/data/hejiakai/external_src/EdgeCrafter-main/ecdetseg")
if EDGECRAFTER_ROOT.exists():
    edgecrafter_root_str = str(EDGECRAFTER_ROOT)
    if edgecrafter_root_str not in sys.path:
        sys.path.append(edgecrafter_root_str)
    from engine.core.yaml_config import YAMLConfig


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


class EdgeCrafterDeployedModel(nn.Module):
    def __init__(self, config_path: Path, checkpoint_path: Path) -> None:
        super().__init__()
        cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path))
        cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        cfg.model.load_state_dict(state)
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        self.eval_spatial_size = tuple(int(v) for v in cfg.yaml_cfg["eval_spatial_size"])
        self.task = str(cfg.yaml_cfg["task"])

    def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


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
    parser.add_argument("--detector-backend", type=str, default="yolo", choices=["yolo", "rfdetr", "edgecrafter"])
    parser.add_argument(
        "--edgecrafter-config",
        type=Path,
        default=Path("/mnt/data/hejiakai/external_src/EdgeCrafter-main/ecdetseg/configs/ecdet/ecdet_m_sais_probe.yml"),
    )
    parser.add_argument("--classifier-weights", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True, help="Raw competition root with XML files.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--det-conf", type=float, default=0.20)
    parser.add_argument("--det-iou", type=float, default=0.50)
    parser.add_argument("--det-imgsz", type=int, default=1280)
    parser.add_argument("--cls-batch-size", type=int, default=128)
    parser.add_argument("--cls-min-prob", type=float, default=0.0)
    parser.add_argument("--cls-min-margin", type=float, default=0.0)
    parser.add_argument("--cls-min-cos", type=float, default=0.0)
    parser.add_argument("--box-expand-ratio", type=float, default=0.02)
    parser.add_argument("--slice-size", type=int, default=0)
    parser.add_argument("--slice-overlap", type=int, default=0)
    parser.add_argument("--slice-merge-iou", type=float, default=0.50)
    parser.add_argument("--det-tta-mode", type=str, default="none")
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


def xywh_to_xyxy(box: list[int]) -> list[int]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


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
    def __init__(
        self,
        detector_backend: str,
        detector_weights: Path,
        edgecrafter_config: Path,
        classifier_weights: Path,
        device: str,
        det_conf: float,
        det_iou: float,
        det_imgsz: int,
        cls_batch_size: int,
        cls_min_prob: float,
        cls_min_margin: float,
        cls_min_cos: float,
        box_expand_ratio: float,
        slice_size: int,
        slice_overlap: int,
        slice_merge_iou: float,
        det_tta_mode: str,
    ) -> None:
        self.detector_backend = detector_backend
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if detector_backend == "yolo":
            self.detector = YOLO(str(detector_weights))
        elif detector_backend == "rfdetr":
            from rfdetr import from_checkpoint as rfdetr_from_checkpoint
            self.detector = rfdetr_from_checkpoint(str(detector_weights))
        elif detector_backend == "edgecrafter":
            if not EDGECRAFTER_ROOT.exists():
                raise FileNotFoundError(f"EdgeCrafter root not found: {EDGECRAFTER_ROOT}")
            self.detector = EdgeCrafterDeployedModel(edgecrafter_config, detector_weights).to(self.device)
            self.detector.eval()
        else:
            raise ValueError(f"Unsupported detector backend: {detector_backend}")
        ckpt = torch.load(classifier_weights, map_location="cpu", weights_only=False)
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
        self.det_conf = det_conf
        self.det_iou = det_iou
        self.det_imgsz = det_imgsz
        self.cls_batch_size = cls_batch_size
        self.cls_min_prob = cls_min_prob
        self.cls_min_margin = cls_min_margin
        self.cls_min_cos = cls_min_cos
        self.box_expand_ratio = box_expand_ratio
        self.slice_size = slice_size
        self.slice_overlap = slice_overlap
        self.slice_merge_iou = slice_merge_iou
        self.det_tta_mode = det_tta_mode
        self.transform = transforms.Compose(
            [
                transforms.Lambda(pad_to_square),
                transforms.Resize((self.classifier_img_size, self.classifier_img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            ]
        )
        self.edgecrafter_transform = transforms.Compose(
            [
                transforms.Resize(self.detector.eval_spatial_size if detector_backend == "edgecrafter" else (self.det_imgsz, self.det_imgsz)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    def _detector_device(self) -> int | str:
        return 0 if self.device.type == "cuda" else "cpu"

    def _predict_raw_boxes(self, image: Image.Image, imgsz: int) -> list[DetectionBox]:
        if self.detector_backend == "yolo":
            prediction = self.detector.predict(
                source=image,
                conf=self.det_conf,
                iou=self.det_iou,
                imgsz=imgsz,
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
                        bbox=xyxy_to_xywh(
                            [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]
                        ),
                        score=float(score),
                    )
                )
            return boxes

        if self.detector_backend == "edgecrafter":
            width, height = image.size
            spatial_size = (imgsz, imgsz) if imgsz > 0 else self.detector.eval_spatial_size
            transform = transforms.Compose(
                [
                    transforms.Resize(spatial_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            tensor = transform(image).unsqueeze(0).to(self.device)
            orig_sizes = torch.tensor([[width, height]], device=self.device)
            with torch.inference_mode():
                outputs = self.detector(tensor, orig_sizes)
            if self.detector.task == "segmentation":
                labels, pred_boxes, pred_scores, _ = outputs
            else:
                labels, pred_boxes, pred_scores = outputs
            keep = pred_scores[0] > self.det_conf
            if keep.sum().item() == 0:
                return []
            boxes = []
            for xyxy, score in zip(pred_boxes[0][keep], pred_scores[0][keep]):
                x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy.tolist()]
                boxes.append(DetectionBox(bbox=xyxy_to_xywh([x1, y1, x2, y2]), score=float(score.item())))
            return nms_boxes(boxes, iou_threshold=self.det_iou)

        detections = self.detector.predict(
            image,
            threshold=self.det_conf,
            shape=(imgsz, imgsz),
            include_source_image=False,
        )
        if len(detections) == 0:
            return []

        boxes: list[DetectionBox] = []
        xyxy_array = detections.xyxy
        conf_array = detections.confidence
        for xyxy, score in zip(xyxy_array, conf_array):
            x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy.tolist()]
            boxes.append(DetectionBox(bbox=xyxy_to_xywh([x1, y1, x2, y2]), score=float(score)))
        return nms_boxes(boxes, iou_threshold=self.det_iou)

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

        planned_views: list[tuple[int, bool]] = [(self.det_imgsz, False)]
        if self.det_tta_mode in {"hflip", "hflip+scale1536"}:
            planned_views.append((self.det_imgsz, True))
        if self.det_tta_mode in {"scale1536", "hflip+scale1536"}:
            planned_views.append((1536, False))
        if self.det_tta_mode == "hflip+scale1536":
            planned_views.append((1536, True))

        seen_views: set[tuple[int, bool]] = set()
        flipped_rgb = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        for imgsz, flipped in planned_views:
            key = (imgsz, flipped)
            if key in seen_views:
                continue
            seen_views.add(key)
            add_view_boxes(flipped_rgb if flipped else rgb, imgsz=imgsz, flipped=flipped)
        return all_boxes

    def _predict_sliced_boxes(self, rgb: Image.Image) -> list[DetectionBox]:
        width, height = rgb.size
        if self.slice_size <= 0:
            return []

        boxes: list[DetectionBox] = []
        x_starts = sliding_starts(width, self.slice_size, self.slice_overlap)
        y_starts = sliding_starts(height, self.slice_size, self.slice_overlap)
        for top in y_starts:
            for left in x_starts:
                right = min(width, left + self.slice_size)
                bottom = min(height, top + self.slice_size)
                tile = rgb.crop((left, top, right, bottom))
                for box in self._predict_raw_boxes(tile, imgsz=self.slice_size):
                    x, y, w, h = box.bbox
                    boxes.append(DetectionBox(bbox=[x + left, y + top, w, h], score=box.score))
        return boxes

    def _merge_boxes(self, boxes: list[DetectionBox], width: int, height: int) -> list[list[int]]:
        merged = nms_boxes(boxes, iou_threshold=self.slice_merge_iou)
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

    def predict(self, image_path: Path) -> list[Item]:
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
                for start in range(0, len(crops), self.cls_batch_size):
                    batch = torch.stack(crops[start : start + self.cls_batch_size]).to(self.device)
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
                            pred_prob < self.cls_min_prob
                            or pred_margin < self.cls_min_margin
                            or pred_cos < self.cls_min_cos
                        ):
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
        detector_backend=args.detector_backend,
        detector_weights=args.detector_weights,
        edgecrafter_config=args.edgecrafter_config,
        classifier_weights=args.classifier_weights,
        device=args.device,
        det_conf=args.det_conf,
        det_iou=args.det_iou,
        det_imgsz=args.det_imgsz,
        cls_batch_size=args.cls_batch_size,
        cls_min_prob=args.cls_min_prob,
        cls_min_margin=args.cls_min_margin,
        cls_min_cos=args.cls_min_cos,
        box_expand_ratio=args.box_expand_ratio,
        slice_size=args.slice_size,
        slice_overlap=args.slice_overlap,
        slice_merge_iou=args.slice_merge_iou,
        det_tta_mode=args.det_tta_mode,
    )

    total_tp = total_fp = total_fn = 0
    total_gt = total_pred = 0
    image_paths = sorted(path for path in args.images_dir.iterdir() if path.is_file())
    started_at = time.perf_counter()

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
        "elapsed_sec": time.perf_counter() - started_at,
        "iou_threshold": args.iou_threshold,
        "num_images": len(image_paths),
        "total_gt_chars": total_gt,
        "total_pred_chars": total_pred,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
