#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageFile, ImageOps, ImageStat
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from transformers import Dinov2Model


IMAGE_MEAN = (0.85233593, 0.85246795, 0.8517555)
IMAGE_STD = (0.31232414, 0.3122127, 0.31273854)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Unicode character classifier with a selectable backbone + ArcFace.")
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["efficientnet_b0", "dinov2_small"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--arcface-s", type=float, default=30.0)
    parser.add_argument("--arcface-m", type=float, default=0.35)
    parser.add_argument("--init-backbone-from", type=Path, default=None)
    parser.add_argument("--class-balanced-sampler", type=str, default="none", choices=["none", "inv_sqrt"])
    parser.add_argument("--loss", type=str, default="cross_entropy", choices=["cross_entropy", "balanced_softmax"])
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--polarity-normalization",
        type=str,
        default="auto",
        choices=["off", "auto"],
        help="Normalize crop polarity so dark-background light-foreground crops are inverted to a consistent style.",
    )
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume full training state from a previous checkpoint.")
    parser.add_argument(
        "--reset-best-on-resume",
        action="store_true",
        help="Ignore historical best validation accuracy from the resumed checkpoint and re-select best.pt under the current validation setup.",
    )
    parser.add_argument(
        "--resume-optimizer-state",
        action="store_true",
        help="Also restore optimizer and scheduler state. Leave off when extending a finished run whose LR already decayed to zero.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Sample:
    path: str
    label: int


class JsonImageDataset(Dataset):
    def __init__(
        self,
        json_path: Path,
        img_size: int,
        train: bool,
        mean: Sequence[float],
        std: Sequence[float],
        polarity_normalization: str,
    ) -> None:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.samples = [Sample(path=item["path"], label=int(item["label"])) for item in data]
        self.raw_items = data
        self.path_exists = [bool(sample.path) and Path(sample.path).exists() for sample in self.samples]
        base = [
            transforms.Lambda(lambda image: normalize_image_polarity(image, mode=polarity_normalization)),
            transforms.Lambda(lambda image: pad_to_square(image)),
        ]
        if train:
            base.extend(
                [
                    transforms.RandomRotation(5, fill=(255, 255, 255)),
                ]
            )
        base.extend(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.transform = transforms.Compose(base)
        self.class_counts = Counter(sample.label for sample in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        item = self.raw_items[index]
        if self.path_exists[index]:
            with Image.open(sample.path) as image:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                else:
                    image = image.copy()
        elif "source_image_path" in item and "bbox" in item:
            x1, y1, x2, y2 = [int(v) for v in item["bbox"]]
            with Image.open(item["source_image_path"]) as image:
                image = image.convert("RGB").crop((x1, y1, x2, y2))
        else:
            with Image.open(sample.path) as image:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                else:
                    image = image.copy()
        return self.transform(image), sample.label


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


def estimate_border_mean(gray: Image.Image) -> float:
    width, height = gray.size
    if width < 4 or height < 4:
        return float(ImageStat.Stat(gray).mean[0])
    border = max(1, min(width, height) // 12)
    strips = [
        gray.crop((0, 0, width, border)),
        gray.crop((0, height - border, width, height)),
        gray.crop((0, border, border, height - border)),
        gray.crop((width - border, border, width, height - border)),
    ]
    total_weight = 0
    total_value = 0.0
    for strip in strips:
        stat = ImageStat.Stat(strip)
        weight = strip.size[0] * strip.size[1]
        total_weight += weight
        total_value += float(stat.mean[0]) * weight
    if total_weight == 0:
        return float(ImageStat.Stat(gray).mean[0])
    return total_value / total_weight


def normalize_image_polarity(image: Image.Image, mode: str) -> Image.Image:
    if mode == "off":
        return image
    rgb = image.convert("RGB") if image.mode != "RGB" else image.copy()
    gray = rgb.convert("L")
    border_mean = estimate_border_mean(gray)
    overall_mean = float(ImageStat.Stat(gray).mean[0])
    # Dark borders with a brighter interior usually mean black background + light glyph.
    if border_mean + 4.0 < overall_mean:
        return ImageOps.invert(rgb)
    return rgb


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.s

    def infer(self, embeddings: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        return cosine * self.s


class EfficientNetArcFace(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        backbone = models.efficientnet_b0(weights=weights)
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


class Dinov2ArcFace(nn.Module):
    def __init__(self, embedding_dim: int, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        backbone = Dinov2Model.from_pretrained("facebook/dinov2-small")
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable()
        self.backbone = backbone
        hidden_size = int(backbone.config.hidden_size)
        self.embedding = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)
        cls_token = outputs.last_hidden_state[:, 0]
        embeddings = self.embedding(cls_token)
        return F.normalize(embeddings)


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts: list[int]) -> None:
        super().__init__()
        counts = torch.tensor(class_counts, dtype=torch.float32)
        counts = torch.clamp(counts, min=1.0)
        self.register_buffer("log_class_counts", counts.log())

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + self.log_class_counts.to(logits.device)
        return F.cross_entropy(adjusted_logits, labels)


def load_backbone_init(model: nn.Module, init_path: Path | None) -> None:
    if init_path is None:
        return
    checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        return
    state = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Initialized backbone from {init_path}")
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")


def load_resume_checkpoint(
    model: nn.Module,
    arcface: ArcMarginProduct,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    resume_path: Path | None,
    resume_optimizer_state: bool,
) -> tuple[int, float, list[dict]]:
    if resume_path is None:
        return 0, 0.0, []

    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    arcface.load_state_dict(checkpoint["arcface_state_dict"])
    if resume_optimizer_state:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    history = list(checkpoint.get("history", []))
    start_epoch = int(checkpoint.get("epoch", 0))
    best_val_acc = max((float(item.get("val_acc", 0.0)) for item in history), default=0.0)
    print(f"Resumed full state from {resume_path}")
    print(f"Resume epoch: {start_epoch}")
    print(f"Resume best val acc: {best_val_acc:.6f}")
    print(f"Resume optimizer state: {resume_optimizer_state}")
    return start_epoch, best_val_acc, history


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.backbone == "dinov2_small":
        return Dinov2ArcFace(
            embedding_dim=args.embedding_dim,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    return EfficientNetArcFace(embedding_dim=args.embedding_dim)


def get_dataset_stats(backbone: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if backbone == "dinov2_small":
        return IMAGENET_MEAN, IMAGENET_STD
    return IMAGE_MEAN, IMAGE_STD


def evaluate(
    model: nn.Module,
    arcface: ArcMarginProduct,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            embeddings = model(images)
            train_logits = arcface(embeddings, labels)
            infer_logits = arcface.infer(embeddings)
            loss = criterion(train_logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (infer_logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    label_map = json.loads(args.label_map.read_text(encoding="utf-8"))
    num_classes = len(label_map)
    print(f"Loaded label map with {num_classes} classes.", flush=True)

    mean, std = get_dataset_stats(args.backbone)
    train_dataset = JsonImageDataset(
        args.train_json,
        args.img_size,
        train=True,
        mean=mean,
        std=std,
        polarity_normalization=args.polarity_normalization,
    )
    val_dataset = JsonImageDataset(
        args.val_json,
        args.img_size,
        train=False,
        mean=mean,
        std=std,
        polarity_normalization=args.polarity_normalization,
    )
    print(f"Loaded datasets: train={len(train_dataset)} val={len(val_dataset)}", flush=True)

    sampler = None
    shuffle = True
    if args.class_balanced_sampler == "inv_sqrt":
        sample_weights = [1.0 / math.sqrt(max(train_dataset.class_counts[sample.label], 1)) for sample in train_dataset.samples]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    has_val = len(val_dataset) > 0
    val_loader = None
    if has_val:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    model = build_model(args)
    arcface = ArcMarginProduct(
        in_features=args.embedding_dim,
        out_features=num_classes,
        s=args.arcface_s,
        m=args.arcface_m,
    )

    if args.resume_from is None:
        load_backbone_init(model, args.init_backbone_from)

    if args.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False

    model = model.to(device)
    arcface = arcface.to(device)

    class_counts = [train_dataset.class_counts.get(label, 0) for label in range(num_classes)]
    if args.loss == "balanced_softmax":
        criterion = BalancedSoftmaxLoss(class_counts=class_counts)
    else:
        criterion = nn.CrossEntropyLoss()
    backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.lr
    backbone_params = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    head_params = [
        parameter
        for parameter in list(model.embedding.parameters()) + list(arcface.parameters())
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch, best_val_acc, history = load_resume_checkpoint(
        model=model,
        arcface=arcface,
        optimizer=optimizer,
        scheduler=scheduler,
        resume_path=args.resume_from,
        resume_optimizer_state=args.resume_optimizer_state,
    )
    if not has_val:
        best_val_acc = float("-inf")
    if args.reset_best_on_resume:
        best_val_acc = 0.0
        print("Reset best validation accuracy after resume.", flush=True)
    if args.epochs <= start_epoch:
        raise ValueError(f"--epochs ({args.epochs}) must be greater than resume epoch ({start_epoch}).")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            embeddings = model(images)
            train_logits = arcface(embeddings, labels)
            infer_logits = arcface.infer(embeddings)
            loss = criterion(train_logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (infer_logits.argmax(dim=1) == labels).sum().item()
            running_total += images.size(0)
            progress.set_postfix(
                loss=f"{running_loss / max(running_total, 1):.4f}",
                acc=f"{running_correct / max(running_total, 1):.4f}",
            )

        scheduler.step()

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        if has_val and val_loader is not None:
            val_loss, val_acc = evaluate(model, arcface, val_loader, criterion, device)
        else:
            val_loss, val_acc = 0.0, 0.0
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        if not has_val:
            record["note"] = "fulltrain_no_val"
        history.append(record)
        print(record, flush=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "arcface_state_dict": arcface.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "label_map": label_map,
            "args": vars(args),
            "history": history,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if (not has_val) or (val_acc >= best_val_acc):
            best_val_acc = val_acc
            torch.save(checkpoint, args.output_dir / "best.pt")

    (args.output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    if has_val:
        print(f"Training finished. Best val acc: {best_val_acc:.4f}")
    else:
        print("Training finished. Fulltrain run saved last.pt and rolling best.pt without validation.")


if __name__ == "__main__":
    main()
