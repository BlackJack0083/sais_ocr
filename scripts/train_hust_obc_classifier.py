#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from PIL import ImageFile
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


IMAGE_MEAN = (0.85233593, 0.85246795, 0.8517555)
IMAGE_STD = (0.31232414, 0.3122127, 0.31273854)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 + ArcFace on HUST-OBC.")
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--arcface-s", type=float, default=30.0)
    parser.add_argument("--arcface-m", type=float, default=0.35)
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
    def __init__(self, json_path: Path, img_size: int, train: bool) -> None:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.samples = [Sample(path=item["path"], label=int(item["label"])) for item in data]
        base = [
            transforms.Resize((img_size, img_size)),
        ]
        if train:
            base.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(10, fill=(255, 255, 255)),
                ]
            )
        base.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            ]
        )
        self.transform = transforms.Compose(base)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            if image.mode != "RGB":
                image = image.convert("RGB")
            else:
                image = image.copy()
        return self.transform(image), sample.label


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


class EfficientNetArcFace(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int) -> None:
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
            logits = arcface(embeddings, labels)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    label_map = json.loads(args.label_map.read_text(encoding="utf-8"))
    num_classes = len(label_map)
    print(f"Loaded label map with {num_classes} classes.", flush=True)

    train_dataset = JsonImageDataset(args.train_json, args.img_size, train=True)
    val_dataset = JsonImageDataset(args.val_json, args.img_size, train=False)
    print(
        f"Loaded datasets: train={len(train_dataset)} samples, val={len(val_dataset)} samples.",
        flush=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    print(
        f"Built dataloaders with batch_size={args.batch_size}, num_workers={args.num_workers}.",
        flush=True,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}.", flush=True)
    model = EfficientNetArcFace(num_classes=num_classes, embedding_dim=args.embedding_dim).to(device)
    arcface = ArcMarginProduct(
        in_features=args.embedding_dim,
        out_features=num_classes,
        s=args.arcface_s,
        m=args.arcface_m,
    ).to(device)
    print("Initialized EfficientNet-B0 backbone and ArcFace head.", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict] = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
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
            logits = arcface(embeddings, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += images.size(0)
            progress.set_postfix(
                loss=f"{running_loss / max(running_total, 1):.4f}",
                acc=f"{running_correct / max(running_total, 1):.4f}",
            )

        scheduler.step()

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_loss, val_acc = evaluate(model, arcface, val_loader, criterion, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(record)

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

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, args.output_dir / "best.pt")

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training finished. Best val acc: {best_val_acc:.4f}")
    print(f"Best checkpoint: {args.output_dir / 'best.pt'}")
    print(f"Last checkpoint: {args.output_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
