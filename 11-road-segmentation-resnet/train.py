import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import InterpolationMode


@dataclass
class TrainConfig:
    data_root: str
    num_classes: int
    save_dir: str
    batch_size: int = 4
    workers: int = 4
    base_size: int = 1024
    crop_size: Tuple[int, int] = (1024, 512)
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    epochs: int = 40
    max_iters: Optional[int] = None
    poly_power: float = 0.9
    class_weights: Optional[List[float]] = None
    ignore_index: int = 255
    print_freq: int = 20
    amp: bool = True
    resume: Optional[str] = None
    checkpoint: Optional[str] = None
    eval_only: bool = False
    export_onnx: Optional[str] = None
    seed: int = 3407
    clip_grad: float = 0.0
    tta: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def colorize_mask(mask: np.ndarray, num_classes: int) -> Image.Image:
    palette = np.array(
        [
            [0, 0, 0],
            [128, 64, 128],
            [244, 35, 232],
            [70, 70, 70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170, 30],
            [220, 220, 0],
            [107, 142, 35],
            [152, 251, 152],
            [70, 130, 180],
            [220, 20, 60],
            [255, 0, 0],
            [0, 0, 142],
            [0, 0, 70],
            [0, 60, 100],
            [0, 80, 100],
            [0, 0, 230],
            [119, 11, 32],
        ],
        dtype=np.uint8,
    )
    if num_classes + 1 > len(palette):
        repeats = math.ceil((num_classes + 1) / len(palette))
        palette = np.tile(palette, (repeats, 1))
    colored = palette[mask % len(palette)]
    return Image.fromarray(colored.astype(np.uint8))


class SegmentationAugmentation:
    def __init__(
        self,
        base_size: int,
        crop_size: Tuple[int, int],
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        is_train: bool = True,
    ) -> None:
        self.base_size = base_size
        self.crop_h, self.crop_w = crop_size[1], crop_size[0]
        self.is_train = is_train
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Tensor, Tensor]:
        if self.is_train:
            image, mask = self.random_scale(image, mask)
            if random.random() < 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            image, mask = self.random_crop(image, mask)
            image = self.color_jitter(image)
        else:
            image, mask = self.fixed_resize(image, mask)
        image_tensor = transforms.functional.to_tensor(image)
        image_tensor = transforms.functional.normalize(image_tensor, self.mean, self.std)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_tensor, mask_tensor

    def random_scale(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = image.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        image = transforms.functional.resize(image, (oh, ow), InterpolationMode.BILINEAR)
        mask = transforms.functional.resize(mask, (oh, ow), InterpolationMode.NEAREST)
        return image, mask

    def random_crop(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        w, h = image.size
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            image = transforms.functional.pad(image, (0, 0, pad_w, pad_h), fill=0)
            mask = transforms.functional.pad(mask, (0, 0, pad_w, pad_h), fill=255)
        w, h = image.size
        x1 = random.randint(0, w - self.crop_w)
        y1 = random.randint(0, h - self.crop_h)
        image = transforms.functional.crop(image, y1, x1, self.crop_h, self.crop_w)
        mask = transforms.functional.crop(mask, y1, x1, self.crop_h, self.crop_w)
        return image, mask

    def fixed_resize(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        target_h, target_w = self.crop_h, self.crop_w
        image = transforms.functional.resize(image, (target_h, target_w), InterpolationMode.BILINEAR)
        mask = transforms.functional.resize(mask, (target_h, target_w), InterpolationMode.NEAREST)
        return image, mask


class RoadSegDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        transform: SegmentationAugmentation,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if len(self.images) == 0:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        image_path = self.images[idx]
        mask_path = self.mask_dir / image_path.name.replace(image_path.suffix, ".png")
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image, mask = self.transform(image, mask)
        return image, mask


def build_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = SegmentationAugmentation(cfg.base_size, cfg.crop_size, mean, std, True)
    val_transform = SegmentationAugmentation(cfg.base_size, cfg.crop_size, mean, std, False)

    train_set = RoadSegDataset(Path(cfg.data_root) / "train" / "images", Path(cfg.data_root) / "train" / "masks", train_transform)
    val_set = RoadSegDataset(Path(cfg.data_root) / "val" / "images", Path(cfg.data_root) / "val" / "masks", val_transform)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=True)
    return train_loader, val_loader


def build_model(num_classes: int) -> nn.Module:
    model = deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def poly_lr_lambda(current_step: int, max_steps: int, power: float) -> float:
    if max_steps <= 0:
        return 1.0
    return (1 - float(current_step) / float(max_steps)) ** power


def compute_confusion_matrix(pred: Tensor, target: Tensor, num_classes: int, ignore_index: int) -> Tensor:
    mask = target != ignore_index
    target = target[mask].flatten()
    pred = pred[mask].flatten()
    if target.numel() == 0:
        return torch.zeros((num_classes, num_classes), device=pred.device)
    indices = num_classes * target + pred
    matrix = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    return matrix


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, cfg: TrainConfig) -> Tuple[float, List[float]]:
    model.eval()
    confusion = torch.zeros((cfg.num_classes, cfg.num_classes), device=device)
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.ignore_index)
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            logits = outputs["out"]
            loss = criterion(logits, masks)
            total_loss += loss.item() * images.size(0)
            probs = torch.argmax(logits, dim=1)
            confusion += compute_confusion_matrix(probs, masks, cfg.num_classes, cfg.ignore_index)
    total_samples = len(loader.dataset)
    loss_avg = total_loss / total_samples
    iou = torch.diag(confusion) / (confusion.sum(1) + confusion.sum(0) - torch.diag(confusion) + 1e-6)
    miou = iou.mean().item()
    return miou, iou.tolist(), loss_avg


def save_checkpoint(state: Dict, filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(model: nn.Module, optimizer: Optional[torch.optim.Optimizer], scaler: Optional[GradScaler], path: str) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], Optional[GradScaler], int, float]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    start_epoch = checkpoint.get("epoch", 0)
    best_miou = checkpoint.get("best_miou", 0.0)
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return model, optimizer, scaler, start_epoch, best_miou


def train(cfg: TrainConfig) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg.num_classes).to(device)

    class_weights = torch.tensor(cfg.class_weights, device=device) if cfg.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=cfg.ignore_index)
    optimizer = SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    total_steps = cfg.max_iters if cfg.max_iters else cfg.epochs * len(train_loader)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: poly_lr_lambda(step, total_steps, cfg.poly_power))
    scaler = GradScaler(enabled=cfg.amp)

    start_epoch = 0
    best_miou = 0.0
    if cfg.resume:
        model, optimizer, scaler, start_epoch, best_miou = load_checkpoint(model, optimizer, scaler, cfg.resume)
        print(f"Resumed from {cfg.resume} (epoch {start_epoch}, best mIoU {best_miou:.4f})")

    best_path = Path(cfg.save_dir) / "best.pt"
    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for step, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.amp):
                outputs = model(images)
                logits = outputs["out"]
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            if cfg.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
            global_step += 1

            if (step + 1) % cfg.print_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch+1}/{cfg.epochs}] Step [{step+1}/{len(train_loader)}] Loss: {loss.item():.4f} LR: {current_lr:.6f}")

            if cfg.max_iters and global_step >= cfg.max_iters:
                break
        avg_epoch_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch+1} completed. Avg loss: {avg_epoch_loss:.4f}")

        miou, class_iou, val_loss = evaluate(model, val_loader, device, cfg)
        print(f"Validation mIoU: {miou:.4f}, loss: {val_loss:.4f}")
        print(f"Per-class IoU: {class_iou}")

        checkpoint_path = Path(cfg.save_dir) / "last.pt"
        save_checkpoint(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "best_miou": best_miou,
            },
            checkpoint_path,
        )

        if miou > best_miou:
            best_miou = miou
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch + 1,
                    "best_miou": best_miou,
                },
                best_path,
            )
            print(f"New best mIoU {best_miou:.4f}. Checkpoint saved to {best_path}")

        if cfg.max_iters and global_step >= cfg.max_iters:
            print("Reached max iterations, stopping training.")
            break

    return best_path if best_path.exists() else Path(cfg.save_dir) / "last.pt"


def export_onnx(model: nn.Module, cfg: TrainConfig, device: torch.device) -> None:
    model.eval()
    dummy_input = torch.randn(1, 3, cfg.crop_size[1], cfg.crop_size[0], device=device)
    torch.onnx.export(
        model,
        dummy_input,
        cfg.export_onnx,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Exported ONNX to {cfg.export_onnx}")


def run_inference(model: nn.Module, image_path: str, cfg: TrainConfig, device: torch.device) -> None:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = SegmentationAugmentation(cfg.base_size, cfg.crop_size, mean, std, is_train=False)
    image = Image.open(image_path).convert("RGB")
    mask_dummy = Image.new("L", image.size, 0)
    tensor, _ = transform(image, mask_dummy)
    tensor = tensor.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(tensor)["out"]
        if cfg.tta:
            flipped = torch.flip(tensor, dims=[3])
            logits_flip = model(flipped)["out"]
            logits = 0.5 * logits + 0.5 * torch.flip(logits_flip, dims=[3])
        preds = torch.argmax(F.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False), dim=1)[0]
    mask_np = preds.cpu().numpy().astype(np.uint8)
    colored = colorize_mask(mask_np, cfg.num_classes)
    save_path = Path(cfg.save_dir) / "inference.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    colored.save(save_path)
    print(f"Saved inference mask to {save_path}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Road segmentation training (ResNet backbone)")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory of dataset")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of segmentation classes")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save checkpoints and logs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--crop-size", type=int, nargs=2, default=(1024, 512))
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--class-weights", type=float, nargs="*", default=None)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--export-onnx", type=str, default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--clip-grad", type=float, default=0.0)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--inference", type=str, default=None, help="Run inference on a single image")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        num_classes=args.num_classes,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        base_size=args.base_size,
        crop_size=(args.crop_size[0], args.crop_size[1]),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        max_iters=args.max_iters,
        poly_power=args.poly_power,
        class_weights=args.class_weights,
        ignore_index=args.ignore_index,
        print_freq=args.print_freq,
        amp=args.amp,
        resume=args.resume,
        checkpoint=args.checkpoint,
        eval_only=args.eval_only,
        export_onnx=args.export_onnx,
        seed=args.seed,
        clip_grad=args.clip_grad,
        tta=args.tta,
    )
    return cfg, args.inference


def main() -> None:
    cfg, inference_path = parse_args()
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(Path(cfg.save_dir) / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.num_classes).to(device)

    if cfg.checkpoint:
        state = torch.load(cfg.checkpoint, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"Loaded checkpoint from {cfg.checkpoint}")

    if cfg.export_onnx:
        export_onnx(model, cfg, device)
        return

    if cfg.eval_only:
        train_loader, val_loader = build_dataloaders(cfg)
        miou, class_iou, val_loss = evaluate(model, val_loader, device, cfg)
        print(f"Eval-only mIoU: {miou:.4f}, loss: {val_loss:.4f}")
        print(f"Per-class IoU: {class_iou}")
        return

    best_ckpt = train(cfg)

    if inference_path:
        if best_ckpt and best_ckpt.exists():
            state = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(state["model"] if "model" in state else state)
            print(f"Loaded best checkpoint from {best_ckpt} for inference")
        run_inference(model, inference_path, cfg, device)


if __name__ == "__main__":
    main()
