#!/usr/bin/env python3
"""
07_train_optimized.py - Optimized training with:
- Transfer learning (pretrained ResNet50/EfficientNet)
- Mixed precision training (2-3x faster)
- Better augmentation and regularization
- Optimized hyperparameters

Expected: 75-85% accuracy (vs 55% baseline)
Training time: 2-3x faster with mixed precision
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from collections import Counter
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_model(model_name='resnet50', num_classes=30, pretrained=True):
    """
    Create a pretrained model with custom classifier

    Args:
        model_name: 'resnet50', 'resnet34', 'efficientnet_b0', 'efficientnet_b2'
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
    """
    logger.info(f"Creating {model_name} (pretrained={pretrained})...")

    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, num_classes)
        )
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, num_classes)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def get_data_loaders(data_dir, batch_size=64, img_size=224, num_workers=8):
    """
    Create data loaders with optimized augmentation

    Args:
        batch_size: Larger batch = faster training (if GPU memory allows)
        img_size: 224 for ResNet, EfficientNet
        num_workers: More workers = faster data loading
    """
    # Strong augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), shear=10),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])

    # No augmentation for val/test
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets - use flattened structure if available, otherwise use nested
    train_dir = data_dir / 'train_flat' if (data_dir / 'train_flat').exists() else data_dir / 'train'
    val_dir = data_dir / 'val_flat' if (data_dir / 'val_flat').exists() else data_dir / 'val'
    test_dir = data_dir / 'test_flat' if (data_dir / 'test_flat').exists() else data_dir / 'test'

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

    # Create loaders with optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for inference
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset.classes, train_dataset.targets


def calculate_class_weights(targets, num_classes, device):
    """Calculate class weights for imbalanced datasets"""
    class_counts = Counter(targets)
    total = sum(class_counts.values())

    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        if i in class_counts:
            weights[i] = total / (num_classes * class_counts[i])
        else:
            weights[i] = 1.0

    # Normalize weights
    weights = weights / weights.sum() * num_classes

    logger.info(f"Class distribution (first 10):")
    for i in range(min(10, num_classes)):
        count = class_counts.get(i, 0)
        logger.info(f"  Class {i}: {count} samples, weight={weights[i]:.3f}")

    return weights.to(device)


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp=True):
    """Train for one epoch with mixed precision"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision training
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_time = time.time() - start_time
    return running_loss / len(train_loader), 100. * correct / total, epoch_time


def validate(model, val_loader, criterion, device, use_amp=True):
    """Validate with mixed precision"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100. * correct / total


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0


def main():
    # ===========================
    # Configuration
    # ===========================
    logger.info("\n" + "="*70)
    logger.info("🚀 OPTIMIZED FISH CLASSIFICATION TRAINING")
    logger.info("="*70)

    logger.info("\n[STEP 1/6] Loading configuration...")
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
    MODEL_DIR = PROJECT_ROOT / 'models'
    MODEL_DIR.mkdir(exist_ok=True)

    # Hyperparameters (TUNED for better accuracy)
    MODEL_NAME = 'resnet50'  # Options: resnet50, resnet34, efficientnet_b0, efficientnet_b2
    BATCH_SIZE = 32          # Larger batch = faster (reduce to 16 if OOM on MPS)
    LEARNING_RATE = 0.001    # Good starting point for pretrained models
    WEIGHT_DECAY = 0.005     # L2 regularization
    NUM_EPOCHS = 40          # Enough for convergence with pretrained models
    NUM_WORKERS = 0          # Set to 0 for MPS (Apple Silicon), 4-8 for CUDA
    USE_AMP = True           # Mixed precision = 2-3x faster on GPU (auto-disabled on MPS)
    EARLY_STOP_PATIENCE = 12
    LABEL_SMOOTHING = 0.1    # Helps generalization

    logger.info("✓ Configuration loaded")
    logger.info(f"  Model: {MODEL_NAME}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Max epochs: {NUM_EPOCHS}")
    logger.info(f"  Learning rate: {LEARNING_RATE}")

    # Device setup
    logger.info("\n[STEP 2/6] Setting up compute device...")
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"✓ Using device: {device}")

    # Disable AMP for MPS (Apple Silicon) - not yet supported
    if device.type == 'mps':
        USE_AMP = False
        logger.info("  Note: Mixed precision disabled (MPS doesn't support AMP yet)")
    elif device.type == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  Mixed precision: {'Enabled' if USE_AMP else 'Disabled'}")
    else:
        USE_AMP = False
        logger.info("  Warning: Training on CPU will be slower")

    # ===========================
    # Load Data
    # ===========================
    logger.info("\n[STEP 3/6] Loading and preparing dataset...")
    logger.info("  Checking for data at: {}".format(DATA_DIR))

    if not DATA_DIR.exists():
        logger.error(f"  ✗ Dataset not found at {DATA_DIR}")
        logger.error("  Please run: ./build_dataset.sh first")
        return

    logger.info("  Loading images with augmentation...")
    train_loader, val_loader, test_loader, class_names, train_targets = get_data_loaders(
        DATA_DIR, BATCH_SIZE, num_workers=NUM_WORKERS
    )
    num_classes = len(class_names)
    logger.info(f"✓ Dataset loaded successfully")
    logger.info(f"  Classes: {num_classes} fish species")
    logger.info(f"  Training samples: {len(train_loader.dataset)}")
    logger.info(f"  Validation samples: {len(val_loader.dataset)}")
    logger.info(f"  Test samples: {len(test_loader.dataset)}")
    logger.info(f"  Batches per epoch: {len(train_loader)}")

    # ===========================
    # Create Model
    # ===========================
    logger.info(f"\n[STEP 4/6] Building {MODEL_NAME} model...")
    logger.info("  Loading pretrained ImageNet weights...")
    model = get_model(MODEL_NAME, num_classes=num_classes, pretrained=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✓ Model built and moved to {device}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

    # ===========================
    # Setup Training
    # ===========================
    logger.info("\n[STEP 5/6] Setting up training components...")

    # Class weights for imbalanced data
    logger.info("  Calculating class weights for imbalanced data...")
    class_weights = calculate_class_weights(train_targets, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    logger.info(f"  ✓ Loss function: CrossEntropyLoss (weighted + label smoothing)")

    # Optimizer: AdamW with weight decay
    logger.info("  Setting up AdamW optimizer...")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    logger.info(f"  ✓ Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")

    # Learning rate scheduler
    logger.info("  Configuring learning rate scheduler...")
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    logger.info("  ✓ Scheduler: CosineAnnealingWarmRestarts")

    # Mixed precision scaler
    if USE_AMP:
        logger.info("  Setting up mixed precision training...")
        scaler = GradScaler()
        logger.info("  ✓ Mixed precision: Enabled (FP16/FP32)")
    else:
        scaler = None
        logger.info("  Mixed precision: Disabled")

    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)
    logger.info(f"  ✓ Early stopping: Enabled (patience={EARLY_STOP_PATIENCE} epochs)")

    # ===========================
    # Training Loop
    # ===========================
    logger.info("\n" + "="*70)
    logger.info("[STEP 6/6] Starting training loop...")
    logger.info("="*70)
    logger.info("\nTraining Configuration:")
    logger.info(f"  • Model: {MODEL_NAME} (pretrained on ImageNet)")
    logger.info(f"  • Batch size: {BATCH_SIZE}")
    logger.info(f"  • Learning rate: {LEARNING_RATE}")
    logger.info(f"  • Mixed precision: {USE_AMP}")
    logger.info(f"  • Label smoothing: {LABEL_SMOOTHING}")
    logger.info(f"  • Early stopping patience: {EARLY_STOP_PATIENCE}")
    logger.info(f"  • Device: {device}")

    logger.info("\n" + "-"*70)
    logger.info("Beginning epoch training...")
    logger.info("-"*70 + "\n")

    best_val_acc = 0.0
    total_training_time = 0

    for epoch in range(NUM_EPOCHS):
        logger.info(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Training on {len(train_loader.dataset)} samples...")

        train_loss, train_acc, epoch_time = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, USE_AMP
        )
        total_training_time += epoch_time

        logger.info(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Validating on {len(val_loader.dataset)} samples...")
        val_loss, val_acc = validate(model, val_loader, criterion, device, USE_AMP)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Calculate estimated time remaining
        avg_time_per_epoch = total_training_time / (epoch + 1)
        remaining_epochs = NUM_EPOCHS - (epoch + 1)
        eta_minutes = (avg_time_per_epoch * remaining_epochs) / 60

        logger.info(
            f"[Epoch {epoch+1:2d}/{NUM_EPOCHS}] "
            f"Train: {train_acc:5.2f}% (loss={train_loss:.4f}) | "
            f"Val: {val_acc:5.2f}% (loss={val_loss:.4f}) | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s | "
            f"ETA: {eta_minutes:.1f}m"
        )

        # Save best model
        if val_acc > best_val_acc:
            improvement = val_acc - best_val_acc
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'model_name': MODEL_NAME,
            }, MODEL_DIR / 'best_model_optimized.pth')
            if improvement > 0:
                logger.info(f"  🎯 New best model! Val: {val_acc:.2f}% (+{improvement:.2f}%) - Saved to models/")
            else:
                logger.info(f"  💾 Model saved (Val: {val_acc:.2f}%)")

        # Early stopping
        early_stopping(val_acc)
        if early_stopping.early_stop:
            logger.info(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            logger.info(f"   No improvement for {EARLY_STOP_PATIENCE} consecutive epochs")
            break
        elif early_stopping.counter > 0:
            logger.info(f"  ⏳ No improvement for {early_stopping.counter}/{EARLY_STOP_PATIENCE} epochs")

        logger.info("")  # Blank line between epochs

    # ===========================
    # Final Evaluation
    # ===========================
    logger.info("\n" + "="*70)
    logger.info("📊 FINAL EVALUATION ON TEST SET")
    logger.info("="*70)

    logger.info("\nLoading best model from checkpoint...")
    checkpoint = torch.load(MODEL_DIR / 'best_model_optimized.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✓ Loaded model from epoch {checkpoint['epoch'] + 1}")

    logger.info(f"\nEvaluating on {len(test_loader.dataset)} test images...")
    test_loss, test_acc = validate(model, test_loader, criterion, device, USE_AMP)

    logger.info("\n" + "="*70)
    logger.info("🎉 TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"\n📈 Results:")
    logger.info(f"  • Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"  • Final test accuracy: {test_acc:.2f}%")
    logger.info(f"  • Total training time: {total_training_time/60:.1f} minutes")
    logger.info(f"  • Epochs completed: {epoch + 1}/{NUM_EPOCHS}")
    logger.info(f"\n💾 Model saved to: {MODEL_DIR / 'best_model_optimized.pth'}")
    logger.info("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
