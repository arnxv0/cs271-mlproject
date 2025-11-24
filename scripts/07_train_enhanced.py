#!/usr/bin/env python3
"""
07_train_enhanced.py - Enhanced CNN training with:
- Aggressive data augmentation
- Deeper architecture with residual connections
- Learning rate scheduler
- Weighted loss for class imbalance
- AdamW optimizer with weight decay
- Early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from collections import Counter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.1)

        # Skip connection (downsample if needed)
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Skip connection
        out = self.relu(out)

        return out


class EnhancedFishCNN(nn.Module):
    """Deeper CNN with residual connections (6 blocks)"""

    def __init__(self, num_classes):
        super(EnhancedFishCNN, self).__init__()

        # Initial conv layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual blocks (6 blocks total)
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)    # 56x56
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)   # 28x28
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)  # 14x14
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)  # 7x7

        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def get_data_loaders(data_dir, batch_size=32):
    """Create data loaders with aggressive augmentation"""

    # Aggressive augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # No augmentation for val/test
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(data_dir / 'train', transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / 'val', transform=val_transform)
    test_dataset = datasets.ImageFolder(data_dir / 'test', transform=val_transform)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset.classes, train_dataset.targets


def calculate_class_weights(targets, num_classes, device):
    """Calculate inverse frequency weights for class imbalance"""
    class_counts = Counter(targets)
    total = sum(class_counts.values())

    # Inverse frequency weighting
    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        if i in class_counts:
            weights[i] = total / (num_classes * class_counts[i])
        else:
            weights[i] = 1.0

    # Normalize
    weights = weights / weights.sum() * num_classes

    logger.info("Class weights:")
    for i, w in enumerate(weights):
        count = class_counts.get(i, 0)
        logger.info(f"  Class {i}: count={count}, weight={w:.4f}")

    return weights.to(device)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100. * correct / total


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=10, min_delta=0.001):
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
    # Config
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
    MODEL_DIR = PROJECT_ROOT / 'models'

    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.01  # L2 regularization
    NUM_EPOCHS = 75      # Longer training
    EARLY_STOP_PATIENCE = 15

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading datasets with aggressive augmentation...")
    train_loader, val_loader, test_loader, class_names, train_targets = get_data_loaders(DATA_DIR, BATCH_SIZE)
    num_classes = len(class_names)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {class_names}")

    # Create model
    model = EnhancedFishCNN(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Calculate class weights for imbalanced data
    class_weights = calculate_class_weights(train_targets, num_classes, device)

    # Weighted loss for class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler (reduce on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    # Training loop
    best_val_acc = 0.0
    logger.info("Starting enhanced training...")
    logger.info(f"  - Aggressive augmentation: ON")
    logger.info(f"  - Residual connections: ON")
    logger.info(f"  - Weighted loss: ON")
    logger.info(f"  - AdamW + weight decay: ON")
    logger.info(f"  - Early stopping patience: {EARLY_STOP_PATIENCE}")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler based on validation accuracy
        scheduler.step(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                   f"LR: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
            }, MODEL_DIR / 'best_model_enhanced.pth')
            logger.info(f"  -> Saved new best model (val_acc: {val_acc:.2f}%)")

        # Early stopping check
        early_stopping(val_acc)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Final evaluation on test set
    logger.info("\nEvaluating best model on test set...")
    checkpoint = torch.load(MODEL_DIR / 'best_model_enhanced.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    logger.info(f"Test Accuracy: {test_acc:.2f}%")

    logger.info("\nTraining complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
