# Training Guide - Improving from 55% to 75-85% Accuracy

## Quick Start

Run the optimized training script:
```bash
python scripts/07_train_optimized.py
```

## What Changed? Key Improvements

### 1. Transfer Learning (BIGGEST WIN)
- **Old**: Training CNN from scratch with random weights
- **New**: Using pretrained ResNet50 with ImageNet weights
- **Impact**: +15-25% accuracy improvement
- **Why**: Pretrained models already learned useful features from 1M+ images

### 2. Mixed Precision Training (SPEED BOOST)
- **Old**: Full 32-bit precision
- **New**: Mixed FP16/FP32 precision with `torch.cuda.amp`
- **Impact**: 2-3x faster training, less GPU memory
- **Note**: Automatically disabled on Apple Silicon (MPS) - not yet supported

### 3. Better Data Augmentation
- **Old**: Basic flip + rotation
- **New**:
  - RandomResizedCrop (scale 0.7-1.0)
  - Stronger rotation (30°)
  - More aggressive ColorJitter
  - RandomAffine with shear
  - RandomErasing (cutout)
- **Impact**: Better generalization, reduces overfitting

### 4. Optimized Hyperparameters
| Parameter | Old | New | Impact |
|-----------|-----|-----|--------|
| Batch Size | 32 | 64 | Faster training |
| Epochs | 25 | 40 | Better convergence |
| Num Workers | 4 | 8 | Faster data loading |
| Label Smoothing | 0 | 0.1 | Better generalization |
| LR Scheduler | ReduceLROnPlateau | CosineAnnealingWarmRestarts | Smoother learning |

### 5. Faster Data Loading
- Larger batch size (64 vs 32)
- More workers (8 vs 4)
- `persistent_workers=True` - reuse workers
- Larger batch for validation/test (128)

## Speed Comparison

| Method | Time per Epoch | Total Time (40 epochs) |
|--------|---------------|------------------------|
| Basic CNN | ~60s | ~25 min |
| Optimized (no AMP) | ~40s | ~27 min |
| Optimized (with AMP) | ~15s | ~10 min |

**Note**: Times assume CUDA GPU. CPU will be slower.

## Expected Results

| Model | Expected Accuracy | Training Time |
|-------|------------------|---------------|
| Basic CNN (07_train.py) | 50-60% | 25 min |
| Enhanced CNN (07_train_enhanced.py) | 60-70% | 45 min |
| **Optimized (07_train_optimized.py)** | **75-85%** | **10-15 min** |

## Model Options

Edit `MODEL_NAME` in the script (line 255):

```python
MODEL_NAME = 'resnet50'  # Best balance (recommended)
# MODEL_NAME = 'resnet34'       # Faster, slightly lower accuracy
# MODEL_NAME = 'efficientnet_b0'  # Very efficient
# MODEL_NAME = 'efficientnet_b2'  # Higher accuracy, slower
```

### Model Comparison

| Model | Params | Speed | Expected Accuracy |
|-------|--------|-------|-------------------|
| ResNet34 | 21M | Fast | 72-78% |
| **ResNet50** | 25M | Medium | **75-85%** |
| EfficientNet-B0 | 5M | Very Fast | 73-80% |
| EfficientNet-B2 | 9M | Medium | 76-83% |

## Memory Issues?

If you get out-of-memory errors, reduce batch size:

```python
BATCH_SIZE = 32  # or even 16
```

Or use a smaller model:
```python
MODEL_NAME = 'resnet34'  # Uses less memory
```

## CPU Training?

The script auto-detects your device. On CPU:
- Training will be slower (2-3 hours instead of 10 min)
- Consider using ResNet34 or EfficientNet-B0 for faster training

## Further Improvements (Advanced)

If you want even better results:

1. **Test-Time Augmentation (TTA)**
   - Apply multiple augmentations during inference
   - Average predictions
   - +2-3% accuracy

2. **More Data**
   - Increase target_images in config.yaml
   - More data = better accuracy

3. **Ensemble Models**
   - Train multiple models (ResNet50 + EfficientNet-B2)
   - Average predictions
   - +3-5% accuracy

4. **Fine-tuning Strategy**
   - First train only the classifier (freeze backbone)
   - Then unfreeze and train entire model
   - Better for small datasets

5. **Advanced Augmentation**
   - Use albumentations library
   - Add MixUp or CutMix
   - +2-3% accuracy

## Troubleshooting

**Q: Still getting low accuracy?**
- Check if dataset is properly downloaded
- Verify class distribution: `cat data/metadata/class_distribution.json`
- Some species might have very few samples

**Q: Training is slow?**
- Check if CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Increase NUM_WORKERS if you have more CPU cores
- Reduce batch size if data loading is bottleneck

**Q: Model not improving after epoch 10?**
- This is normal! Pretrained models converge fast
- Let it run - fine improvements happen later
- Early stopping will stop if no improvement

## Summary

The optimized script should give you:
- **75-85% accuracy** (vs 55% before)
- **2-3x faster training** with mixed precision
- **Better generalization** with improved augmentation

Just run: `python scripts/07_train_optimized.py`
