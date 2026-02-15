# HanziTiny: Lightweight Chinese Character Recognition

A highly efficient, lightweight Convolutional Neural Network (CNN) designed for recognizing handwritten Chinese characters. 

![Demo](https://via.placeholder.com/600x400?text=HanziTiny+Demo+Preview)

## 📌 Project Status
**Current Version**: `v1.0-Tiny`
**Target**: 631 Common Chinese Characters (HWDB1.1 subset)(625 classes as 6 missing datasets)
**Accuracy**: > 94.27% on validation set.

**⚠️ Note**: This repository contains several experimental models. **`HanziTiny` (located in `model/hanzi_tiny.py`) is the ONLY successful and recommended model.** 
- `LeNet`, `ModernNet`, and others found in `legacy/` or `model/` are archiving failed experiments and should be ignored for production use.

## 🌟 Features
- **Ultra-Lightweight**: Uses Depthwise Separable Convolutions (DSConv) + Squeeze-and-Excitation (SE) blocks.
- **Robust Inference**: Includes advanced preprocessing (Auto-Crop, Simulated Blur) to handle the domain gap between digital drawing and training data (scans).
- **Real-time GUI**: `gui_hanzi_tiny.py` provides a drawing board for instant testing.

## 📂 Project Structure
```
.
├── checkpoints/       # Trained model weights (.pth) and class mappings
├── model/             # Model definitions (HanziTiny is the main one)
├── train/             # Training scripts
├── tests/             # Diagnostic and verification scripts
├── utils/             # Helper scripts (data extraction, etc.)
├── legacy/            # Old/Failed experiments and demos
└── gui_hanzi_tiny.py  # Main Application Entry Point
```

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11.2
- PyTorch
- Torchvision
- Pillow (PIL)
- Tkinter (usually built-in with Python)

```bash
pip install torch torchvision pillow
```

### 2. Run the Demo
Launch the handwriting recognition GUI:
```bash
python gui_hanzi_tiny.py
```
Draw a character on the canvas, and the model will predict it in real-time (after releasing the mouse).

### 3. Training (Optional)
If you want to retrain the model:
1. Download **HWDB1.1** dataset and place the `train` data in `HWDB1.1/subset_631` (organized by character folders).
2. Run the training script:
```bash
python train/train_hanzi_tiny.py
```

## 🛠 Model Architecture
`HanziTiny` is inspired by MobileNetV3 but simplified for single-channel character inputs:
- **Steam**: Standard Conv3x3 to reduce resolution.
- **Body**: Stacked `DSConv` blocks (Depthwise 3x3 + Pointwise 1x1 + SE Attention).
- **Head**: Global Average Pooling + Linear Classifier.
- **Parameters**: ~200k (vs LeNet's 500k+).

## 📄 License
This project is open-source.
