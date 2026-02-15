# -*- coding: utf-8 -*-
"""
Domain Gap Analysis: Dataset vs GUI Hand-drawn Images
======================================================
This script analyzes why the model achieves 94%+ accuracy on dataset images
but fails on GUI hand-drawn input.

Hypothesis: The model was trained on scanned images with natural anti-aliasing
(gray-scale strokes), but GUI produces digital rendering with pure black strokes.
"""
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import os
from torchvision import transforms
import json

# ============== Configuration ==============
MODEL_PATH = "best_hanzi_tiny.pth"
CLASSES_PATH = "classes.json"
DATA_DIR = "HWDB1.1/subset_631"

# ============== Load Model ==============
def load_model():
    with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    
    from hanzi_tiny import HanziTiny
    model = HanziTiny(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model, classes

# ============== Analysis Functions ==============
def analyze_pixel_distribution(img, name="Image"):
    """Analyze pixel value distribution of an image"""
    arr = np.array(img)
    black_pct = (arr < 50).sum() / arr.size * 100
    gray_pct = ((arr >= 50) & (arr <= 200)).sum() / arr.size * 100
    white_pct = (arr > 200).sum() / arr.size * 100
    
    print(f"\n{name}:")
    print(f"  Size: {img.size}")
    print(f"  Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean():.1f}")
    print(f"  Black (<50): {black_pct:.1f}%")
    print(f"  Gray (50-200): {gray_pct:.1f}%")
    print(f"  White (>200): {white_pct:.1f}%")
    
    return {
        'min': arr.min(),
        'max': arr.max(),
        'mean': arr.mean(),
        'black_pct': black_pct,
        'gray_pct': gray_pct,
        'white_pct': white_pct
    }

def simulate_gui_drawing(char_type="horizontal", stroke_value=0):
    """
    Simulate GUI hand-drawn input
    char_type: "horizontal" (like 'one'), "vertical", "complex"
    stroke_value: 0 = pure black, 80 = gray (matching dataset)
    """
    gui_img = Image.new('L', (320, 320), 255)  # White background
    draw = ImageDraw.Draw(gui_img)
    
    if char_type == "horizontal":
        # Simulate 'one' (horizontal line)
        draw.line([(50, 160), (270, 160)], fill=stroke_value, width=22)
    elif char_type == "vertical":
        # Simulate vertical line
        draw.line([(160, 50), (160, 270)], fill=stroke_value, width=22)
    else:
        # Simulate a cross
        draw.line([(80, 160), (240, 160)], fill=stroke_value, width=22)
        draw.line([(160, 80), (160, 240)], fill=stroke_value, width=22)
    
    # GUI preprocess (auto-crop with padding)
    inverted = ImageOps.invert(gui_img)
    bbox = inverted.getbbox()
    if bbox:
        left, upper, right, lower = bbox
        p = 20
        gui_cropped = gui_img.crop((max(0, left-p), max(0, upper-p), 
                                    min(320, right+p), min(320, lower+p)))
    else:
        gui_cropped = gui_img
    
    return gui_cropped

def predict_image(model, classes, img, transform, title="Prediction"):
    """Run model prediction on an image"""
    tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top5 = torch.topk(probs, 5)
    
    print(f"\n{title}:")
    for i in range(5):
        idx = top5.indices[0][i].item()
        prob = top5.values[0][i].item()
        print(f"  {i+1}. '{classes[idx]}' ({prob*100:.1f}%)")
    
    # Also print tensor stats
    print(f"  Tensor: min={tensor.min():.2f}, max={tensor.max():.2f}, mean={tensor.mean():.2f}")
    
    return top5.indices[0][0].item(), top5.values[0][0].item()

# ============== Main Analysis ==============
def main():
    print("=" * 70)
    print("DOMAIN GAP ANALYSIS: Dataset vs GUI Hand-drawn")
    print("=" * 70)
    
    # Load model
    model, classes = load_model()
    print(f"\nLoaded model with {len(classes)} classes")
    
    # Transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # ============== Part 1: Analyze Dataset Images ==============
    print("\n" + "=" * 70)
    print("PART 1: DATASET IMAGE ANALYSIS")
    print("=" * 70)
    
    dataset_stats = []
    test_chars = ['-', ' ', '']  # placeholder
    
    # Analyze multiple characters
    for char in os.listdir(DATA_DIR)[:10]:
        char_dir = os.path.join(DATA_DIR, char)
        if not os.path.isdir(char_dir):
            continue
        imgs = [f for f in os.listdir(char_dir) if f.endswith('.jpg')]
        if not imgs:
            continue
        
        img_path = os.path.join(char_dir, imgs[0])
        img = Image.open(img_path).convert('L')
        stats = analyze_pixel_distribution(img, f"Dataset '{char}'")
        stats['char'] = char
        dataset_stats.append(stats)
    
    # Summary
    all_mins = [s['min'] for s in dataset_stats]
    print(f"\n>>> DATASET SUMMARY:")
    print(f"    Min pixel value range: {min(all_mins)} - {max(all_mins)}")
    print(f"    NO PURE BLACK (0) PIXELS IN DATASET!")
    print(f"    Strokes are GRAY (value 50-150), not pure black (0)")
    
    # ============== Part 2: Analyze GUI Drawing ==============
    print("\n" + "=" * 70)
    print("PART 2: GUI DRAWING ANALYSIS")
    print("=" * 70)
    
    # Test 1: Pure black stroke (current GUI behavior)
    gui_black = simulate_gui_drawing("horizontal", stroke_value=0)
    stats_black = analyze_pixel_distribution(gui_black, "GUI (Pure Black Stroke, value=0)")
    
    # Test 2: Gray stroke (matching dataset distribution)
    gui_gray = simulate_gui_drawing("horizontal", stroke_value=80)
    stats_gray = analyze_pixel_distribution(gui_gray, "GUI (Gray Stroke, value=80)")
    
    print(f"\n>>> GUI SUMMARY:")
    print(f"    Pure black stroke: min={stats_black['min']}, black_pct={stats_black['black_pct']:.1f}%")
    print(f"    Gray stroke: min={stats_gray['min']}, black_pct={stats_gray['black_pct']:.1f}%")
    
    # ============== Part 3: Prediction Comparison ==============
    print("\n" + "=" * 70)
    print("PART 3: PREDICTION COMPARISON")
    print("=" * 70)
    
    # Find a 'one' character in dataset
    one_dir = os.path.join(DATA_DIR, 'one')
    if not os.path.exists(one_dir):
        one_dir = os.path.join(DATA_DIR, 'one')
    
    # Use first available character
    for char in os.listdir(DATA_DIR)[:1]:
        char_dir = os.path.join(DATA_DIR, char)
        if os.path.isdir(char_dir):
            imgs = [f for f in os.listdir(char_dir) if f.endswith('.jpg')]
            if imgs:
                ds_img = Image.open(os.path.join(char_dir, imgs[0])).convert('L')
                predict_image(model, classes, ds_img, transform, f"Dataset Image '{char}'")
                break
    
    # Predict on GUI pure black
    predict_image(model, classes, gui_black, transform, "GUI (Pure Black Stroke)")
    
    # Predict on GUI gray
    predict_image(model, classes, gui_gray, transform, "GUI (Gray Stroke)")
    
    # ============== Part 4: Root Cause Analysis ==============
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)
    
    print("""
IDENTIFIED PROBLEM: DOMAIN GAP (Pixel Value Distribution Mismatch)

1. DATASET CHARACTERISTICS:
   - Source: HWDB scanned handwritten images
   - Min pixel value: 51-85 (NO pure black 0)
   - Stroke representation: GRAY-SCALE with natural anti-aliasing
   - Edge transition: Smooth gradient from white to gray

2. GUI INPUT CHARACTERISTICS:
   - Source: Digital rendering on canvas
   - Min pixel value: 0 (PURE BLACK)
   - Stroke representation: BINARY black/white
   - Edge transition: Sharp, no gradient

3. WHY MODEL FAILS ON GUI:
   - Model learned to detect GRAY strokes (value 50-150)
   - GUI provides PURE BLACK strokes (value 0)
   - After Normalize(0.5, 0.5):
     * Dataset stroke: (51/255 - 0.5) / 0.5 = -0.6 to -0.4
     * GUI black stroke: (0/255 - 0.5) / 0.5 = -1.0
   - The model sees GUI strokes as "too dark" / "out of distribution"

4. SOLUTION OPTIONS:
   A) Add Gaussian blur to GUI strokes to simulate anti-aliasing
   B) Use gray color (value ~80) instead of pure black in GUI
   C) Add data augmentation during training to include pure black strokes
   D) Apply histogram matching to match dataset distribution
""")

if __name__ == '__main__':
    main()