"""
KAGGLE: WINCLIP ZERO-SHOT BASELINE EVALUATION
================================================
Goal: Evaluate baseline WinCLIP performance (No Training)
Features:
- Pure Zero-shot inference
- Uses "damaged [obj]" vs "flawless [obj]" prompts
- Fast execution (Inference only)
- Uses same 20% test split as RAML for fair comparison
"""

import os
import sys
import subprocess
import json
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from glob import glob
from sklearn.metrics import roc_auc_score

# ==========================================
# AUTO-INSTALL DEPENDENCIES
# ==========================================
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import open_clip
except ImportError:
    print("⏳ Installing open_clip_torch...")
    install("open_clip_torch")
    install("ftfy")
    install("regex")
    import open_clip
    print("✅ Installed dependencies!")

# ==========================================
# CONFIG
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "ViT-B-16-plus-240"
PRETRAINED = "laion400m_e31"
CLASSES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

# ==========================================
# KAGGLE DATASET DETECTION
# ==========================================
def find_mvtec_root():
    """Auto-detect MVTec root directory in Kaggle"""
    candidates = [
        "/kaggle/input/mvtec-anomaly-detection",
        "/kaggle/input/mvtec-ad",
        "/kaggle/input/mvtecad",
        "/kaggle/input/industrial-defect-detection/mvtec_anomaly_detection"
    ]
    
    # Check known paths
    for p in candidates:
        if os.path.exists(p) and os.path.isdir(p):
            if 'bottle' in os.listdir(p):
                print(f"✅ Found dataset at: {p}")
                return p
    
    # Recursive search
    print("🔍 Searching recursively in /kaggle/input...")
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'bottle' in dirs and 'carpet' in dirs:
            print(f"✅ Found dataset at: {root}")
            return root
    
    raise FileNotFoundError("❌ MVTec dataset not found in /kaggle/input. Please add the dataset.")

# ==========================================
# HELPERS
# ==========================================
def encode_text_prompts(model, tokenizer, obj_name, device):
    """Create zero-shot classifier weights for [Normal, Anomaly]"""
    normal_prompts = [f"a photo of a flawless {obj_name}"]
    anomaly_prompts = [f"a photo of a damaged {obj_name}"]

    with torch.no_grad():
        norm_tokens = tokenizer(normal_prompts).to(device)
        anom_tokens = tokenizer(anomaly_prompts).to(device)

        norm_embed = model.encode_text(norm_tokens)
        anom_embed = model.encode_text(anom_tokens)

        norm_embed /= norm_embed.norm(dim=-1, keepdim=True)
        anom_embed /= anom_embed.norm(dim=-1, keepdim=True)

        norm_feat = norm_embed.mean(dim=0, keepdim=True)
        anom_feat = anom_embed.mean(dim=0, keepdim=True)

        text_weights = torch.cat([norm_feat, anom_feat], dim=0)
        text_weights /= text_weights.norm(dim=-1, keepdim=True)

    return text_weights

def transform_image(img_path, image_size=240):
    tr = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))
    ])
    img = Image.open(img_path).convert('RGB')
    return tr(img).unsqueeze(0)

# ==========================================
# MAIN EVALUATION
# ==========================================
def run_winclip_baseline():
    print("="*60)
    print("🚀 WINCLIP ZERO-SHOT BASELINE EVALUATION")
    print("="*60)
    
    DATA_ROOT = find_mvtec_root()
    print(f"Model: {CLIP_MODEL} ({PRETRAINED})")
    print(f"Device: {DEVICE}")

    # Load CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    model.to(DEVICE)
    model.eval()

    results = {}

    for cls in CLASSES:
        print(f"\n▶️ Evaluating: {cls.upper()}")

        test_dir = os.path.join(DATA_ROOT, cls, 'test')
        if not os.path.exists(test_dir):
            print(f"⚠️ Skipped {cls}: path not found")
            continue

        gt_labels = []
        pred_scores = []

        # Prepare Text Classifiers
        text_weights = encode_text_prompts(model, tokenizer, cls, DEVICE)

        # ====================================================
        # REPLICATE THE SPLIT LOGIC FROM RAML TRAINING
        # ====================================================
        train_dir = os.path.join(DATA_ROOT, cls, 'train', 'good')
        test_good_dir = os.path.join(DATA_ROOT, cls, 'test', 'good')

        all_normal_files = glob(os.path.join(train_dir, '*.png')) + \
                           glob(os.path.join(test_good_dir, '*.png'))
        all_normal_files = sorted(all_normal_files)

        all_anomaly_files = []
        test_root = os.path.join(DATA_ROOT, cls, 'test')
        for subdir in os.listdir(test_root):
            if subdir != 'good':
                fs = glob(os.path.join(test_root, subdir, '*.png'))
                all_anomaly_files.extend(fs)
        all_anomaly_files = sorted(all_anomaly_files)

        # --- SPLIT LOGIC (MUST MATCH RAML EXACTLY) ---
        random.seed(42)  # CRITICAL: Same seed as Training

        random.shuffle(all_normal_files)
        n_normal = len(all_normal_files)
        split_idx_norm = int(n_normal * 0.8)
        test_normal_files = all_normal_files[split_idx_norm:]

        random.shuffle(all_anomaly_files)
        n_anom = len(all_anomaly_files)
        split_idx_anom = int(n_anom * 0.8)
        test_anomaly_files = all_anomaly_files[split_idx_anom:]

        print(f"   Sample Set: {len(test_normal_files)} Normal, {len(test_anomaly_files)} Anomalies (20% Split)")

        norm_imgs = [(f, 0) for f in test_normal_files]
        anom_imgs = [(f, 1) for f in test_anomaly_files]
        all_samples = norm_imgs + anom_imgs

        for img_path, label in tqdm(all_samples, desc=cls):
            try:
                img_tensor = transform_image(img_path).to(DEVICE)
                with torch.no_grad():
                    image_feat = model.encode_image(img_tensor)
                    image_feat /= image_feat.norm(dim=-1, keepdim=True)

                    logits = image_feat @ text_weights.t()
                    probs = logits.softmax(dim=-1)
                    score = probs[0, 1].item()

                    pred_scores.append(score)
                    gt_labels.append(label)
            except Exception as e:
                print(f"Error {img_path}: {e}")

        if len(gt_labels) > 0 and len(set(gt_labels)) > 1:
            auroc = roc_auc_score(gt_labels, pred_scores)
            results[cls] = auroc
            print(f"   Score: {auroc*100:.2f}%")
        else:
            results[cls] = 0.5
            print(f"   Score: N/A (single class)")

    # Final Summary
    print("\n" + "="*60)
    print("🏆 WINCLIP ZERO-SHOT RESULTS (20% Test Split)")
    print("="*60)
    avg_score = np.mean(list(results.values()))
    for k, v in sorted(results.items()):
        print(f"   {k:12s}: {v*100:.2f}%")
    print("-" * 60)
    print(f"   {'AVERAGE':12s}: {avg_score*100:.2f}%")

    # Save Results
    result_data = {
        "WinCLIP_ZeroShot": {
            "mean_auroc": round(avg_score * 100, 2),
            "details": {k: round(v * 100, 2) for k, v in results.items()}
        }
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/result_WinCLIP_ZeroShot.json", "w") as f:
        json.dump(result_data, f, indent=4)
    print("\n💾 Saved to results/result_WinCLIP_ZeroShot.json")

if __name__ == "__main__":
    run_winclip_baseline()
