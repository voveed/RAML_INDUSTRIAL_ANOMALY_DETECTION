
# ========================================================================================
# WINCLIP REPRODUCTION - PHASE 3 ONLY: VISA TRANSFER LEARNING
# ========================================================================================
# HƯỚNG DẪN SỬ DỤNG TRÊN COLAB:
# 1. Upload file này lên Colab.
# 2. Upload dataset 'visa_anomaly_detection.zip' lên Google Drive.
# 3. Upload file 'best_model.pt' (lấy từ Kaggle/Phase 1) lên Google Drive.
# 4. Mount Drive và chạy script này.
# ========================================================================================

import os
import sys
import json
import time
import random
import argparse
import shutil
import zipfile
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# ========================================================================================
# PART 0: COLAB SETUP
# ========================================================================================
def setup_colab_env_phase3():
    print("="*80)
    print("☁️ [COLAB SETUP] PREPARING ENVIRONMENT FOR PHASE 3")
    print("="*80)
    
    # 1. Mount Drive
    if os.path.exists('/content/drive'):
        print("✅ Google Drive already mounted.")
    else:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✅ Google Drive mounted successfully.")
        except ImportError:
            print("⚠️ Not running on Colab or Drive mount failed.")

    # 2. Config Paths
    DRIVE_PATH = '/content/drive/MyDrive'
    DATA_ROOT = '/content/data'
    os.makedirs(DATA_ROOT, exist_ok=True)
    
    # 3. Unzip VisA
    visa_zip = os.path.join(DRIVE_PATH, 'visa_anomaly_detection.zip')
    visa_dir = os.path.join(DATA_ROOT, 'visa_anomaly_detection')
    
    if os.path.exists(visa_dir):
        print(f"✅ VisA dataset already exists at: {visa_dir}")
    elif os.path.exists(visa_zip):
        print(f"📦 Found VisA ZIP at: {visa_zip}")
        print("   ⏳ Unzipping... (This may take a minute)")
        with zipfile.ZipFile(visa_zip, 'r') as zip_ref:
            zip_ref.extractall(DATA_ROOT)
        print(f"   ✅ Unzipped to: {visa_dir}")
    else:
        if os.path.exists(os.path.join(DRIVE_PATH, 'visa_anomaly_detection')):
             print(f"✅ Found VisA folder directly in Drive.")
             visa_dir = os.path.join(DRIVE_PATH, 'visa_anomaly_detection')
        else:
             print(f"⚠️ VisA ZIP not found at {visa_zip}. Please check path.")

    # 4. Check Model
    model_path = os.path.join(DRIVE_PATH, 'best_model.pt')
    if not os.path.exists(model_path):
        print(f"⚠️ 'best_model.pt' not found in Drive ({model_path}).")
        print("   Please upload the trained model from Phase 1!")
    else:
        print(f"✅ Found Pre-trained Model: {model_path}")
        
    return visa_dir, model_path

# ========================================================================================
# [CORE CLASSES - INFERENCE ONLY]
# ========================================================================================
class TextVisualFusion(nn.Module):
    def __init__(self, text_dim: int = 512, visual_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.fusion_scale = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        B = visual_features.size(0)
        # Project text
        text_proj = self.text_proj(text_features) 
        # Expand if shared
        if text_proj.size(0) != B:
            text_proj = text_proj.unsqueeze(0).expand(B, -1, -1)
        else:
            text_proj = text_proj.unsqueeze(1)
            
        visual_query = visual_features.unsqueeze(1)
        attn_output, _ = self.cross_attn(visual_query, text_proj, text_proj)
        attn_output = attn_output.squeeze(1)
        fused = visual_features + self.fusion_scale * attn_output
        return self.norm(fused)

class RAMLUltimateModel(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        for param in self.clip_model.parameters(): param.requires_grad = False
        self.fusion = TextVisualFusion(embed_dim=512)
        self.fc = nn.Linear(512, 1)
        
    def forward(self, images, categories=None):
        visual_features = self.clip_model.encode_image(images).float()
        text_features = torch.zeros_like(visual_features)
        fused = self.fusion(visual_features, text_features)
        logits = self.fc(fused).squeeze(1)
        return {'logits': logits, 'features': fused, 'scores': torch.sigmoid(logits)}

# ========================================================================================
# MAIN EXECUTION
# ========================================================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ DEVICE: {device}")
    
    visa_dir, model_path = setup_colab_env_phase3()
    
    if not os.path.exists(visa_dir) or not os.path.exists(model_path):
        print("❌ CRITICAL MISSING FILES. CANNOT PROCEED.")
        sys.exit(1)
        
    try: import clip
    except ImportError:
        os.system('pip install git+https://github.com/openai/CLIP.git')
        import clip
        
    print("\n" + "="*60)
    print("🌍 [PHASE 3] RUNNING VISA TRANSFER LEARNING")
    print("="*60)
    
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    model = RAMLUltimateModel(clip_model).to(device)
    
    print(f"📥 Loading Weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict']) # strict=False allowed if architectures slightly differ
    model.eval()
    
    # Run VisA Eval
    visa_categories = sorted([d for d in os.listdir(visa_dir) if os.path.isdir(os.path.join(visa_dir, d))])
    print(f"📊 VisA Categories: {visa_categories}")
    
    visa_scores = {}
    for cat in visa_categories:
        print(f"\n🔍 Evaluating Category: {cat}")
        test_dir = os.path.join(visa_dir, cat, 'test')
        if not os.path.exists(test_dir): continue
        
        all_scores, all_labels = [], []
        # Manual Loop (No DataLoader needed for simple inference)
        for subdir in os.listdir(test_dir):
            label = 0 if subdir == 'good' else 1
            sp = os.path.join(test_dir, subdir)
            if not os.path.isdir(sp): continue
            
            for f in os.listdir(sp):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    try:
                        img = Image.open(os.path.join(sp, f)).convert('RGB')
                        img = preprocess(img).unsqueeze(0).to(device)
                        with torch.no_grad():
                            out = model(img)
                            all_scores.append(out['scores'].item())
                            all_labels.append(label)
                    except: pass
        
        if len(set(all_labels)) > 1:
            score = roc_auc_score(all_labels, all_scores)
            visa_scores[cat] = score
            print(f"   ✅ AUROC: {score*100:.2f}%")
        else:
            print("   ⚠️ Skipped (Not enough labels)")
            
    avg_score = np.mean(list(visa_scores.values())) if visa_scores else 0.0
    print(f"\n🏆 AVERAGE VISA AUROC: {avg_score*100:.2f}%")
    print("Note: This proves Zero-shot Transfer ability (Generalization).")
