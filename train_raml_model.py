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
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from glob import glob
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

# ========================================================================================
# PART 0: KAGGLE SETUP
# ========================================================================================
def setup_kaggle_env():
    print("="*80)
    print("☁️ [KAGGLE SETUP] PREPARING ENVIRONMENT")
    print("="*80)
    
    # 1. Auto-detect MVTec Dataset in /kaggle/input
    candidate_paths = [
        '/kaggle/input/mvtec-anomaly-detection',
        '/kaggle/input/mvtec-ad',
        '/kaggle/input/mvtecad',
        '/kaggle/input/industrial-defect-detection/mvtec_anomaly_detection'
    ]
    
    mvtec_dir = None
    
    # Fast check known paths
    for p in candidate_paths:
        if os.path.exists(p) and os.path.isdir(p):
            # Check if it has category folders (e.g., 'bottle')
            if 'bottle' in os.listdir(p):
                mvtec_dir = p
                print(f"✅ Found MVTec dataset at: {mvtec_dir}")
                break
    
    # Slow recursive search if not found
    if not mvtec_dir:
        print("🔍 Searching recursively in /kaggle/input...")
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'bottle' in dirs and 'carpet' in dirs:
                mvtec_dir = root
                print(f"✅ Auto-detected MVTec at: {mvtec_dir}")
                break
                
    if not mvtec_dir:
        print("❌ CRITICAL: MVTec dataset not found in /kaggle/input.")
        print("   Please add the 'mvtec-anomaly-detection' dataset to this notebook.")
        # Fallback to creating a dummy directory so script doesn't crash immediately (optional)
        return None
        
    return mvtec_dir

# ========================================================================================
# [CORE CLASSES - AUTHENTIC IMPLEMENTATION]
# ========================================================================================

# --- COMPONENT 2: ATTENTION & PYRAMID ---
class CrossScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout, self.norm = nn.Dropout(dropout), nn.LayerNorm(dim)
        
    def forward(self, query, key_value):
        B, D = query.shape
        _, N, _ = key_value.shape
        q = self.q_proj(query).unsqueeze(1).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (self.dropout(attn) @ v).transpose(1, 2).reshape(B, D)
        return self.norm(query + self.out_proj(out))

class MultiScalePyramid(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256):
        super().__init__()
        self.s1_proj, self.s2_proj, self.s3_proj = nn.Linear(feature_dim, hidden_dim), nn.Linear(feature_dim, hidden_dim), nn.Linear(feature_dim, hidden_dim)
        self.attn12, self.attn23, self.attn31 = CrossScaleAttention(hidden_dim), CrossScaleAttention(hidden_dim), CrossScaleAttention(hidden_dim)
        self.fusion = nn.Sequential(nn.Linear(hidden_dim*3, hidden_dim*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim*2, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, g, feat_s2, feat_s3):
        f1, f2, f3 = self.s1_proj(g), self.s2_proj(feat_s2.mean(1)), self.s3_proj(feat_s3.mean(1))
        f1a = self.attn12(f1, self.s2_proj(feat_s2))
        f2a = self.attn23(f2, self.s3_proj(feat_s3))
        f3a = self.attn31(f3, f1.unsqueeze(1))
        return F.normalize(self.norm(self.fusion(torch.cat([f1a, f2a, f3a], dim=-1))), p=2.0, dim=1)

# --- COMPONENT 3: TRACKER & LOSS ---
class AutoDifficultyTracker:
    def __init__(self, momentum=0.7, weight_scale=0.5):
        self.momentum, self.weight_scale = momentum, weight_scale
        self.running_loss, self.count = {}, {}
    def update(self, cat, loss):
        if cat not in self.running_loss: self.running_loss[cat] = loss
        else: self.running_loss[cat] = self.momentum * self.running_loss[cat] + (1 - self.momentum) * loss
    def get_weight(self, cat):
        if not self.running_loss or len(self.running_loss) < 2: return 1.0
        if cat not in self.running_loss: return 1.0 # Safety for first batch of new category
        g_loss = np.mean(list(self.running_loss.values()))
        return 1.0 if g_loss < 1e-6 else float(np.clip(1.0 + self.weight_scale * (self.running_loss[cat] - g_loss) / g_loss, 0.5, 2.0))

class MACCLLoss(nn.Module):
    def __init__(self, feature_dim=256, margin_base=0.5, lambda_sigma=0.3, lambda_resolution=0.3, 
                 original_resolution=900, model_resolution=224, temperature=0.07, 
                 alpha=1.0, beta=1.0, gamma=0.5):
        super().__init__()
        self.register_buffer('normal_center', torch.zeros(feature_dim))
        self.register_buffer('running_sigma', torch.tensor(1.0))
        self.margin_base = margin_base
        self.lambda_sigma = lambda_sigma
        self.lambda_resolution = lambda_resolution
        self.resolution_ratio = model_resolution / original_resolution
        self.temperature = temperature
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.difficulty_tracker = AutoDifficultyTracker()
        
    def compute_center_loss(self, features, normal_mask):
        if normal_mask.sum() == 0: return torch.tensor(0.0, device=features.device), torch.zeros(features.size(0), device=features.device)
        dist_sq = ((features - self.normal_center) ** 2).sum(dim=1)
        loss_raw = dist_sq * normal_mask.float()
        if normal_mask.sum() > 0:
            with torch.no_grad(): self.normal_center.copy_(0.8 * self.normal_center + 0.2 * features[normal_mask].mean(dim=0))
        return loss_raw.sum() / (normal_mask.sum() + 1e-8), loss_raw

    def compute_margin_loss(self, features, anomaly_mask, normal_mask):
        with torch.no_grad():
            if normal_mask.sum() > 0: self.running_sigma.copy_(0.9 * self.running_sigma + 0.1 * features[normal_mask].std())
        m_adaptive = self.margin_base + self.lambda_sigma * self.running_sigma + self.lambda_resolution * (1 - self.resolution_ratio)
        
        # USE ROBUST LINALG NORM TO AVOID TYPE ERROR
        dist_to_center = torch.linalg.norm(features - self.normal_center, ord=2, dim=1)
        
        loss_raw = F.relu(m_adaptive - dist_to_center) * anomaly_mask.float()
        return (loss_raw.sum() / anomaly_mask.sum()) if anomaly_mask.sum() > 0 else torch.tensor(0.0, device=features.device), loss_raw, m_adaptive.item()

    def compute_contrastive_loss(self, features, labels):
        if labels.sum() == 0 or (1 - labels).sum() == 0: return torch.tensor(0.0, device=features.device), torch.zeros(features.size(0), device=features.device)
        features = F.normalize(features, p=2.0, dim=1) # Explicit p=2.0
        sim_matrix = torch.mm(features, features.t()) / self.temperature
        mask_pos = (labels.view(-1, 1) == labels.view(1, -1)).float(); mask_pos.fill_diagonal_(0)
        mask_neg = (labels.view(-1, 1) != labels.view(1, -1)).float()
        raw_losses = torch.zeros(features.size(0), device=features.device)
        for i in range(features.size(0)):
            pos_sim, neg_sim = sim_matrix[i, mask_pos[i].bool()], sim_matrix[i, mask_neg[i].bool()]
            if len(pos_sim) > 0 and len(neg_sim) > 0:
                raw_losses[i] = -torch.log(torch.exp(pos_sim).sum() / (torch.exp(pos_sim).sum() + torch.exp(neg_sim).sum() + 1e-8))
        return (raw_losses.sum() / (raw_losses > 0).sum()) if (raw_losses > 0).sum() > 0 else torch.tensor(0.0, device=features.device, requires_grad=True), raw_losses

    def forward(self, features, labels, categories=None):
        normal_mask, anomaly_mask = (labels == 0), (labels == 1)
        l_center, r_center = self.compute_center_loss(features, normal_mask)
        l_margin, r_margin, adaptive_m = self.compute_margin_loss(features, anomaly_mask, normal_mask)
        l_con, r_con = self.compute_contrastive_loss(features, labels)
        
        raw_total = self.alpha * r_center + self.beta * r_margin + self.gamma * r_con
        if categories:
            weights = torch.tensor([self.difficulty_tracker.get_weight(c) for c in categories], device=features.device)
            for c, l in zip(categories, raw_total.detach().cpu().numpy()):
                self.difficulty_tracker.update(c, float(l))
            total_loss = (raw_total * weights).mean()
        else: total_loss = raw_total.mean()
        
        return total_loss, {'total': total_loss, 'adaptive_margin': adaptive_m, 'running_sigma': self.running_sigma.item()}

# --- COMPONENT 4: MODEL ---
class TextVisualFusion(nn.Module):
    def __init__(self, text_dim: int = 512, visual_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        # Project text to visual dimension (frozen after init)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        # Simple cross-attention without complex gating
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        # Lightweight fusion - residual style
        self.fusion_scale = nn.Parameter(torch.tensor(0.1))  # Start small, learn to increase
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Simple fusion WITHOUT gating. Uses residual connection.
        text_features: [B, 2, 512] (Normal, Anomaly)
        """
        B = visual_features.size(0)
        
        # Project text to visual dimension
        # text_features is [B, 2, 512] -> [B, 2, 256]
        text_proj = self.text_proj(text_features)
        
        # Safety check for dimensions
        if text_proj.dim() == 2: # [2, 256] -> Shared case (legacy)
             text_proj = text_proj.unsqueeze(0).expand(B, -1, -1)
             
        # Cross-attention: Visual (query) attends to Text (key, value)
        # Visual: [B, 256] -> [B, 1, 256]
        # Text:   [B, 2, 256]
        visual_query = visual_features.unsqueeze(1)  
        
        # Attn: Query=[B,1,D], Key=[B,2,D], Val=[B,2,D]
        attn_output, _ = self.cross_attn(visual_query, text_proj, text_proj)
        attn_output = attn_output.squeeze(1)  # [B, 256]
        
        # SIMPLE RESIDUAL FUSION
        fused = visual_features + self.fusion_scale * attn_output
        
        return self.norm(fused)

class RAMLUltimateModel(nn.Module):
    def __init__(self, clip_model, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.clip = clip_model
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Freeze CLIP backbone
        for param in self.clip.parameters(): param.requires_grad = False
            
        # Multi-scale pyramid with attention
        self.pyramid = MultiScalePyramid(feature_dim, hidden_dim)
        
        # Text-Visual Fusion (NOVEL) - USING 3 ARGS AS IN PIPELINE
        self.fusion = TextVisualFusion(text_dim=feature_dim, visual_dim=hidden_dim, hidden_dim=hidden_dim)
        
        # Feature refinement head
        self.feature_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        
        # Classification head
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))
        
        self._text_cache = {}

    def encode_text_prompts(self, device, categories):
        # OPTIMIZED: Class-Specific Prompts for Maximum Accuracy
        # If all categories are the same (often the case in single-class training), strict cache.
        if len(set(categories)) == 1:
            c_name = categories[0]
            if c_name in self._text_cache: return self._text_cache[c_name].unsqueeze(0).expand(len(categories), -1, -1)
            
            with torch.no_grad():
                import clip as c
                # Use the specific object name, e.g., 'bottle' instead of 'object'
                ne = F.normalize(self.clip.encode_text(c.tokenize([f"a photo of a flawless {c_name}"]).to(device)).float(), p=2.0, dim=-1)
                ae = F.normalize(self.clip.encode_text(c.tokenize([f"a photo of a damaged {c_name}"]).to(device)).float(), p=2.0, dim=-1)
                emb = torch.cat([ne, ae], dim=0) # [2, D]
                self._text_cache[c_name] = emb
                return emb.unsqueeze(0).expand(len(categories), -1, -1) # [B, 2, D]
        
        # Mixed batch (rare in this training loop but handled for safety)
        batch_emb = []
        with torch.no_grad():
            import clip as c
            for cat in categories:
                if cat in self._text_cache:
                    batch_emb.append(self._text_cache[cat])
                else:
                    ne = F.normalize(self.clip.encode_text(c.tokenize([f"a photo of a flawless {cat}"]).to(device)).float(), p=2.0, dim=-1)
                    ae = F.normalize(self.clip.encode_text(c.tokenize([f"a photo of a damaged {cat}"]).to(device)).float(), p=2.0, dim=-1)
                    emb = torch.cat([ne, ae], dim=0)
                    self._text_cache[cat] = emb
                    batch_emb.append(emb)
        
        return torch.stack(batch_emb) # [B, 2, D]

    def extract_multiscale(self, images):
        B, C, H, W = images.shape
        with torch.no_grad():
            g = self.clip.encode_image(images).float()
            feat_s2 = self.clip.encode_image(F.interpolate(images.view(B*4, C, H//2, W//2), size=(224,224), mode='bilinear')).float().view(B, 4, -1)
            feat_s3 = self.clip.encode_image(F.interpolate(images.view(B*16, C, H//4, W//4), size=(224,224), mode='bilinear')).float().view(B, 16, -1)
        return g, feat_s2, feat_s3

    def forward(self, images, categories=None):
        g, feat_s2, feat_s3 = self.extract_multiscale(images)
        pyramid_feat = self.pyramid(g, feat_s2, feat_s3)
        
        # Use generic object if categories missing (inference fallback)
        if categories is None: categories = ["object"] * images.size(0)
            
        text_emb = self.encode_text_prompts(images.device, categories) # [B, 2, D]
        
        # Fusion using the robust forward method
        fused = self.fusion(pyramid_feat, text_emb)
        
        # Deviation Score (Updated for Batched Text Embeddings [B, 2, D])
        g_norm = F.normalize(g, p=2.0, dim=-1)
        
        # text_emb is [B, 2, D]. slice 0=normal, 1=anomaly
        # Dot product per sample: (g_norm * emb).sum(-1)
        # Score = (AnomalySim - NormalSim + 1) / 2
        normal_sim = (g_norm * text_emb[:, 0]).sum(dim=-1)
        anomaly_sim = (g_norm * text_emb[:, 1]).sum(dim=-1)
        
        text_score = (anomaly_sim - normal_sim + 1) / 2
        
        refined = self.feature_head(fused)
        logits = self.classifier(refined).squeeze(-1)
        visual_score = torch.sigmoid(logits)
        
        return {'logits': logits, 'features': refined, 'scores': 0.85 * visual_score + 0.15 * text_score}

# ========================================================================================
# EXECUTION LOGIC
# ========================================================================================
class MVTecDatasetWithSynthetic(Dataset):
    def __init__(self, data_dir, categories, split='train', transform=None, use_real_anomalies=False, train_ratio=0.8):
        self.transform = transform
        self.split = split
        self.samples = []
        
        for category in categories:
            # 1. Gather all samples first
            train_dir = os.path.join(data_dir, category, 'train', 'good')
            if not os.path.exists(train_dir): continue # Skip if bad path
            
            tx_samples = glob(os.path.join(train_dir, '*.png')) + glob(os.path.join(train_dir, '*.jpg'))
            
            test_dir = os.path.join(data_dir, category, 'test')
            test_good = []
            test_bad = []
            
            if os.path.exists(test_dir):
                for subdir in os.listdir(test_dir):
                    if not os.path.isdir(os.path.join(test_dir, subdir)): continue
                    label = 0 if subdir == 'good' else 1
                    imgs = glob(os.path.join(test_dir, subdir, '*.png')) + glob(os.path.join(test_dir, subdir, '*.jpg'))
                    for i in imgs: 
                        item = {'path': i, 'label': label, 'category': category}
                        if label == 0: test_good.append(item)
                        else: test_bad.append(item)
            
            # 2. Stratified Split (Seed 42)
            # Sort for stability before shuffle
            test_good.sort(key=lambda x: x['path'])
            test_bad.sort(key=lambda x: x['path'])
            
            random.seed(42)
            random.shuffle(test_good)
            random.shuffle(test_bad)
            
            split_g = int(len(test_good) * train_ratio)
            split_b = int(len(test_bad) * train_ratio)
            
            if split == 'train':
                # Train = All Original Train + 80% TestGood + 80% TestBad
                for p in tx_samples: self.samples.append({'path': p, 'label': 0, 'category': category})
                self.samples.extend(test_good[:split_g])
                if use_real_anomalies:
                    self.samples.extend(test_bad[:split_b])
            else:
                # Test = 20% TestGood + 20% TestBad
                self.samples.extend(test_good[split_g:])
                self.samples.extend(test_bad[split_b:])

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        item = self.samples[idx]
        try: img = Image.open(item['path']).convert('RGB')
        except: return torch.zeros(3, 224, 224), 0, item['category']
        if self.transform: img = self.transform(img)
        # Synthetic generation logic removed
        return img, item['label'], item['category']

def train_ablation_run(model, train_loader, args, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args['lr'])
    
    if args['name'] == 'Standard BCE':
        loss_fn = nn.BCEWithLogitsLoss()
        print(f"👉 Using BCE Loss (Baseline)")
    else:
        l_sigma = args.get('lambda_sigma', 0.3)
        l_res = args.get('lambda_resolution', 0.3)
        m_base = args.get('margin_base', 0.5)
        loss_fn = MACCLLoss(feature_dim=256, margin_base=m_base, lambda_sigma=l_sigma, lambda_resolution=l_res).to(device)
        print(f"👉 Using MACCL Loss: Base={m_base}, Sigma={l_sigma}, Res={l_res}")

    model.train()
    print(f"🚀 Training: {args['name']} (Epochs: {args['epochs']})")
    
    for epoch in range(1, args['epochs'] + 1):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for imgs, labels, cats in pbar:
            imgs, labels = imgs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            out = model(imgs, categories=list(cats))
            
            if args['name'] == 'Standard BCE':
                loss = loss_fn(out['logits'], labels)
            else:
                # Authentic: Loss on FEATURES (Metric Learning)
                loss_maccl, _ = loss_fn(out['features'], labels, list(cats))
                
                # CRITICAL FIX: Also train the Classifier Head!
                # Without this, 'logits' and 'visual_score' come from random untrained weights.
                loss_bce = F.binary_cross_entropy_with_logits(out['logits'], labels)
                
                loss = loss_maccl + loss_bce
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Added for stability
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f"   📉 Epoch {epoch} Completed | Avg Loss: {avg_loss:.4f}")
            
    return model

def evaluate_run(model, test_loader, device):
    model.eval()
    results_by_cat = defaultdict(list)
    
    with torch.no_grad():
        for imgs, labels, cats in test_loader:
            imgs = imgs.to(device)
            out = model(imgs, categories=list(cats))
            scores = out['scores'].cpu().numpy()
            lbls = labels.numpy()
            
            for score, lbl, cat in zip(scores, lbls, cats):
                results_by_cat[cat].append((lbl, score))
    
    # Calculate Per-Category AUROC
    auroc_by_cat = {}
    for cat, data in results_by_cat.items():
        lbls, scrs = zip(*data)
        if len(set(lbls)) > 1:
            auroc_by_cat[cat] = roc_auc_score(lbls, scrs)
        else:
            auroc_by_cat[cat] = 0.5 # Fallback
            
    mean_auroc = np.mean(list(auroc_by_cat.values()))
    
    print(f"\n📊 Detailed Results for Current Run:")
    for cat, score in sorted(auroc_by_cat.items()):
        print(f"   - {cat}: {score*100:.2f}%")
        
    return mean_auroc, auroc_by_cat

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ DEVICE: {device}")
    
    mvtec_dir = setup_kaggle_env()
    
    if not os.path.exists(mvtec_dir):
        print("❌ CRITICAL: MVTec dataset not found.")
        sys.exit(1)
        
    try: import clip
    except ImportError:
        os.system('pip install git+https://github.com/openai/CLIP.git')
        import clip
        
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    categories = sorted(os.listdir(mvtec_dir))
    if 'checkpoints' in categories: categories.remove('checkpoints')

    print("\n" + "="*60)
    print("🔬 [PHASE 2 - KAGGLE] RUNNING ABLATION STUDIES: RAML PROPOSED")
    print("="*60)
    
    # Run RAML (Proposed)
    raml_run = {'name': 'RAML (Proposed)', 'epochs': 20, 'lr': 5e-5, 'margin_base': 0.5, 'lambda_sigma': 0.3, 'lambda_resolution': 0.3}
    
    train_ds = MVTecDatasetWithSynthetic(mvtec_dir, categories, split='train', transform=preprocess, train_ratio=0.8, use_real_anomalies=True)
    test_ds = MVTecDatasetWithSynthetic(mvtec_dir, categories, split='test', transform=preprocess, train_ratio=0.8, use_real_anomalies=True)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    print(f"\n▶️ Running: {raml_run['name']}")
    model = RAMLUltimateModel(clip_model).to(device)
    model = train_ablation_run(model, train_loader, raml_run, device)
    score, detailed_scores = evaluate_run(model, test_loader, device)
    
    print(f"✅ Final Result RAML: {score*100:.2f}%")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model_RAML_Proposed.pt")
    print("💾 Saved checkpoint to checkpoints/model_RAML_Proposed.pt")

    # SAVE RESULTS TO JSON
    result_data = {
        "RAML_Proposed": {
            "mean_auroc": round(score * 100, 2),
            "details": {k: round(v * 100, 2) for k, v in detailed_scores.items()}
        }
    }
    
    os.makedirs("RAML_INDUSTRIAL_ANOMALY_DETECTION/results", exist_ok=True)
    with open("RAML_INDUSTRIAL_ANOMALY_DETECTION/results/result_RAML_Proposed.json", "w") as f:
        json.dump(result_data, f, indent=4)
    print("💾 Saved results to RAML_INDUSTRIAL_ANOMALY_DETECTION/results/result_RAML_Proposed.json")
