"""
VISUALIZATION SCRIPT FOR RAML REPORT
=====================================
Generates all charts and saves to 'newimages/' folder.
Run locally with: python visualize_results.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# ==========================================
# CONFIG
# ==========================================
OUTPUT_DIR = "newimages"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette
COLORS = {
    'RAML': '#E74C3C',        # Red
    'WinCLIP': '#3498DB',     # Blue
    'Fixed': '#2ECC71',       # Green
    'Strong': '#9B59B6',      # Purple
    'VisA': '#F39C12'         # Orange
}

plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# ==========================================
# LOAD DATA
# ==========================================
def load_results():
    results_dir = "results"
    data = {}
    
    files = {
        'RAML': 'result_RAML_Proposed.json',
        'WinCLIP': 'result_WinCLIP_ZeroShot.json',
        'Fixed': 'result_Fixed_Margin_05.json',
        'Strong': 'result_Strong_Margin_10.json',
        'VisA': 'result_VisA_Transfer.json'
    }
    
    for key, filename in files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data[key] = json.load(f)
            print(f"✅ Loaded {filename}")
        else:
            print(f"⚠️ Not found: {filename}")
    
    return data

# ==========================================
# CHART 1: Overall AUROC Comparison (Bar)
# ==========================================
def plot_overall_comparison(data):
    print("\n📊 Creating: Overall AUROC Comparison...")
    
    methods = []
    scores = []
    colors = []
    
    if 'WinCLIP' in data:
        methods.append('WinCLIP\n(Zero-shot)')
        scores.append(list(data['WinCLIP'].values())[0]['mean_auroc'])
        colors.append(COLORS['WinCLIP'])
    
    if 'Strong' in data:
        methods.append('Strong Margin\n(m=1.0)')
        scores.append(list(data['Strong'].values())[0]['mean_auroc'])
        colors.append(COLORS['Strong'])
    
    if 'Fixed' in data:
        methods.append('Fixed Margin\n(m=0.5)')
        scores.append(list(data['Fixed'].values())[0]['mean_auroc'])
        colors.append(COLORS['Fixed'])
    
    if 'RAML' in data:
        methods.append('RAML\n(Proposed)')
        scores.append(list(data['RAML'].values())[0]['mean_auroc'])
        colors.append(COLORS['RAML'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, scores, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(f'{score:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('AUROC (%)', fontsize=14)
    ax.set_title('Overall AUROC Comparison on MVTec AD (20% Test Split)', fontsize=16, fontweight='bold')
    ax.set_ylim(80, 100)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'overall_comparison.png'))
    plt.close()
    print(f"   ✅ Saved: {OUTPUT_DIR}/overall_comparison.png")

# ==========================================
# CHART 2: Per-Category Comparison (Grouped Bar)
# ==========================================
def plot_per_category_comparison(data):
    print("\n📊 Creating: Per-Category Comparison...")
    
    if 'RAML' not in data or 'WinCLIP' not in data:
        print("   ⚠️ Missing RAML or WinCLIP data")
        return
    
    raml_details = list(data['RAML'].values())[0]['details']
    winclip_details = list(data['WinCLIP'].values())[0]['details']
    
    categories = sorted(raml_details.keys())
    raml_scores = [raml_details[c] for c in categories]
    winclip_scores = [winclip_details[c] for c in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bars1 = ax.bar(x - width/2, winclip_scores, width, label='WinCLIP (Zero-shot)', 
                   color=COLORS['WinCLIP'], edgecolor='black')
    bars2 = ax.bar(x + width/2, raml_scores, width, label='RAML (Proposed)', 
                   color=COLORS['RAML'], edgecolor='black')
    
    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('AUROC (%)', fontsize=12)
    ax.set_title('Per-Category AUROC: RAML vs WinCLIP Zero-shot', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in categories], rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(50, 105)
    ax.grid(axis='y', alpha=0.3)
    
    # Highlight improvements
    for i, (r, w) in enumerate(zip(raml_scores, winclip_scores)):
        diff = r - w
        if diff > 5:
            ax.annotate(f'+{diff:.1f}', xy=(x[i] + width/2, r), 
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=8, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'per_category_comparison.png'))
    plt.close()
    print(f"   ✅ Saved: {OUTPUT_DIR}/per_category_comparison.png")

# ==========================================
# CHART 3: Radar Chart (Spider Plot) - Trained Models
# ==========================================
def plot_radar_chart(data):
    print("\n📊 Creating: Radar Chart (Trained Models)...")
    
    if 'RAML' not in data or 'Fixed' not in data or 'Strong' not in data:
        print("   ⚠️ Missing RAML, Fixed, or Strong data")
        return
    
    raml_details = list(data['RAML'].values())[0]['details']
    fixed_details = list(data['Fixed'].values())[0]['details']
    strong_details = list(data['Strong'].values())[0]['details']
    
    categories = sorted(raml_details.keys())
    N = len(categories)
    
    raml_values = [raml_details[c] for c in categories]
    fixed_values = [fixed_details[c] for c in categories]
    strong_values = [strong_details[c] for c in categories]
    
    # Complete the loop
    raml_values += raml_values[:1]
    fixed_values += fixed_values[:1]
    strong_values += strong_values[:1]
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    ax.plot(angles, strong_values, 'o-', linewidth=2, label='Strong Margin (m=1.0)', color=COLORS['Strong'])
    ax.fill(angles, strong_values, alpha=0.15, color=COLORS['Strong'])
    
    ax.plot(angles, fixed_values, 'o-', linewidth=2, label='Fixed Margin (m=0.5)', color=COLORS['Fixed'])
    ax.fill(angles, fixed_values, alpha=0.15, color=COLORS['Fixed'])
    
    ax.plot(angles, raml_values, 'o-', linewidth=2, label='RAML (Adaptive)', color=COLORS['RAML'])
    ax.fill(angles, raml_values, alpha=0.25, color=COLORS['RAML'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(50, 105)
    ax.set_title('Per-Category Performance: Trained Models Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'radar_chart.png'))
    plt.close()
    print(f"   ✅ Saved: {OUTPUT_DIR}/radar_chart.png")

# ==========================================
# CHART 4: Ablation Study (Margin Comparison)
# ==========================================
def plot_ablation_margin(data):
    print("\n📊 Creating: Ablation Study - Margin Comparison...")
    
    methods = []
    scores = []
    colors = []
    
    if 'Strong' in data:
        methods.append('Strong Margin\n(m=1.0, λ=0)')
        scores.append(list(data['Strong'].values())[0]['mean_auroc'])
        colors.append('#E8E8E8')
    
    if 'Fixed' in data:
        methods.append('Fixed Margin\n(m=0.5, λ=0)')
        scores.append(list(data['Fixed'].values())[0]['mean_auroc'])
        colors.append('#C0C0C0')
    
    if 'RAML' in data:
        methods.append('Adaptive Margin\n(m=0.5+λσ+λr)')
        scores.append(list(data['RAML'].values())[0]['mean_auroc'])
        colors.append(COLORS['RAML'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(methods, scores, color=colors, edgecolor='black', height=0.6)
    
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.annotate(f'{score:.2f}%',
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('AUROC (%)', fontsize=12)
    ax.set_title('Ablation Study: Effect of Adaptive Margin', fontsize=14, fontweight='bold')
    ax.set_xlim(80, 100)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_margin.png'))
    plt.close()
    print(f"   ✅ Saved: {OUTPUT_DIR}/ablation_margin.png")

# ==========================================
# CHART 5: VisA Transfer Learning Results
# ==========================================
def plot_visa_transfer(data):
    print("\n📊 Creating: VisA Transfer Learning Results...")
    
    if 'VisA' not in data:
        print("   ⚠️ Missing VisA data")
        return
    
    visa_details = list(data['VisA'].values())[0]['details']
    mean_auroc = list(data['VisA'].values())[0]['mean_auroc']
    
    categories = sorted(visa_details.keys())
    scores = [visa_details[c] for c in categories]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = [COLORS['VisA'] if s >= 70 else '#FFD93D' if s >= 60 else '#FF6B6B' for s in scores]
    bars = ax.bar(categories, scores, color=colors, edgecolor='black')
    
    ax.axhline(y=mean_auroc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_auroc:.2f}%')
    
    ax.set_xlabel('VisA Category', fontsize=12)
    ax.set_ylabel('AUROC (%)', fontsize=12)
    ax.set_title('Transfer Learning: MVTec-trained RAML on VisA Dataset', fontsize=14, fontweight='bold')
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylim(50, 100)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'visa_transfer.png'))
    plt.close()
    print(f"   ✅ Saved: {OUTPUT_DIR}/visa_transfer.png")

# ==========================================
# CHART 6: Improvement Heatmap
# ==========================================
def plot_improvement_heatmap(data):
    print("\n📊 Creating: Improvement Heatmap...")
    
    if 'RAML' not in data or 'WinCLIP' not in data:
        print("   ⚠️ Missing data")
        return
    
    raml_details = list(data['RAML'].values())[0]['details']
    winclip_details = list(data['WinCLIP'].values())[0]['details']
    
    categories = sorted(raml_details.keys())
    improvements = [raml_details[c] - winclip_details[c] for c in categories]
    
    # Sort by improvement
    sorted_data = sorted(zip(categories, improvements), key=lambda x: x[1], reverse=True)
    categories_sorted = [x[0] for x in sorted_data]
    improvements_sorted = [x[1] for x in sorted_data]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#27AE60' if x > 0 else '#E74C3C' for x in improvements_sorted]
    bars = ax.barh(categories_sorted, improvements_sorted, color=colors, edgecolor='black')
    
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('AUROC Improvement (RAML - WinCLIP)', fontsize=12)
    ax.set_title('Per-Category Improvement: RAML vs WinCLIP Zero-shot', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, improvements_sorted):
        x_pos = bar.get_width()
        offset = 3 if val > 0 else -25
        ax.annotate(f'{val:+.1f}%', xy=(x_pos, bar.get_y() + bar.get_height()/2),
                   xytext=(offset, 0), textcoords='offset points',
                   ha='left' if val > 0 else 'right', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'improvement_heatmap.png'))
    plt.close()
    print(f"   ✅ Saved: {OUTPUT_DIR}/improvement_heatmap.png")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("🎨 RAML VISUALIZATION SCRIPT")
    print("="*60)
    
    data = load_results()
    
    if not data:
        print("\n❌ No result files found. Make sure you're in the correct directory.")
        exit(1)
    
    # Generate all charts
    plot_overall_comparison(data)
    plot_per_category_comparison(data)
    plot_radar_chart(data)
    plot_ablation_margin(data)
    plot_visa_transfer(data)
    plot_improvement_heatmap(data)
    
    print("\n" + "="*60)
    print(f"✅ ALL CHARTS SAVED TO: {OUTPUT_DIR}/")
    print("="*60)
    print("\nFiles created:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"   📊 {f}")
