# RAML: Resolution-Adaptive Margin Learning cho Industrial Anomaly Detection

Repository này chứa mã nguồn chính thức của **RAML (Resolution-Adaptive Margin Learning)**, một phương pháp mới cho bài toán phát hiện bất thường công nghiệp kết hợp đặc trưng hình ảnh từ CLIP với adaptive margin learning.

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Kiến trúc](#kiến-trúc)
- [Đóng góp chính](#đóng-góp-chính)
- [Tập dữ liệu](#tập-dữ-liệu)
- [Pre-trained Model](#pre-trained-model)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)
- [Bản quyền](#bản-quyền)

---

## Tổng quan

Bài toán phát hiện bất thường công nghiệp đối mặt với nhiều thách thức: ảnh độ phân giải cao với các lỗi rất nhỏ, đa dạng loại lỗi giữa các danh mục sản phẩm, và số lượng mẫu lỗi có nhãn rất hạn chế. Các phương pháp zero-shot hiện có như WinCLIP thiếu khả năng học decision boundary, trong khi các phương pháp supervised thường sử dụng fixed margin không thích ứng được với đặc điểm từng category.

RAML giải quyết các hạn chế này thông qua:
- Adaptive margin tự điều chỉnh dựa trên phân bố feature và resolution loss
- Multi-scale feature extraction để phát hiện lỗi ở nhiều kích thước khác nhau
- Cross-scale attention để tổng hợp thông tin giữa các scale
- Auto-difficulty weighting để cân bằng training giữa các category

**Tóm tắt kết quả:**
| Đánh giá | AUROC (%) |
|----------|-----------|
| MVTec AD (20% test split) | 94.09 |
| VisA (zero-shot transfer) | 72.44 |

---

## Kiến trúc

### Tổng quan hệ thống

```
Ảnh đầu vào (H x W)
       |
       v
+------------------+
| Multi-Scale      |  Scale 1: 1x1 (global)    -> 1 x 512-dim
| Feature Pyramid  |  Scale 2: 2x2 patches     -> 4 x 512-dim  
|                  |  Scale 3: 4x4 patches     -> 16 x 512-dim
+------------------+
       |
       v (tổng cộng 21 lần forward CLIP)
+------------------+
| Cross-Scale      |  Bidirectional attention giữa các scale
| Attention        |  Kết hợp global <-> local context
+------------------+
       |
       v
+------------------+
| Text-Visual      |  Text prompts: "flawless {category}" vs
| Fusion           |  "damaged {category} with defects"
+------------------+
       |
       v
+------------------+
| MACCL Loss       |  Center Loss + Margin Loss + Contrastive Loss
+------------------+
       |
       v
   Anomaly Score
```

### Multi-Scale Feature Pyramid

CLIP xử lý input cố định 224x224, gây mất mát thông tin đáng kể khi áp dụng cho ảnh công nghiệp độ phân giải cao (thường 700-1024 pixels). Multi-scale pyramid trích xuất features ở ba mức:

| Scale | Patch Grid | Số patches | Effective Resolution |
|-------|------------|------------|---------------------|
| 1 | 1x1 | 1 | 224x224 (global context) |
| 2 | 2x2 | 4 | tương đương 448x448 |
| 3 | 4x4 | 16 | tương đương 896x896 |

Mỗi patch được resize về 224x224 và xử lý qua CLIP, cho ra 21 feature vectors (1 + 4 + 16) mỗi ảnh.

### Cross-Scale Attention

Features từ các scale khác nhau tương tác qua cross-attention:

```
f1' = CrossAttn(f1 -> F2)    # Global attend đến 2x2 patches
f2' = CrossAttn(f2 -> F3)    # 2x2 attend đến 4x4 patches  
f3' = CrossAttn(f3 -> f1)    # 4x4 attend đến global
```

Điều này cho phép local patches hiểu được global context và ngược lại.

### Công thức Resolution-Adaptive Margin

Đóng góp cốt lõi là margin tự thích ứng với cả phân bố dữ liệu và hiệu ứng preprocessing:

```
m = m_base + lambda_sigma * sigma + lambda_resolution * (1 - r/r_max)
```

Trong đó:
- `m_base = 0.5`: margin cơ sở
- `lambda_sigma = 0.3`: trọng số cho độ lệch chuẩn feature
- `lambda_resolution = 0.3`: trọng số cho resolution penalty
- `sigma`: running standard deviation của normal features (cập nhật bằng EMA)
- `r = 224`: resolution đầu vào model
- `r_max = 900`: resolution gốc của ảnh (trung bình MVTec)

**Ví dụ tính toán:**
```
sigma = 0.6 (đo từ normal features)
resolution_penalty = 1 - 224/900 = 0.751

m = 0.5 + 0.3 * 0.6 + 0.3 * 0.751
  = 0.5 + 0.18 + 0.225
  = 0.905
```

### MACCL Loss Function

Margin-Aware Center-Contrastive Loss kết hợp ba thành phần:

```
L_total = alpha * L_center + beta * L_margin + gamma * L_contrastive
```

| Thành phần | Trọng số | Mục đích |
|------------|----------|----------|
| Center Loss | alpha = 1.0 | Kéo normal features về gần center |
| Margin Loss | beta = 1.0 | Đẩy anomaly features ra xa hơn adaptive margin |
| Contrastive Loss | gamma = 0.5 | Tăng khả năng phân biệt normal/anomaly |

### Auto-Difficulty Tracker

Các category có loss cao hơn trung bình sẽ được tăng trọng số:

```
w_cat = 1 + scale * (L_cat - L_global) / L_global
```

Cấu hình: `momentum = 0.7`, `weight_scale = 0.5`, weight được clip trong khoảng [0.5, 2.0].

### Logic Inference

Anomaly score cuối cùng kết hợp visual classifier và text similarity:

```
s_final = 0.85 * s_visual + 0.15 * s_text
```

Trong đó:
- `s_visual`: sigmoid output từ classifier đã train
- `s_text`: chênh lệch cosine similarity giữa anomaly và normal text embeddings

---

## Đóng góp chính

1. **Resolution-Adaptive Margin**: công thức đầu tiên kết hợp feature distribution (sigma) và resolution loss vào tính toán margin
2. **Multi-Scale Feature Pyramid**: trích xuất 21 patches đạt effective resolution 896x896 với backbone 224x224
3. **Cross-Scale Attention**: bidirectional attention fusion giữa các scale
4. **Auto-Difficulty Tracker**: tự động điều chỉnh trọng số loss theo độ khó từng category
5. **Text-Visual Fusion**: residual cross-attention với learnable fusion scale

---

## Tập dữ liệu

### MVTec Anomaly Detection Dataset (Primary)

| Thuộc tính | Giá trị |
|------------|---------|
| Tác giả | P. Bergmann, M. Fauser, D. Sattlegger, C. Steger |
| Công bố | CVPR 2019 |
| Paper | "MVTec AD - A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection" |
| Số ảnh | 5,354 tổng cộng (3,629 train, 1,725 test) |
| Số category | 15 (5 textures, 10 objects) |
| Resolution | 700x700 đến 1024x1024 |
| License | CC BY-NC-SA 4.0 |
| Link | https://www.mvtec.com/company/research/datasets/mvtec-ad |

**Danh sách category:** bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

**Cách chia dữ liệu đã sử dụng:**
- Training: 100% tập train chính thức + 80% tập test chính thức (stratified)
- Testing: 20% tập test chính thức còn lại (held out)

### VisA Dataset (Transfer Evaluation)

| Thuộc tính | Giá trị |
|------------|---------|
| Tác giả | Y. Zou, J. Jeong, L. Pemula, D. Zhang, O. Dabeer |
| Công bố | ECCV 2022 |
| Paper | "SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation" |
| Số ảnh | 10,821 tổng cộng |
| Số category | 12 |
| License | CC BY 4.0 |
| Link | https://github.com/amazon-science/spot-diff |

**Danh sách category:** candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum

**Cách sử dụng:** chỉ dùng để đánh giá zero-shot transfer (model train trên MVTec, đánh giá trên VisA mà không fine-tune)

---

## Pre-trained Model

### CLIP Backbone

| Thuộc tính | Giá trị |
|------------|---------|
| Model | CLIP ViT-B/16 |
| Nhà cung cấp | OpenAI |
| Số parameters | 86M (frozen trong quá trình training) |
| Kích thước input | 224x224 |
| Chiều feature | 512 |
| Link tải | https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ec59a0ca1a1b55/ViT-B-16.pt |

Script tự động tải model qua thư viện `open_clip`. Với môi trường offline, tải thủ công và đặt vào thư mục cache.

---

## Yêu cầu hệ thống

### Phần mềm

| Package | Phiên bản |
|---------|-----------|
| Python | 3.8+ |
| PyTorch | 1.10+ |
| open_clip_torch | mới nhất |
| torchvision | tương thích với PyTorch |
| numpy | 1.20+ |
| scikit-learn | 0.24+ |
| Pillow | 8.0+ |
| tqdm | 4.60+ |

### Phần cứng

| Thành phần | Tối thiểu | Khuyến nghị |
|------------|-----------|-------------|
| GPU | NVIDIA với 8GB VRAM | Tesla T4/P100 với 16GB VRAM |
| RAM | 16GB | 32GB |
| Ổ cứng | 10GB trống | 20GB trống |

---

## Cài đặt

```bash
# clone repository
git clone https://github.com/voveed/RAML_INDUSTRIAL_ANOMALY_DETECTION.git
cd RAML_INDUSTRIAL_ANOMALY_DETECTION

# tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # linux/mac
venv\Scripts\activate     # windows

# cài đặt dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install open_clip_torch scikit-learn pillow tqdm
```

---

## Hướng dẫn sử dụng

### Huấn luyện model RAML (Phase 1)

```bash
python train_raml_model.py
```

Script này sẽ:
- Tải và chuẩn bị MVTec dataset
- Train model RAML trong 25 epochs
- Lưu checkpoint vào `checkpoints/`
- Xuất AUROC scores cho từng category

**Các hyperparameters chính (có thể chỉnh trong script):**

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| epochs | 25 | số epochs training |
| batch_size | 16 | số samples mỗi batch |
| learning_rate | 5e-5 | learning rate cho AdamW |
| weight_decay | 0.02 | L2 regularization |
| margin_base | 0.5 | giá trị margin cơ sở |
| lambda_sigma | 0.3 | hệ số sigma |
| lambda_resolution | 0.3 | hệ số resolution |
| temperature | 0.07 | temperature cho contrastive loss |

### Chạy Ablation Baselines (Phase 2)

```bash
# fixed margin baseline (m=0.5, không adaptive)
python baselines/train_baseline_fixed_margin_05.py

# strong margin baseline (m=1.0, không adaptive)
python baselines/train_baseline_strong_margin_10.py
```

### Đánh giá VisA Transfer (Phase 3)

```bash
python KAGGLE_PHASE_3_VISA.py
```

Đánh giá model đã train trên MVTec với tập VisA mà không fine-tune.

### WinCLIP Zero-shot Baseline

```bash
python evaluate_winclip_baseline.py
```

Chạy WinCLIP zero-shot evaluation để so sánh.

---

## Kết quả thực nghiệm

### Ablation Study: Chiến lược Margin

| Cấu hình | Loại Margin | AUROC (%) |
|----------|-------------|-----------|
| WinCLIP zero-shot | - | 86.98 |
| Fixed Margin (m=0.5) | fixed | 87.99 |
| Strong Margin (m=1.0) | fixed | 86.24 |
| **RAML (adaptive)** | **adaptive** | **94.09** |

### Kết quả theo từng Category (MVTec AD)

| Category | RAML (%) | WinCLIP (%) | Cải thiện |
|----------|----------|-------------|-----------|
| bottle | 100.00 | 94.48 | +5.52 |
| cable | 85.09 | 85.13 | -0.04 |
| capsule | 94.55 | 68.18 | +26.37 |
| carpet | 100.00 | 99.55 | +0.45 |
| grid | 100.00 | 98.39 | +1.61 |
| hazelnut | 91.07 | 95.89 | -4.82 |
| leather | 100.00 | 100.00 | 0.00 |
| metal_nut | 97.89 | 57.79 | +40.10 |
| pill | 89.66 | 73.52 | +16.14 |
| screw | 86.57 | 64.90 | +21.67 |
| tile | 100.00 | 99.78 | +0.22 |
| toothbrush | 83.33 | 91.11 | -7.78 |
| transistor | 84.38 | 86.14 | -1.76 |
| wood | 100.00 | 98.15 | +1.85 |
| zipper | 98.81 | 91.67 | +7.14 |
| **Trung bình** | **94.09** | **86.98** | **+7.11** |

### Kết quả VisA Transfer

| Category | AUROC (%) |
|----------|-----------|
| candle | 85.14 |
| capsules | 58.76 |
| cashew | 56.84 |
| chewinggum | 95.73 |
| fryum | 81.57 |
| macaroni1 | 66.35 |
| macaroni2 | 65.76 |
| pcb1 | 67.19 |
| pcb2 | 59.08 |
| pcb3 | 66.03 |
| pcb4 | 87.72 |
| pipe_fryum | 79.05 |
| **Trung bình** | **72.44** |

---

## Cấu trúc dự án

```
RAML_INDUSTRIAL_ANOMALY_DETECTION/
├── train_raml_model.py              # script training chính
├── evaluate_winclip_baseline.py     # đánh giá winclip zero-shot
├── KAGGLE_PHASE_3_VISA.py           # đánh giá visa transfer
├── visualize_results.py             # visualization kết quả
├── baselines/
│   ├── train_baseline_fixed_margin_05.py   # ablation fixed m=0.5
│   ├── train_baseline_strong_margin_10.py  # ablation fixed m=1.0
│   └── train_baselines.py                  # script baseline tổng hợp
├── checkpoints/
│   ├── model_RAML_Proposed.pt       # model RAML chính (adaptive margin)
│   ├── model_Fixed_Margin_05.pt     # model baseline fixed m=0.5
│   └── model_Strong_Margin_10.pt    # model baseline fixed m=1.0
├── results/
│   ├── result_RAML_Proposed.json    # kết quả raml
│   ├── result_Fixed_Margin_05.json  # kết quả fixed margin
│   ├── result_Strong_Margin_10.json # kết quả strong margin
│   ├── result_WinCLIP_ZeroShot.json # baseline winclip
│   └── result_VisA_Transfer.json    # kết quả visa transfer
└── README.md
```

---

## Tài liệu tham khảo

1. **WinCLIP** - J. Jeong, Y. Zou, T. Kim, D. Zhang, A. Ravichandran, O. Dabeer. "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation." CVPR 2023.

2. **CLIP** - A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, I. Sutskever. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.

3. **MVTec AD** - P. Bergmann, M. Fauser, D. Sattlegger, C. Steger. "MVTec AD - A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." CVPR 2019.

4. **VisA** - Y. Zou, J. Jeong, L. Pemula, D. Zhang, O. Dabeer. "SPot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation." ECCV 2022.

5. **Center Loss** - Y. Wen, K. Zhang, Z. Li, Y. Qiao. "A Discriminative Feature Learning Approach for Deep Face Recognition." ECCV 2016.

6. **Attention** - A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin. "Attention Is All You Need." NeurIPS 2017.

---

## Bản quyền

Mã nguồn được phát hành theo giấy phép MIT License.

Các tập dữ liệu tuân theo license riêng:
- MVTec AD: CC BY-NC-SA 4.0 (chỉ sử dụng phi thương mại)
- VisA: CC BY 4.0
