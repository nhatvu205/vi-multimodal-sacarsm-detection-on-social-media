# ViMMSarc-Fine: Building a Modality-Level Fine-Grained Dataset for Vietnamese Multimodal Sarcasm Detection

Repository for collecting, preprocessing, annotating and experimenting a **Vietnamese multimodal sarcasm detection dataset** named **ViMMSarc-Fine**.

The dataset consists of **(text, image)** social-media posts (Threads/Facebook) with optional **OCR text**, and labels at:
- **Text-level** (`text_label`)
- **Image-level** (`image_label`)
- **Multimodal/post-level** (`mm_label`)

This repo also contains:
- Two-round LLM-assisted annotation pipelines (**coarse-to-fine**)
- Experimental baselines 

---

## Dataset statistics (ViMMSarc-Fine)

### Overview
- Final size (after cleaning + 2-round annotation): **7,355** posts
- Sources: **Threads 4,456 (60.58%)**, **Facebook 2,899 (39.42%)**
- OCR availability: **4,941 / 7,355 (67.18%)** samples have `ocr_text`
- Each record includes: `text` + `image_path` (+ optional `ocr_text`)
- Labels:
  - `text_label`: sarcasm from text alone
  - `image_label`: sarcasm from image alone
  - `mm_label`: multimodal/post-level sarcasm (default target label in experiments)

### Fixed splits
| Split | #Samples | Threads/Facebook (%) |
|---|---:|---:|
| Train | 5,884 | 60.54 / 39.46 |
| Dev | 735 | 61.09 / 38.91 |
| Test | 736 | 60.46 / 39.54 |

### Label distribution by split
| Split | `mm_label` (0/1) | `text_label` (0/1) | `image_label` (0/1) |
|---|---:|---:|---:|
| Train | 3,305 / 2,579 | 4,882 / 1,002 | 5,197 / 687 |
| Dev | 413 / 322 | 610 / 125 | 649 / 86 |
| Test | 414 / 322 | 611 / 125 | 650 / 86 |

### Most common positive (MM=1) label combinations
When `mm_label=1`, the distribution over representative (T,I,MM) combinations is:
| Combo (T,I,MM) | Count | % within MM=1 |
|---|---:|---:|
| (0,0,1) | 1,602 | 49.71% |
| (1,0,1) | 859 | 26.65% |
| (0,1,1) | 629 | 19.52% |
| (1,1,1) | 133 | 4.13% |

---

## Experimental setup

### Task and target label
- We evaluate models on the **post-level multimodal sarcasm label**: `mm_label`.
- Splits are fixed to **Train/Dev/Test = 80/10/10** (see dataset statistics above).

### Ablation design (text branch only)
We use a 2×2 ablation design that changes only **text preprocessing** and **emoji handling**, while keeping the image branch unchanged:
- **s1**: raw text, emoji kept
- **s2**: raw text, emoji removed
- **s3**: preprocessed text, emoji kept
- **s4**: preprocessed text, emoji removed

### Model families
- **Text-only** encoders: RoBERTa-base, PhoBERT-base, mBERT (trained/evaluated under s1–s4)
- **Image-only** classifiers: ViT-B/32, CLIP ViT-L/14 (evaluated in **s1 only** since images are unchanged across scenarios)
- **Multimodal** models: DT4MID, CIRM, LLaVA-1.6-7B, Qwen3-VL-8B, and 3 additional `sarcasm_detection` baselines (evaluated under s1–s4 where applicable)

### Additional integrated baselines
The repo now also exposes the 3 modeling approaches from `sarcasm_detection/` through the unified `experiment_setup/` pipeline:
- **Multimodal Fusion**: direct multimodal fusion (PhoBERT/LM + CLIP image encoder → MLP, 4-way classification)
- **Staged Gating**: 3-phase pipeline (text binary branch + multimodal binary branch + gating network, 4-way classification)
- **Hierarchical Cross-Attention**: hierarchical multimodal model (binary multi-sarcasm stage + 3-way cross-attention stage)

Because the main dataset in this repo is stored as modality labels (`mm_label`, `text_label`, `image_label`) rather than the original 4-class target used by `sarcasm_detection/`, `experiment_setup` derives a compatible 4-way label during training/evaluation:
- `0` = non-sarcasm (`MM=0`)
- `1` = multi-sarcasm / interaction-driven sarcasm (all remaining `MM=1` cases not assigned below)
- `2` = text-sarcasm (`MM=1, T=1, I=0`)
- `3` = image-sarcasm (`MM=1, T=0, I=1`)

This keeps the methods runnable inside the current repo, but it is an adaptation of the original setup rather than a byte-for-byte reproduction.

### Metrics
- Primary: **Accuracy**, **F1-macro**
- (Optional) **AUC** is reported only when a model provides probabilistic outputs.

### Key hyperparameters (summary)
These are the main settings used in the report:
- Max epochs: **10**
- Early stopping: by **weighted F1** on Dev, patience = **2**
- Learning rate: **2e-5**
- Max text length: **256**
- Batch size (train/eval): **4 / 8** (default for trained models)
- Generative multimodal inference: `temperature=0.0`

For the full configuration table, see the report appendix:
- `report/acl_latex.tex` (Vietnamese)
- `report/acl_latex_en.tex` (English)

---

## Baseline results (Accuracy / F1-macro)

Baselines are evaluated under four text *ablation* scenarios:
- **s1**: raw text, emoji kept
- **s2**: raw text, emoji removed
- **s3**: preprocessed text, emoji kept
- **s4**: preprocessed text, emoji removed

### Text-only models
| Model | s1 | s2 | s3 | s4 |
|---|---|---|---|---|
| RoBERTa-base | 0.6685 / 0.6627 | 0.6712 / 0.6680 | 0.6726 / 0.6660 | 0.6726 / 0.6638 |
| **PhoBERT-base** | **0.6848 / 0.6838** | **0.6848 / 0.6838** | 0.6821 / 0.6798 | 0.6793 / 0.6776 |
| mBERT | 0.6671 / 0.6650 | 0.6386 / 0.6385 | 0.6576 / 0.6353 | 0.6399 / 0.6368 |

### Image-only models (evaluated in s1 only)
| Model | s1 |
|---|---|
| ViT-B/32 | 0.5666 / 0.5666 |
| **CLIP ViT-L/14** | **0.6603 / 0.6456** |

### Multimodal models
| Model | s1 | s2 | s3 | s4 |
|---|---|---|---|---|
| **DT4MID** | 0.6929 / 0.6864 | 0.6848 / 0.6841 | 0.6807 / 0.6730 | **0.6943 / 0.6884** |
| CIRM | 0.6318 / 0.6292 | 0.4375 / 0.3043 | 0.4375 / 0.3043 | 0.6128 / 0.5943 |
| LLaVA-1.6-7B | 0.5639 / 0.3634 | 0.5639 / 0.3634 | 0.5639 / 0.3634 | 0.5639 / 0.3634 |
| Qwen3-VL-8B | 0.5734 / 0.5734 | 0.5910 / 0.5908 | 0.5734 / 0.5734 | 0.5910 / 0.5908 |

For additional metrics (e.g., AUC where available), see:
- `report/acl_latex.tex` (Vietnamese)
- `report/acl_latex_en.tex` (English)

---

## Repository layout (high level)

| Path | Description |
|---|---|
| `data/` | Processed dataset JSONs + images (images not committed; see `data/README.md`). |
| `data_collection/` | Crawling/collection utilities (Threads/Facebook). |
| `preprocessing/` | Cleaning, normalization, deduplication, OCR mapping, etc. |
| `experiment_setup/` | Unified config-driven experiment runner for text, image, multimodal, VLM, and integrated `sarcasm_detection` baselines. |
| `round-1-annotation/` | Round-1 pipeline: binary sarcasm annotation (LLM-as-annotator). |
| `round-2-annotation/` | Round-2 pipeline: modality-level fine-grained annotation (LLM-as-annotator). |
| `sarcasm_detection/` | Original standalone implementations of 3 additional multimodal baselines later adapted into `experiment_setup/`. |

---

## Quick start

### 1) Environment
Create a Python environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Note: Some scripts (e.g., plot generation for the report) require `matplotlib`.

### 2) Dataset files
See `data/README.md` for how to obtain images and where to place them locally.

### 3) Unified experiment runner
Install experiment dependencies:
```bash
pip install -r experiment_setup/requirements.txt
```

Example runs for the integrated `sarcasm_detection` methods:
```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/sarcasm_detection_multimodal_fusion.yaml \
  --stage all \
  --scenario s1
```

```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/sarcasm_detection_staged_gating.yaml \
  --stage all \
  --scenario s1
```

```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/sarcasm_detection_hierarchical_cross_attention.yaml \
  --stage all \
  --scenario s1
```

---

## Annotation pipelines (LLM-as-annotator)

The annotation flow is separated into two independent pipelines:

### Round 1 — binary sarcasm
- Goal: label each post as **sarcastic vs. non-sarcastic**.
- Location: `round-1-annotation/`
- Prompt: `round-1-annotation/prompts/prompt.txt`

### Round 2 — modality-level fine-grained
- Goal: determine sarcasm sources across modalities (e.g., **T/I/MM**).
- Location: `round-2-annotation/`
- Prompt: `round-2-annotation/prompts/prompt.txt`

Operational details, environment variables, and key scripts are documented in `AGENT.md`.
