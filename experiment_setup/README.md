# experiment_setup

Config-driven end-to-end pipeline for sarcasm experiments using the fixed split in `data/final-data/`.

## Main features
- shared cache built from `train.json`, `dev.json`, `test.json`
- one config format for text-only, image-only, and multimodal models
- built-in ablation scenarios from `experiment_guide.md`
- reusable metrics, logging, predictions, checkpoints, and summaries
- support for supervised HF classifiers and generative VLM classifiers

## Scenarios
- `s1`: raw text + raw image
- `s2`: preprocessed text + raw image
- `s3`: raw text + preprocessed image
- `s4`: preprocessed text + preprocessed image

Notes:
- text-only models still run all listed scenarios, but only the text branch changes.
- image-only models should normally use `s1` only.

## Directory layout
```
experiment_setup/
  configs/
    base.yaml
    models/*.yaml
  src/
    ...
  runs/
    <experiment_name>/
      resolved_config.yaml
      cache/
      reports/
      <model_name>/<scenario>/
```

## Install
Use the dedicated requirements file for experiments:

```bash
pip install -r experiment_setup/requirements.txt
```

## Run
### 1) Build shared cache only
```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/base.yaml \
  --stage preprocess
```

### 2) Run a text-only model
```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/phobert_base.yaml \
  --stage all
```

### 3) Run a supervised MMSD3/CIRM-style multimodal model
This pipeline now includes a single-image adaptation of the MMSD3.0 CIRM method:
- core `SequentialModeling` and `DualStageBridgeModule` are integrated from the official MMSD3.0 implementation, so no extra repo checkout is required
- the wrapper is adapted for this project: single image, no OCR, binary sarcasm classification

PhoBERT-based CIRM:
```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/cirm_phobert.yaml \
  --stage all \
  --scenario s1
```

mBERT-based CIRM:
```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/cirm_mbert.yaml \
  --stage all \
  --scenario s1
```

Notes:
- unlike LLaVA/Qwen zero-shot, CIRM is a supervised model and will train by default
- if you only want final evaluation on `test`, pass `--eval_splits test`; training still uses `train` and early stopping still checks `dev` internally

### 3) Run a multimodal zero-shot model
LLaVA-NeXT Mistral 7B:
```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/llava_next_mistral_7b.yaml \
  --stage all \
  --scenario s1
```

Qwen3-VL 8B:
```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/qwen3_vl_8b.yaml \
  --stage all \
  --scenario s1
```

### 4) Run each ablation scenario separately
Inference now supports a runtime `--scenario` argument:
- `--scenario s1`
- `--scenario s2`
- `--scenario s3`
- `--scenario s4`
- `--scenario all`

Example commands for LLaVA-NeXT:
```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/llava_next_mistral_7b.yaml \
  --stage all \
  --scenario s1
```

```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/llava_next_mistral_7b.yaml \
  --stage all \
  --scenario s2
```

```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/llava_next_mistral_7b.yaml \
  --stage all \
  --scenario s3
```

```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/llava_next_mistral_7b.yaml \
  --stage all \
  --scenario s4
```

Example commands for Qwen3-VL:
```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/qwen3_vl_8b.yaml \
  --stage all \
  --scenario s1
```

```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/qwen3_vl_8b.yaml \
  --stage all \
  --scenario s2
```

```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/qwen3_vl_8b.yaml \
  --stage all \
  --scenario s3
```

```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/qwen3_vl_8b.yaml \
  --stage all \
  --scenario s4
```

Run only on the test split:

```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/qwen3_vl_8b.yaml \
  --stage all \
  --scenario s1 \
  --eval_splits test \
  --json_splits /kaggle/input/your-dataset/final-data/train.json /kaggle/input/your-dataset/final-data/dev.json /kaggle/input/your-dataset/final-data/test.json \
  --image_root /kaggle/input/your-dataset
```

Run all four scenarios in one command:
```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/qwen3_vl_8b.yaml \
  --stage all \
  --scenario all
```

## Outputs
Each run saves:
- resolved config
- cached split manifests
- dataset report
- per-scenario predictions (`jsonl`)
- per-scenario metrics (`json`)
- per-model summary (`json` + `csv`)
- checkpoints for trainable models

## Recommended configs
The guide mentions `Qwen-VL-Chat`. That config is included for compatibility. If you want a stronger newer replacement, add a config using the same pipeline for `Qwen2-VL` or `Qwen2.5-VL`.


## Runtime path overrides
You can override the dataset JSON paths and image root from the command line:

```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/qwen3_vl_8b.yaml \
  --stage all \
  --scenario s1 \
  --json_splits /kaggle/input/.../train.json /kaggle/input/.../dev.json /kaggle/input/.../test.json \
  --image_root /kaggle/input/.../
```

## Multimodal QLoRA fine-tuning
Separate entrypoint for multimodal QLoRA fine-tuning:

```bash
python -m experiment_setup.cli.finetune_mm \
  --config experiment_setup/configs/finetune/llava_next_mistral_7b_qlora.yaml \
  --scenario s1 \
  --json_splits /kaggle/input/.../train.json /kaggle/input/.../dev.json /kaggle/input/.../test.json \
  --image_root /kaggle/input/.../
```

```bash
python -m experiment_setup.cli.finetune_mm \
  --config experiment_setup/configs/finetune/qwen3_vl_8b_qlora.yaml \
  --scenario s1 \
  --json_splits /kaggle/input/.../train.json /kaggle/input/.../dev.json /kaggle/input/.../test.json \
  --image_root /kaggle/input/.../
```

The fine-tune command saves a LoRA adapter under:
- `experiment_setup/runs/<experiment_name>/finetune/<model_key>/<scenario>/adapter`
- intermediate checkpoints under:
  - `experiment_setup/runs/<experiment_name>/finetune/<model_key>/<scenario>/checkpoint-*`

The default QLoRA configs are tuned for faster Kaggle runs:
- shorter finetune prompt
- `num_train_epochs: 1`
- LoRA target modules reduced to `q_proj`, `v_proj`
- checkpoint saving every `100` update steps
- auto-resume from the latest checkpoint

To evaluate a fine-tuned adapter with the normal inference pipeline, set `model.adapter_path` in the model config.

Resume manually from a specific checkpoint if needed:
```bash
python -m experiment_setup.cli.finetune_mm \
  --config experiment_setup/configs/finetune/llava_next_mistral_7b_qlora.yaml \
  --scenario s1 \
  --resume_from_checkpoint /kaggle/working/.../checkpoint-100
```

### Evaluate with a saved adapter
```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/llava_next_mistral_7b.yaml \
  --stage run \
  --json_splits /kaggle/input/.../train.json /kaggle/input/.../dev.json /kaggle/input/.../test.json \
  --image_root /kaggle/input/.../
```

Then add this field into the model config before running:
```yaml
model:
  adapter_path: /kaggle/working/.../adapter
```
