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

### 3) Run a multimodal zero-shot model
LLaVA-NeXT Mistral 7B:
```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/llava_next_mistral_7b.yaml \
  --stage all
```

Qwen3-VL 8B:
```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/qwen3_vl_8b.yaml \
  --stage all
```

### 4) Run each ablation scenario separately
The pipeline reads scenarios from the config file.  
Fastest way: create one config per scenario, or temporarily edit:

```yaml
run:
  scenarios: [s1]
```

Then repeat with `s2`, `s3`, `s4`.

Example commands for LLaVA-NeXT:
```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/llava_next_mistral_7b.yaml \
  --stage all
```

Example commands for Qwen3-VL:
```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/qwen3_vl_8b.yaml \
  --stage all
```

Recommended scenario-specific config names:
- `experiment_setup/configs/models/llava_next_mistral_7b_s1.yaml`
- `experiment_setup/configs/models/llava_next_mistral_7b_s2.yaml`
- `experiment_setup/configs/models/llava_next_mistral_7b_s3.yaml`
- `experiment_setup/configs/models/llava_next_mistral_7b_s4.yaml`
- `experiment_setup/configs/models/qwen3_vl_8b_s1.yaml`
- `experiment_setup/configs/models/qwen3_vl_8b_s2.yaml`
- `experiment_setup/configs/models/qwen3_vl_8b_s3.yaml`
- `experiment_setup/configs/models/qwen3_vl_8b_s4.yaml`

Template for a single-scenario config:
```yaml
extends: ../base.yaml

experiment:
  name: llava_next_mistral_7b_s1

run:
  scenarios: [s1]

model:
  key: llava-next-mistral-7b
  family: llava_next_generative
  pretrained_name: llava-hf/llava-v1.6-mistral-7b-hf

training:
  enabled: false
```

Then run:
```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/models/llava_next_mistral_7b_s1.yaml \
  --stage all
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

To evaluate a fine-tuned adapter with the normal inference pipeline, set `model.adapter_path` in the model config.

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
