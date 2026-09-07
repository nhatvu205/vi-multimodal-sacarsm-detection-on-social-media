# Camera-ready experiments on Kaggle

This runbook executes the OCR and cross-platform experiments on Kaggle using the
fixed dataset and a T4 x2 accelerator. It produces self-contained output folders
that can be attached to the final aggregation notebook.

## Notebook setup

Create a Kaggle notebook with Internet and T4 x2 enabled. Run this cell first:

```bash
REPO=/kaggle/working/vimmsarc
set -euo pipefail
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" checkout nvu
  git -C "$REPO" pull --ff-only origin nvu
else
  git clone --branch nvu https://github.com/nhatvu205/vi-multimodal-sacarsm-detection-on-social-media.git "$REPO"
fi
cd "$REPO"
pip install -q -r experiment_setup/requirements-kaggle.txt
export HF_HOME=/kaggle/working/hf-cache
export TOKENIZERS_PARALLELISM=false
```

The data paths are fixed for this runbook:

```bash
DATA_ROOT=/kaggle/input/datasets/nhatvu205/sacasm-dataset-uit/final-data
IMAGE_ROOT=/kaggle/input/datasets/nhatvu205/sacasm-dataset-uit/images
OUT_ROOT=/kaggle/working/camera_ready
```

Before launching a model, run this no-GPU validation once. It reads all split
records and confirms every referenced image exists:

```bash
python -m experiment_setup.main \
  --config experiment_setup/configs/camera_ready/ocr_phobert_caption.yaml \
  --stage preprocess \
  --json_splits "$DATA_ROOT/train.json" "$DATA_ROOT/dev.json" "$DATA_ROOT/test.json" \
  --image_root "$IMAGE_ROOT" \
  --output_root "$OUT_ROOT" \
  --run_name validation/preprocess
```

Check `validation/preprocess/reports/dataset_report.json`: train/dev/test must
contain 5,884/735/736 samples. The image resolver accepts this image-directory
root directly; it also remains compatible with an older dataset-root path.

## Run a job

Use one process per GPU. The following helper records command, commit, package
versions, GPU details, and the resolved config in the run directory.

```bash
run_job() {
  local gpu="$1"
  local config="$2"
  local run_name="$3"
  local seed="$4"
  local command="python -m experiment_setup.main --config $config --stage all --seed $seed --output_root $OUT_ROOT --run_name $run_name"

  CUDA_VISIBLE_DEVICES="$gpu" python -m experiment_setup.main \
    --config "$config" \
    --stage all \
    --seed "$seed" \
    --output_root "$OUT_ROOT" \
    --run_name "$run_name" \
    --json_splits "$DATA_ROOT/train.json" "$DATA_ROOT/dev.json" "$DATA_ROOT/test.json" \
    --image_root "$IMAGE_ROOT"

  python -m experiment_setup.camera_ready.write_manifest \
    --output_dir "$OUT_ROOT/$run_name" \
    --command "$command"
}

run_three_seeds() {
  local config="$1"
  local study="$2"
  run_job 0 "$config" "$study/seed-42" 42 &
  run_job 1 "$config" "$study/seed-123" 123 &
  wait
  run_job 0 "$config" "$study/seed-2026" 2026
}
```

Run OCR studies in two notebook versions to keep their outputs distinct:

```bash
run_three_seeds experiment_setup/configs/camera_ready/ocr_phobert_caption.yaml ocr/phobert-caption
run_three_seeds experiment_setup/configs/camera_ready/ocr_phobert_ocr.yaml ocr/phobert-ocr
run_three_seeds experiment_setup/configs/camera_ready/ocr_phobert_caption_ocr.yaml ocr/phobert-caption-ocr
```

```bash
run_three_seeds experiment_setup/configs/camera_ready/ocr_dt4mid_caption.yaml ocr/dt4mid-caption
run_three_seeds experiment_setup/configs/camera_ready/ocr_dt4mid_caption_ocr.yaml ocr/dt4mid-caption-ocr
```

Run the four cross-platform studies in a third notebook version:

```bash
run_three_seeds experiment_setup/configs/camera_ready/platform_phobert_facebook.yaml platform/phobert-facebook-source
run_three_seeds experiment_setup/configs/camera_ready/platform_phobert_threads.yaml platform/phobert-threads-source
run_three_seeds experiment_setup/configs/camera_ready/platform_dt4mid_facebook.yaml platform/dt4mid-facebook-source
run_three_seeds experiment_setup/configs/camera_ready/platform_dt4mid_threads.yaml platform/dt4mid-threads-source
```

Each platform configuration trains and selects only from its source platform.
It emits `source_test` and `target_test` predictions from the same checkpoint.

Save each notebook version after its jobs finish. Its `/kaggle/working/camera_ready`
directory must be retained as an attachable notebook output for aggregation.

## Aggregate results

Create a fourth notebook, attach the outputs from the OCR and platform notebooks,
then clone the same Git commit as above. Run:

```bash
python -m experiment_setup.camera_ready.analyze \
  --runs_root /kaggle/input \
  --output_dir /kaggle/working/camera_ready_analysis \
  --bootstrap_iterations 10000
```

The command fails if predictions have duplicate IDs, missing run metadata, or a
gold label inconsistent with `mm_label`. It writes:

- `seed_metrics.csv`: full-test and OCR-present/OCR-absent metrics per seed with bootstrap 95% CI.
- `seed_summary.csv`: F1-macro mean and standard deviation over seeds.
- `error_slices.csv`: support, FP/FN, FPR and FNR by label tuple, OCR and platform.
- `candidate_examples.csv`: internal IDs for `(0,0,1)` errors, false positives,
  correct predictions, and OCR-improves/OCR-harms cases.
- `validation.json`: artifact validation result.

Review and anonymize selected captions and images before moving examples into the paper.
