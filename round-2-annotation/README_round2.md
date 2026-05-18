# Round-2 Fine-grained Annotation Pipeline

Pipeline Round-2 chạy độc lập trong `round-2-annotation/`.

## Model hỗ trợ
- `gemma` → `gemma-4-31b-it` qua Gemini API
- `nemotron` → `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` qua OpenRouter, bật reasoning

## Chuẩn bị môi trường

```bash
cd /mnt/e/uit/nam-3/ki-2/social-media-mining
source .venv/bin/activate
cd round-2-annotation
pip install -r requirements.txt
```

Tạo `.env` ở repo root hoặc trong `round-2-annotation/`:

```env
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
```

## Input / Output
- Input mặc định: `data/bronze/raw_data.json`
- OCR mặc định: `data/bronze/ocr_images.json`
- Output checkpoint: `output_dir/round2_results.jsonl`
- Output JSON dễ đọc: `output_dir/round2_results.json`

Pipeline sẽ ghi lại **cả JSONL và JSON** sau mỗi lần đủ `checkpoint_every` sample hoàn thành, và ghi lần cuối khi run xong.

## Các option quan trọng
- `--model {gemma,nemotron}`: chọn VLM
- `--test_mode`: chạy số lượng nhỏ theo thứ tự từ trên xuống
- `--test_size N`: số record khi test mode
- `--max_records N`: chỉ lấy N record đầu sau khi lọc
- `--from N`: chỉ chạy các record có `id >= N`
- `--ocr_path PATH`: override file OCR
- `--no-checkpoint-load`: bỏ resume, chạy mới hoàn toàn

## Chạy test với Gemma

```bash
python3 -m src.pipeline_round2 \
  --config configs/round2.yaml \
  --output_dir outputs/test_gemma \
  --model gemma \
  --test_mode \
  --test_size 5 \
  --no-checkpoint-load
```

## Chạy test với Nemotron

```bash
python3 -m src.pipeline_round2 \
  --config configs/round2.yaml \
  --output_dir outputs/test_nemotron \
  --model nemotron \
  --test_mode \
  --test_size 5 \
  --no-checkpoint-load
```

## Chạy từ một id nhất định

Ví dụ chạy từ `id = 1001` trở đi:

```bash
python3 -m src.pipeline_round2 \
  --config configs/round2.yaml \
  --output_dir outputs/from_1001 \
  --model gemma \
  --from 1001
```

## Resume một run đang dở

Chỉ cần chạy lại cùng `output_dir` và **không** truyền `--no-checkpoint-load`:

```bash
python3 -m src.pipeline_round2 \
  --config configs/round2.yaml \
  --output_dir outputs/from_1001 \
  --model gemma
```

## Hành vi hiện tại
- Không dùng chung source code với Round-1
- Async parallel với `concurrency: 4`
- Checkpoint sau mỗi `checkpoint_every: 10` sample hoàn thành
- Retry thông minh với exponential backoff + jitter
- Nếu fail sau retries: `label_llm2 = -1`
- Log tiến độ theo `id` đang xử lý
