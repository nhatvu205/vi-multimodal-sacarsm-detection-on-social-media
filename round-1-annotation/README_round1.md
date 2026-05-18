# Round-1 Weak Annotation Pipeline

Pipeline Round-1 chạy local terminal trong `round-1-annotation/`.

## Model hỗ trợ
- `gemma` → `gemma-4-31b-it` qua Gemini API
- `nemotron` → `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` qua OpenRouter, bật reasoning

## Chuẩn bị môi trường
Tạo venv trước khi chạy: python -m venv .venv
```bash
source .venv/bin/activate
cd round-1-annotation
pip install -r requirements.txt # chỉ cần cài 1 lần khi tạo .venv
```

Tạo `.env` ở repo root hoặc trong `round-1-annotation/`:

```env
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEYS=key_1,key_2,key_3
# fallback nếu chỉ có 1 key:
OPENROUTER_API_KEY=your_openrouter_key
```

## Input / Output
- Input mặc định: `data/raw-data/raw_data.json`
- Output checkpoint: `output_dir/round1_results.jsonl`
- Output JSON dễ đọc: `output_dir/round1_results.json`

Pipeline sẽ ghi lại **cả JSONL và JSON** sau mỗi lần đủ `checkpoint_every` sample hoàn thành, và ghi lần cuối khi run xong.

## Các option quan trọng
- `--model {gemma,nemotron}`: chọn VLM
- `--test_mode`: chạy số lượng nhỏ theo thứ tự từ trên xuống
- `--test_size N`: số record khi test mode
- `--max_records N`: chỉ lấy N record đầu sau khi lọc
- `--from N`: chỉ chạy các record có `id >= N`
- `--no-checkpoint-load`: bỏ resume, chạy mới hoàn toàn

## Chạy test với Gemma

```bash
python3 -m src.pipeline_round1 \
  --config configs/round1.yaml \
  --output_dir outputs/test_gemma \
  --model gemma \
  --test_mode \
  --test_size 5 \
  --no-checkpoint-load
```

## Chạy test với Nemotron

```bash
python3 -m src.pipeline_round1 \
  --config configs/round1.yaml \
  --output_dir outputs/test_nemotron \
  --model nemotron \
  --test_mode \
  --test_size 5 \
  --no-checkpoint-load
```

## Chạy từ một id nhất định

Ví dụ chạy từ `id = 1001` trở đi:

```bash
python3 -m src.pipeline_round1 \
  --config configs/round1.yaml \
  --output_dir outputs/from_1001 \
  --model gemma \
  --from 1001
```

## Resume một run đang dở

Chỉ cần chạy lại cùng `output_dir` và **không** truyền `--no-checkpoint-load`:

```bash
python3 -m src.pipeline_round1 \
  --config configs/round1.yaml \
  --output_dir outputs/from_1001 \
  --model gemma
```

## Hành vi hiện tại
- Async parallel với `concurrency: 4`
- Checkpoint sau mỗi `checkpoint_every: 10` sample hoàn thành
- Retry thông minh với exponential backoff + jitter
- Nếu fail sau retries: `label_llm1 = -1`
- Log tiến độ theo `id` đang xử lý
