# Round-1 Weak Annotation Pipeline

Pipeline gán nhãn Round-1 chạy local bằng **Gemma 4 31B qua Gemini API**.

## Setup

```bash
cd round-1-annotation
pip install -r requirements.txt
```

Tạo `.env` ở repo root hoặc trong `round-1-annotation/`:

```env
GEMINI_API_KEY=your_api_key
```

## Chạy test_mode

```bash
python3 -m src.pipeline_round1 \
  --config configs/round1.yaml \
  --output_dir outputs/test_run \
  --test_mode \
  --no-checkpoint-load
```

## Chạy full

```bash
python3 -m src.pipeline_round1 \
  --config configs/round1.yaml \
  --output_dir outputs/full_run
```

## Hành vi hiện tại

- Input mặc định: `data/bronze/raw_data.json`
- `test_mode`: lấy 5 record đầu tiên theo thứ tự
- Async parallel với `concurrency: 4` mặc định
- Checkpoint sau mỗi `checkpoint_every: 10` sample hoàn thành
- Retry tối đa 3 lần, nghỉ 5 giây giữa các lần retry
- Nếu fail sau retries: `label_llm1 = -1`
- Chỉ ghi **1 file output**: `round1_results.jsonl`
- File này cũng là checkpoint để resume ở lần chạy sau
- Log tiến độ chính hiển thị bằng progress bar
