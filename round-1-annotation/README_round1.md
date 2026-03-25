# Round-1 Weak Annotation Pipeline

Sinh nhãn yếu (silver labels) cho bài toán phát hiện mỉa mai đa phương thức tiếng Việt bằng phương pháp **LLM-as-a-judge** sử dụng mô hình **Qwen2.5-VL-7B-Instruct** chạy cục bộ trên GPU.

> Pipeline được thiết kế để chạy trên **Kaggle Notebooks** (GPU T4/P100 miễn phí).

> Nhãn đầu ra là **silver labels** — nhãn tạm thời, chưa phải gold labels. Round-2 sẽ có bước kiểm định bởi con người.

---

## Tổng quan kiến trúc

```
data/data.json  -->  pipeline_round1.py  -->  llm_judge.py (Qwen2.5-VL local)
                          |                          |
                    fusion_router.py          round1_all.jsonl
                          |                  round1_auto_accepted.jsonl
                    round1_stats.json         round1_human_queue.jsonl
```

Toàn bộ quá trình inference chạy **cục bộ trên GPU Kaggle** — không gọi API bên ngoài.

---

## Chuẩn bị trước khi chạy

### Bước 1 — Push repo lên GitHub

Đảm bảo toàn bộ code (không bao gồm data và `.venv`) đã được push:

```bash
git add .
git commit -m "round-1: migrate to local inference"
git push
```

File `.gitignore` đã loại trừ `.venv/`, `outputs/`, `hf_cache/`, `*.safetensors`.

### Bước 2 — Upload data lên Kaggle Dataset

Data (ảnh + JSON) **không** nằm trong repo do kích thước lớn. Cần upload thủ công lên Kaggle:

1. Vào [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. Upload toàn bộ thư mục `data/` (gồm `data.json`, `data-sample.json`, `images/`, `images-sample/`)
3. Đặt tên dataset (ví dụ: `visomd-data`) và tạo dataset ở chế độ **Private**
4. Ghi nhớ tên dataset — sẽ dùng ở bước 5

### Bước 3 — Tạo Kaggle Notebook mới

1. Vào [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. **Settings > Accelerator**: chọn **GPU T4 x1**
3. **Settings > Internet**: bật **On**
4. Upload file `round-1-annotation/notebooks/kaggle_run_round1.ipynb` lên notebook (hoặc copy nội dung)

### Bước 4 — Thêm HF_TOKEN vào Kaggle Secrets

HuggingFace token dùng để download mô hình Qwen2.5-VL-7B-Instruct. Mô hình này là public nên token không bắt buộc, nhưng nên cài để tránh rate limit.

1. Trong notebook → **Add-ons > Secrets > Add a new secret**
2. Name: `HF_TOKEN`
3. Value: token HuggingFace của bạn (lấy tại [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

### Bước 5 — Attach Kaggle Dataset vào notebook

1. Trong notebook → **Add Data** (biểu tượng +)
2. Tìm dataset vừa tạo ở Bước 2 và thêm vào
3. Dataset sẽ được mount tại `/kaggle/input/TEN_DATASET/`
4. Mở cell **5** trong notebook, sửa biến `KAGGLE_DATASET_NAME` thành tên dataset của bạn

---

## Chạy notebook

Mở file `round-1-annotation/notebooks/kaggle_run_round1.ipynb` và chạy từng cell theo thứ tự:

| Cell | Mô tả |
|------|-------|
| 1 | Kiểm tra GPU (`nvidia-smi`) |
| 2 | `git clone` repo từ GitHub vào `/kaggle/working/` |
| 3 | Cài đặt dependencies từ `requirements.txt` |
| 4 | Load `HF_TOKEN` từ Kaggle Secrets, set `HF_HOME` |
| 5 | Tạo symlink `data/` -> `/kaggle/input/TEN_DATASET/` |
| 6 | Chạy pipeline trên `data-sample.json` (kiểm tra nhanh) |
| 7 | Chạy pipeline trên `data.json` (toàn bộ dữ liệu) |
| 8 | Xem thống kê kết quả (`round1_stats.json`) |
| 9 | Nén output thành `.zip` để download |

> **Lưu ý về thời gian chạy:** Qwen2.5-VL-7B-Instruct với 4-bit quantization xử lý khoảng 2–5 giây/record. Kaggle giới hạn session GPU 9 giờ/tuần — hãy kiểm tra bằng data-sample trước khi chạy toàn bộ.

---

## Cấu hình (`configs/round1.yaml`)

| Trường | Mặc định | Mô tả |
|--------|----------|-------|
| `llm_model` | `Qwen/Qwen2.5-VL-7B-Instruct` | HuggingFace model ID |
| `llm_temperature` | `0.1` | Nhiệt độ sampling |
| `device` | `cuda` | Thiết bị chạy inference |
| `load_in_4bit` | `true` | Bật 4-bit quantization (tiết kiệm VRAM) |
| `batch_size` | `8` | Số record mỗi lần lưu checkpoint |
| `sarcastic_high` | `0.85` | `llm_prob_sarcastic >= này` → tự động nhãn sarcastic |
| `nonsarcastic_low` | `0.15` | `llm_prob_sarcastic <= này` → tự động nhãn non_sarcastic |
| `conf_threshold` | `0.70` | Ngưỡng confidence tối thiểu để chấp nhận tự động |
| `random_audit_rate` | `0.10` | Tỉ lệ record auto-accepted được sample lại để QA |
| `seed` | `42` | Random seed |

### Bộ nhớ GPU

| Chế độ | VRAM cần (~7B params) |
|--------|----------------------|
| `load_in_4bit: true` | ~4–5 GB |
| `load_in_4bit: false` (float16) | ~14–15 GB |

Kaggle T4 có 16 GB VRAM — nên bật `load_in_4bit: true`.

---

## Input schema

```json
{
  "id": 0,
  "text": "Nội dung bài đăng tiếng Việt",
  "image_path": "data/images-sample/post0097_img01.jpg"
}
```

Hỗ trợ: JSON array (`[...]`) hoặc JSONL (mỗi dòng một record).

---

## Chính sách routing

Áp dụng theo thứ tự:

1. Ảnh bị thiếu → `needs_human_review` | `missing_image`
2. LLM trả về JSON lỗi sau retry → `needs_human_review` | `invalid_json`
3. LLM trả về nhãn `uncertain` → `needs_human_review` | `uncertain`
4. `llm_confidence < conf_threshold` → `needs_human_review` | `low_conf`
5. `llm_prob_sarcastic >= sarcastic_high` → **sarcastic** | `high_conf`
6. `llm_prob_sarcastic <= nonsarcastic_low` → **non_sarcastic** | `high_conf`
7. Còn lại → `needs_human_review` | `low_conf`
8. `random_audit_rate` trong số auto-accepted → `needs_human_review` | `audit_sampled`

---

## Các file đầu ra

| File | Mô tả |
|------|-------|
| `round1_all.jsonl` | Toàn bộ record đã xử lý |
| `round1_auto_accepted.jsonl` | Record được tự động nhãn `sarcastic` / `non_sarcastic` |
| `round1_human_queue.jsonl` | Record cần kiểm định bởi con người |
| `round1_stats.json` | Thống kê tổng hợp |
| `bad_records.jsonl` | Record không thể xử lý |
| `.checkpoint_llm.jsonl` | Checkpoint resume — lưu sau mỗi batch |

---

## Chạy tests

```bash
cd round-1-annotation
pytest tests/ -v
```

Tests kiểm tra logic routing trong `fusion_router.py` — không cần GPU.

---

## Ghi chú bàn giao

- Nhãn Round-1 là **silver labels** — tạm thời, chưa phải gold.
- Toàn bộ record `needs_human_review` chuyển sang Round-2 để kiểm định bởi con người.
- Record auto-accepted đã được lấy mẫu ngẫu nhiên (`random_audit_rate`) để QA.
- Không có fine-tuning, training hay sử dụng mô hình MMSD3.0.
