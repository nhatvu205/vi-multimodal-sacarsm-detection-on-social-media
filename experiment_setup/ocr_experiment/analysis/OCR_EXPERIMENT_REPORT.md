# Báo cáo camera-ready — phần Nhật Vũ (T05–T08)

Ngày phân tích: 08/09/2026. Nguồn yêu cầu: `ViMMSarc_Camera_Ready_Revision_Plan.xlsx`, sheet **Vi-checklist**, các dòng T05–T08 và sheet **Raw feedback**. Nguồn kết quả: `experiment_setup/camera_ready/outputs/`; dữ liệu đối chiếu: `data/final-data/`. Các bảng trong báo cáo dùng **F1-macro (%)**, SD và CI cũng theo thang phần trăm; chênh lệch là **điểm phần trăm (pp)**.

## 1. Kết luận có thể dùng cho bản camera-ready

PhoBERT caption+OCR đạt **70,05 ± 1,25**, cao hơn caption-only **68,19 ± 0,77**. Tuy nhiên, chênh lệch **+1,86 pp**, bootstrap CI 95% **[−0,64; +4,33]**, chưa đủ để khẳng định cải thiện có ý nghĩa thống kê. OCR-only đạt **56,55 ± 0,49**: OCR có tín hiệu dự đoán, nhưng không thay thế được caption trong thiết lập này.

Với DT4MID, ảnh+caption đạt **67,24 ± 1,39**, ảnh+caption+OCR đạt **67,17 ± 0,83**. Chênh lệch **−0,07 pp**, CI **[−2,28; +2,05]**: không thấy lợi ích OCR nhất quán. Các kết quả mới cũng không hỗ trợ kết luận mô hình multimodal luôn tốt hơn mô hình văn bản.

Cả hai mô hình giảm F1 khi chuyển nền tảng. DT4MID Facebook→Threads chỉ đạt **31,59 ± 8,62**, với độ biến động seed cao. Đây là bằng chứng hạn chế khả năng chuyển miền trong thiết lập đã chạy, không phải bằng chứng rằng riêng một đặc điểm nền tảng gây ra thất bại.

## 2. Kiểm tra artifact và phạm vi bằng chứng


Có metric cho **27/27** lần huấn luyện, **39/39** cặp study/seed/eval-split. Đọc được **39/39** prediction JSONL. Kiểm tra nội dung trả về **0 lỗi**.

Prediction source-test DT4MID Threads seed 123 đã được bổ sung và đối chiếu thành công với 445 ID Threads trong test gốc. Tất cả CI, error slices và ví dụ trong báo cáo hiện dùng đầy đủ ba seed khi có liên quan.

Các kiểm tra đã thực hiện:

- Train/dev/test có 5.884/735/736 ID, không trùng trong tập và không giao nhau giữa các tập.
- Mỗi prediction test đầy đủ chính xác tập ID tương ứng: 736 toàn bộ test, 291 Facebook hoặc 445 Threads. Không có ID dư/thiếu/trùng trong 39 file.
- Gold MM, nhãn (T,I,MM), source, has_ocr đối chiếu với JSON gốc; seed, input mode, scenario và split khớp cấu hình mong đợi.
- Confusion matrix tính lại từ prediction khớp metric. F1/accuracy tính từ confusion matrix khớp số lưu (sai số làm tròn 4 chữ số); summary chứa metric tương ứng.
- Lưu SHA-256 của từng prediction và ba JSON dữ liệu ở `artifact_inventory.csv` và `artifact_audit.json`.

Không có training manifest/resolved config/checkpoint trong bản output gọn. Vì vậy có thể xác minh kết quả và metadata đánh giá; không thể khôi phục độc lập commit huấn luyện, phiên bản dependency, epoch tốt nhất hay chứng minh lịch sử chọn checkpoint chỉ từ các file này. Protocol dưới đây lấy từ code/config hiện có và runbook, không được coi là bản log lịch sử từng notebook.

## 3. Protocol và số mẫu

Theo code/config camera-ready: target chung `mm_label`, scenario s1, giữ emoji, chuẩn hóa khoảng trắng, tối đa 256 token. PhoBERT dùng `vinai/phobert-base`; DT4MID kết hợp backbone này với `google/vit-base-patch16-224-in21k`. Seed 42/123/2026; tối đa 10 epoch, early stopping patience 2 theo dev F1-macro; AdamW, learning rate 2e-5, weight decay 0,01. Batch train/eval trong config hiện có: PhoBERT 16/32, DT4MID 8/16. Không mô tả đây là batch 4/8 của paper gốc.

OCR lấy trực tiếp từ `ocr_text`, không chạy OCR engine mới. OCR-only nhận chuỗi rỗng khi không có OCR; caption+OCR nối bằng marker [OCR], và chỉ giữ caption nếu OCR rỗng. OCR-present là chuỗi OCR không rỗng sau strip, **không đồng nghĩa OCR đúng hoặc hữu ích**. Độ dài 256 token có thể cắt nội dung, nhưng chưa có thống kê truncation để quy nguyên nhân.

Cross-platform chỉ train/dev trên nguồn, đánh giá source_test và target_test từ test gốc. Hai test có ID khác nhau, do đó không dùng paired bootstrap theo ID giữa source_test và target_test. Số mẫu train nguồn khác nhau, tỷ lệ lớp cũng khác; cần nêu cả hai khi diễn giải domain gap.


| Split | Nguồn | N | MM=0 | MM=1 |
| --- | --- | --- | --- | --- |
| train | all | 5884 | 3305 | 2579 |
| train | facebook | 2322 | 763 | 1559 |
| train | threads | 3562 | 2542 | 1020 |
| dev | all | 735 | 413 | 322 |
| dev | facebook | 286 | 94 | 192 |
| dev | threads | 449 | 319 | 130 |
| test | all | 736 | 414 | 322 |
| test | facebook | 291 | 98 | 193 |
| test | threads | 445 | 316 | 129 |

| Test subset | N | MM=0 | MM=1 |
| --- | --- | --- | --- |
| all | 736 | 414 | 322 |
| present | 486 | 271 | 215 |
| absent | 250 | 143 | 107 |

Tỷ lệ MM=1 của test Facebook là 193/291 = **66,32%**, Threads là 129/445 = **28,99%**. OCR hiện diện ở 486/736 = **66,03%** test; toàn bộ dataset có 3.969+486+486 = 4.941/7.355 = **67,18%**.

## 4. Cách tính độ bất định

Mean và sample SD (ddof=1) tính từ F1 của ba seed, không gộp 3×736 dòng như các mẫu độc lập. F1 dùng cố định hai lớp 0/1; metric gốc được tính lại từ confusion matrix để tránh sai số làm tròn.

CI chính trong `mean_f1_bootstrap.csv`: **10.000 bootstrap resample ID có hoàn lại**, lấy chung cùng ID resample cho mọi seed, tính F1 từng seed rồi trung bình. Percentile 2,5/97,5 tạo CI 95%; RNG seed 2026. CI này đo độ bất định do test sampling, **có điều kiện trên các model/seed đã train**, không bao gồm toàn bộ bất định của training population hoặc lỗi nhãn.

OCR delta trong `paired_ocr.csv` dùng chung seed và cùng ID cho caption/caption+OCR (tương tự DT4MID), lấy F1(OCR-enhanced) − F1(control) ở mỗi resample. Đây là CI theo cặp, không phải so sánh hai CI riêng lẻ. Không công bố p-value; phân tích subset là khám phá, không có hiệu chỉnh đa so sánh.

Runbook cũ xuất `bootstrap_ci_lower_mean/upper_mean` trong `seed_summary.csv`: đó chỉ là trung bình endpoints per-seed, **không phải CI hợp lệ của mean qua seed**. Không dùng hai cột này trong paper. Báo cáo dùng bảng CI mới nêu trên.

## 5. T05 — OCR ablation


| Đầu vào | F1 full mean±SD | 95% CI full | F1 OCR-present | F1 OCR-absent |
| --- | --- | --- | --- | --- |
| PhoBERT · caption | 68.19 ± 0.77 | [65.17, 71.10] | 64.75 ± 1.25 | 74.88 ± 0.76 |
| PhoBERT · OCR | 56.55 ± 0.49 | [53.18, 59.87] | 59.62 ± 0.76 | 36.39 ± 0.00 |
| PhoBERT · caption+OCR | 70.05 ± 1.25 | [67.06, 72.94] | 66.61 ± 1.63 | 76.31 ± 1.37 |
| DT4MID · ảnh+caption | 67.24 ± 1.39 | [64.20, 70.16] | 64.75 ± 1.72 | 71.84 ± 1.25 |
| DT4MID · ảnh+caption+OCR | 67.17 ± 0.83 | [64.35, 69.88] | 63.73 ± 2.09 | 73.45 ± 1.33 |

| So sánh thêm OCR | Subset | Δ F1 (pp) | Paired 95% CI | Sửa đúng / làm sai (sample×seed) |
| --- | --- | --- | --- | --- |
| phobert | full | 1.86 | [-0.64, 4.33] | 250 / 207 |
| phobert | ocr_present | 1.86 | [-1.42, 5.17] | 192 / 165 |
| phobert | ocr_absent | 1.43 | [-2.08, 4.83] | 58 / 42 |
| dt4mid | full | -0.07 | [-2.28, 2.05] | 210 / 224 |
| dt4mid | ocr_present | -1.01 | [-4.02, 1.98] | 161 / 184 |
| dt4mid | ocr_absent | 1.61 | [-1.11, 4.34] | 49 / 40 |

PhoBERT cải thiện trung bình tương tự trên full và OCR-present (+1,86 pp), nhưng CI đều chứa 0. DT4MID giảm 1,01 pp trên OCR-present, CI cũng chứa 0. Không nên viết “OCR materially improves all models”.

Trong nhóm OCR-absent, OCR-only dự đoán toàn lớp 0 ở cả ba seed: F1-macro **36,39 ± 0,00**. Điều này phù hợp đầu vào rỗng chung cho cả nhóm, không phải bằng chứng pipeline sao chép prediction. F1 cùng giá trị qua seed ở riêng nhóm này phải được giải thích khi trả lời reviewer về repeated scores.

Caption+OCR có thể đổi kết quả ngay trên OCR-absent mặc dù đầu vào tại inference giống caption: hai model được huấn luyện trên đầu vào khác nhau của toàn bộ train set. Không quy thay đổi ở nhóm này cho OCR xuất hiện ở chính mẫu test.

Các số helps/harms là tổng số **cặp mẫu–seed**, có thể đếm một ID tối đa ba lần; không gọi là số bài đăng khác nhau.

## 6. T06 — Đánh giá chéo nền tảng


| Mô hình | Train→Test | N test | F1 mean±SD (3 seed) | 95% CI | Prediction seeds |
| --- | --- | --- | --- | --- | --- |
| PhoBERT | Facebook→Facebook | 291 | 60.96 ± 0.36 | [55.62, 65.91] | 3/3 |
| PhoBERT | Facebook→Threads | 445 | 47.77 ± 4.39 | [43.65, 51.79] | 3/3 |
| PhoBERT | Threads→Threads | 445 | 60.71 ± 2.10 | [56.56, 64.77] | 3/3 |
| PhoBERT | Threads→Facebook | 291 | 52.45 ± 4.75 | [47.23, 57.25] | 3/3 |
| DT4MID | Facebook→Facebook | 291 | 53.09 ± 8.63 | [48.61, 57.53] | 3/3 |
| DT4MID | Facebook→Threads | 445 | 31.59 ± 8.62 | [28.48, 34.63] | 3/3 |
| DT4MID | Threads→Threads | 445 | 59.75 ± 4.77 | [55.91, 63.44] | 3/3 |
| DT4MID | Threads→Facebook | 291 | 47.73 ± 3.70 | [43.58, 51.80] | 3/3 |

Theo cùng model train nguồn, F1 source→target giảm:

- PhoBERT train Facebook: **13,19 pp**; train Threads: **8,26 pp**.
- DT4MID train Facebook: **21,50 pp**; train Threads: **12,02 pp**.

Các chênh lệch này là mô tả, vì hai tập test có ID và phân bố lớp khác nhau. Đối chứng cùng test đích cũng cho kết quả thấp hơn khi train khác miền: PhoBERT trên Threads giảm 60,71→47,77 (12,94 pp), trên Facebook giảm 60,96→52,45 (8,51 pp); DT4MID tương ứng giảm 59,75→31,59 (28,16 pp) và 53,09→47,73 (5,36 pp). Đây vẫn là thay đổi đồng thời dữ liệu/size/prevalence của tập train, không phải phép tách riêng hiệu ứng ngôn ngữ nền tảng.

DT4MID Facebook-source có SD lớn cả trong miền (8,63 pp) và khác miền (8,62 pp). Trong ba seed, F1 Facebook→Threads là 41,54 / 26,40 / 26,83. Không bỏ các seed điểm thấp hoặc chọn seed tốt nhất để báo cáo. Cần xem training log nếu muốn giải thích instability; output gọn không chứa đủ log để xác định nguyên nhân.

## 7. T07 — Phân tích lỗi

Định nghĩa: FP là MM=0 nhưng dự đoán 1; FN là MM=1 nhưng dự đoán 0; FPR=FP/N(MM=0), FNR=FN/N(MM=1). Nhóm (0,0,1) gồm **160 mẫu** (49,69% của 322 positive test); chỉ có nhãn MM=1, nên FPR không xác định, không ghi 0 để tạo cảm giác model không có lỗi.


| Đầu vào | FP mean | FN mean | FPR % | FNR % | FN mean (001) | FNR (001) % |
| --- | --- | --- | --- | --- | --- | --- |
| PhoBERT · caption | 134.00 | 98.67 | 32.37 | 30.64 | 46.00 / 160 | 28.75 |
| PhoBERT · OCR | 111.67 | 190.67 | 26.97 | 59.21 | 115.00 / 160 | 71.88 |
| PhoBERT · caption+OCR | 124.00 | 94.33 | 29.95 | 29.30 | 50.67 / 160 | 31.67 |
| DT4MID · ảnh+caption | 112.33 | 123.00 | 27.13 | 38.20 | 62.67 / 160 | 39.17 |
| DT4MID · ảnh+caption+OCR | 139.67 | 100.33 | 33.74 | 31.16 | 54.00 / 160 | 33.75 |

DT4MID thêm OCR giảm FN trên (001) từ trung bình 62,67 xuống 54/160, nhưng tăng FP toàn test từ 112,33 lên 139,67/414. Đánh đổi này giải thích vì sao recall của nhóm khó có thể tốt hơn trong khi F1 tổng thể không tăng.

PhoBERT thêm OCR cải thiện F1 full, nhưng FNR (001) tăng từ **28,75%** lên **31,67%**. Vì vậy cải thiện overall không đồng nghĩa giải quyết được nhóm cross-modal theo chú thích. Caption-only vẫn dự đoán đúng phần lớn (001), nên không dùng tổ hợp nhãn để “chứng minh” unimodal không thể dự đoán nhãn MM.

Chi tiết từng seed, từng tổ hợp nhãn, OCR và nền tảng nằm trong `diagnostic_rates.csv`; `error_slices.csv` cung cấp lát cắt giao nhau (T,I,MM)×OCR×source. Không suy ra bất đồng giữa annotator từ FP/FN: đó là lỗi model so với gold hiện có.

### Năm ví dụ thật cho T08 / Figure 4

Đã đọc caption/OCR và xem trực tiếp ảnh gốc của cả năm ID dưới đây. Phần mô tả được diễn giải lại, bỏ tên/handle/avatar; ID chỉ để nhóm truy xuất nội bộ. Đây là các trường hợp minh họa được chọn có chủ đích, không phải mẫu ngẫu nhiên hoặc chứng cứ nhân quả về cơ chế model. Nhãn giữ nguyên theo dataset, chưa có vòng adjudication mới.

Ký hiệu P=PhoBERT caption, P+O=PhoBERT caption+OCR, D=DT4MID ảnh+caption, D+O=DT4MID ảnh+caption+OCR. Mỗi vector theo thứ tự seed **42,123,2026**.


**ID 6564 — Lỗi (001), không OCR**, nguồn facebook, nhãn (0,0,1). Caption diễn đạt vừa mua được điện thoại; ảnh đặt chiếc điện thoại nổi bật trong một căn phòng xuống cấp. Sự tương phản giữa vật mua và bối cảnh sống là một cách đọc mỉa mai khả dĩ. D+O bỏ sót cả ba seed; ảnh không cung cấp OCR để hỗ trợ.

| P | P+O | D | D+O |
| --- | --- | --- | --- |
| 42:0,123:1,2026:1 | 42:0,123:0,2026:0 | 42:1,123:0,2026:0 | 42:0,123:0,2026:0 |

**ID 308 — OCR giúp**, nguồn threads, nhãn (0,0,1). Caption so sánh hai cách nhận xét chuyện ăn nhiều; screenshot chứa lối nói phóng đại việc ăn như một vụ sạt lở và lời đáp bào chữa. P sai cả ba seed, P+O đúng cả ba seed. OCR cung cấp phần hội thoại bị thiếu trong caption.

| P | P+O | D | D+O |
| --- | --- | --- | --- |
| 42:0,123:0,2026:0 | 42:1,123:1,2026:1 | 42:1,123:0,2026:0 | 42:0,123:1,2026:1 |

**ID 5835 — OCR hại / tín hiệu phụ**, nguồn facebook, nhãn (0,0,1). Caption nói về bản thân vài giờ sau; ảnh là bộ xương mặc đồ tiên đang nhảy. OCR chỉ đọc được chữ ký họa sĩ. P đúng cả ba seed, P+O sai cả ba seed; phù hợp giả thuyết văn bản phụ có thể gây nhiễu, nhưng không chứng minh chữ ký là nguyên nhân.

| P | P+O | D | D+O |
| --- | --- | --- | --- |
| 42:1,123:1,2026:1 | 42:0,123:0,2026:0 | 42:1,123:0,2026:1 | 42:1,123:1,2026:0 |

**ID 1964 — False positive**, nguồn facebook, nhãn (0,0,0). Caption thể hiện đồng tình với một lời khuyên nghỉ ngơi đơn giản trên nền cỏ. Gold (000), P+O và D+O đều dự đoán 1 cả ba seed. Đây là ví dụ đọc quá mức hàm ý mỉa mai; OCR bị lỗi ký tự dù ảnh chứa câu ngắn rõ nghĩa.

| P | P+O | D | D+O |
| --- | --- | --- | --- |
| 42:1,123:1,2026:0 | 42:1,123:1,2026:1 | 42:1,123:0,2026:0 | 42:1,123:1,2026:1 |

**ID 6000 — Thành công (001)**, nguồn facebook, nhãn (0,0,1). Caption mô phỏng lời mẹ khen con; screenshot cho thấy con chủ động đề nghị giáo viên thu thêm học phí vì mẹ có tiền. Lời khen có thể được đọc ngược trong ngữ cảnh này. D và D+O đúng cả ba seed; P cũng đúng nên ví dụ không chứng minh ảnh là điều kiện cần cho model.

| P | P+O | D | D+O |
| --- | --- | --- | --- |
| 42:1,123:1,2026:1 | 42:1,123:1,2026:1 | 42:1,123:1,2026:1 | 42:1,123:1,2026:1 |

Các mô tả trên dùng được làm bản thảo định tính ẩn danh. Khi đưa ảnh gốc vào Figure 4, cần tạo bản che avatar/handle và thông tin nhận dạng (đặc biệt ID 308), đồng thời duyệt theo chính sách phát hành của nhóm. Chưa tạo hoặc tuyên bố ảnh gốc đã được ẩn danh. Không gán bảy taxonomy types khi không có nhãn taxonomy đã xác minh.

## 8. Đối chiếu nhiệm vụ và reviewer


| Task | Reviewer IDs | Bằng chứng / trạng thái bàn giao |
| --- | --- | --- |
| T05 | R2-04; R3-05; R4-02; R5-03 | Đủ 5 input × 3 seed; full/OCR-present/absent, phân bố lớp, paired OCR CI. Hoàn tất phân tích OCR. |
| T06 | R4-01; R4-03; R5-04 | Đủ metric, prediction, CI và counts cho 4 hướng × 2 model × 3 seed. Hoàn tất phân tích cross-platform. |
| T07 | R2-04; R5-02; R5-06 | Xuất FP/FN, support, rates, ID và prediction thật; chưa thay thế phân tích bất đồng annotation của T09 hay benchmark khác do T02 phụ trách. |
| T08 | R2-04; R3-05; R4-02; R4-03; R5-02; R5-03; R5-04; R5-06 | Có báo cáo, 5 ví dụ đã diễn giải ẩn danh và draft tiếng Anh bên dưới; ảnh Figure 4 cần ẩn danh/duyệt và nội dung cần tích hợp vào main.tex. |
| Bàn giao T03/T04 | R2-05; R3-04; R5-01 | 10.000 resample, 3 seed, sample SD, paired CI và checksum cho các run thuộc Nhật Vũ; không đại diện toàn bộ baseline s1–s4. |

R2-04 còn phần taxonomy do T10 phụ trách; R5-02 còn annotation disagreement do T09 phụ trách; R4-01/R4-03 còn reliability/claims ngoài thí nghiệm. Không đánh dấu toàn bộ các feedback đa phần này “đã xử lý” chỉ bằng báo cáo kết quả.

## 9. Bản thảo tiếng Anh để tích hợp paper / phản hồi reviewer

### OCR subsection

We evaluate caption-only, OCR-only, caption+OCR, image+caption, and image+caption+OCR inputs using the same binary MM target and fixed data splits. We reuse the supplied OCR text; missing OCR is represented by an empty string for OCR-only inputs, while caption+OCR falls back to the caption. Results use seeds 42, 123, and 2026 in the s1 setting with a maximum text length of 256 tokens. The test set contains 736 posts, including 486 with OCR and 250 without OCR. We report the mean and sample standard deviation across seeds. Confidence intervals use 10,000 test-ID bootstrap resamples shared across seeds; OCR comparisons additionally pair the same test IDs and seeds across input settings.

PhoBERT improves from 68.19 ± 0.77 to 70.05 ± 1.25 macro-F1 with caption+OCR. The paired improvement is 1.86 percentage points (95% CI: −0.64 to 4.33), indicating a positive average trend but inconclusive evidence of a consistent gain. OCR alone achieves 56.55 ± 0.49. For DT4MID, adding OCR changes macro-F1 from 67.24 ± 1.39 to 67.17 ± 0.83, with a paired difference of −0.07 points (95% CI: −2.28 to 2.05). Thus, the usefulness of OCR depends on the evaluated model and input configuration, and these experiments do not establish a general multimodal advantage.

### Cross-platform subsection

We train on each platform separately and select checkpoints using only that platform's development subset under the camera-ready protocol. Source and target tests are platform-specific subsets of the fixed test set. Facebook contains 291 test posts (193 MM-positive), whereas Threads contains 445 (129 MM-positive). PhoBERT trained on Facebook obtains 60.96 ± 0.36 macro-F1 on Facebook and 47.77 ± 4.39 on Threads; training on Threads yields 60.71 ± 2.10 on Threads and 52.45 ± 4.75 on Facebook. DT4MID obtains 53.09 ± 8.63 and 31.59 ± 8.62 when trained on Facebook, and 59.75 ± 4.77 and 47.73 ± 3.70 when trained on Threads, respectively. These results reveal limited cross-platform transfer and seed sensitivity, particularly for DT4MID trained on Facebook. Differences in training-set size, class prevalence, and test composition prevent attributing the entire gap to a single platform characteristic.

### Error analysis subsection

The test set contains 160 posts with the annotated (T,I,MM) combination (0,0,1). With OCR, DT4MID's mean false-negative rate on this subset decreases from 39.17% to 33.75%, while its full-test false-positive rate increases from 27.13% to 33.74%. PhoBERT's overall improvement with OCR does not extend uniformly to this subset: its false-negative rate rises from 28.75% to 31.67%. These findings motivate reporting both subset recall and false-positive behavior rather than interpreting overall macro-F1 as evidence that cross-modal cases have been resolved. Qualitative cases include contextual incongruity missed despite visual input, informative screenshot text, incidental OCR associated with changed predictions, false-positive readings of neutral posts, and a correctly classified contextual example. Such cases illustrate observed behavior rather than establish causal explanations of model decisions.

### Các câu cần sửa khi tích hợp

- IV.A hiện nói target thay đổi theo family, weighted F1 dev, single run và batch 4/8. Phần camera-ready của Nhật Vũ phải ghi MM chung, macro-F1 dev, ba seed và config thực tế; không suy rộng thay đổi này sang mọi baseline cũ chưa audit.
- IV.E/VI/abstract không giữ kết luận “multimodal architectures outperform unimodal approaches” như một kết luận tổng quát. Kết quả cao nhất trong 5 setting mới là PhoBERT caption+OCR.
- Thay “label distribution proves unimodal inadequacy” bằng mô tả phụ thuộc ngữ cảnh theo annotation protocol.
- Không viết “statistically significant OCR gain”: paired CI chứa 0.
- Không diễn giải classifier errors là LLM–human hoặc human–human annotation disagreement.
- T03/T04 cần thống nhất bảng kết quả mới với T02; T14 tích hợp toàn văn. Báo cáo này không tự thay các bảng benchmark cũ trong main.tex.

## 10. Artifact và tái chạy

Chạy từ repository root:

```bash
python3 -m experiment_setup.camera_ready.reviewer_analysis   --runs_root experiment_setup/camera_ready/outputs   --data_root data/final-data   --output_dir experiment_setup/camera_ready/analysis   --bootstrap_iterations 10000
```

| File | Mục đích |
| --- | --- |
| artifact_audit.json / artifact_inventory.csv | Tính đầy đủ, lỗi kiểm tra, ID support, SHA-256 |
| dataset_counts.csv | N và phân bố MM theo split/source/OCR |
| all_run_summary.csv | Mean±SD từ toàn bộ metric của 27 run |
| mean_f1_bootstrap.csv | Mean±SD và CI của đầy đủ ba seed cho từng setting |
| paired_ocr.csv | Chênh lệch OCR theo cặp, CI và helps/harms |
| seed_metrics.csv | Metric và CI theo từng seed/subset |
| seed_summary.csv | Output tương thích runbook; không dùng mean endpoints làm CI |
| diagnostic_rates.csv / error_slices.csv | FP/FN theo nhóm riêng và nhóm giao nhau |
| candidate_examples.csv / example_predictions.csv | ID lỗi và prediction để chọn case study |
| validation.json | Kiểm tra cơ bản của analyzer; dùng artifact_audit.json cho kiểm kê đầy đủ |

Tất cả số trong báo cáo lấy từ artifact nêu trên. Khi bổ sung file, bảng CSV có thể tái tạo bằng một lệnh; cần cập nhật phần prose/đếm file trong báo cáo tương ứng trước khi chốt bản nộp.

## 11. Next steps để hoàn thành feedback reviewer

Phần thí nghiệm của Nhật Vũ đã có đủ artifact để hoàn tất T05--T07. Việc còn lại gần nhất là T08: đưa ba đoạn tiếng Anh ở mục 9 vào `report/main.tex`, thêm một bảng OCR năm setting, một bảng cross-platform bốn hướng, và chọn 4--6 ví dụ từ mục 7. Chỉ dùng ảnh sau khi che avatar, handle, tên người dùng và nội dung định danh; caption trong paper nên là bản diễn giải ẩn danh. Sau khi tích hợp, biên dịch PDF và đối chiếu từng số với CSV trong thư mục `analysis/`.

Để trả lời đầy đủ reviewer, nhóm cần hoàn tất theo thứ tự sau:

1. T01--T04: xác minh lại target MM chung, split, checkpoint selection và seed cho toàn bộ baseline cũ; chạy/tổng hợp các baseline cần giữ với cùng protocol; thay các mô tả single-run, weighted-F1 và target khác nhau trong Mục IV nếu chúng không còn đúng. Đây là phần cần thiết để phản hồi R2-05, R3-02, R3-04 và R5-01 trên phạm vi toàn paper.
2. T09--T11: bổ sung định nghĩa T/I/MM, thông tin annotator, adjudication, tỷ lệ sửa nhãn, provenance test và limitations. Gọi các kappa hiện có là LLM--human agreement; không suy ra human--human IAA nếu không có nhãn độc lập. Hoàn tất taxonomy/diễn giải `(0,0,1)` thận trọng cho R2-03, R3-01, R3-03, R4-01 và R5-02.
3. T12--T13: hoàn chỉnh related-work comparison có nguồn kiểm chứng, ethics/data-release, giấy phép, quy trình ẩn danh và removal contact. Không đưa claim về ToS, copyright hoặc quyền phát hành khi chưa có chứng cứ.
4. T14: gộp các bảng và prose đã duyệt, bỏ claim "multimodal superiority" và "validated high reliability" quá mạnh, kiểm tra reference/figure/PDF, rồi dùng sheet `Vi-checklist` để đánh dấu mỗi raw feedback đã có bằng chứng hoặc limitation rõ ràng.

Trước khi nộp, chạy lại `reviewer_analysis` một lần cuối sau mọi thay đổi artifact, lưu `artifact_audit.json`, `analysis_manifest.json` và commit SHA vào supplement/repository release. Điều này không thay cho việc lưu manifest ở lần huấn luyện; compact output hiện tại không chứa commit, dependency và checkpoint của ba notebook gốc.
