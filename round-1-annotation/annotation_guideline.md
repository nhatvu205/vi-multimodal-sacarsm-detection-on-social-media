# **Annotation Guideline — Sarcasm Detection trên MXH Tiếng Việt**

## **1\. Hệ thống nhãn tổng quan**

| Nhãn | Tên | Mô tả ngắn |
| :---: | ----- | ----- |
| **1** | Sarcastic | Tồn tại conflict rõ ràng giữa nghĩa bề mặt và ý thật |
| **0** | Non-sarcastic | Tất cả phương thức nhất quán, không có conflict ẩn |
| **\-1** | Invalid | Caption và ảnh không liên quan nhau / không đủ thông tin để gán nhãn |

## **2\. Nhãn INVALID (-1) — Định nghĩa và quy định**

### **2.1 Định nghĩa**

INVALID được gán khi **không thể xác định được sarcasm hay không** do dữ liệu đầu vào bị lỗi, thiếu, hoặc caption và ảnh hoàn toàn không liên quan nhau về mặt ngữ nghĩa.

### **2.2 Các trường hợp gán INVALID**

| Trường hợp | Mô tả | Ví dụ |
| ----- | ----- | ----- |
| **Caption–Image unrelated** | Caption và ảnh nói về hai chủ đề hoàn toàn khác nhau, không có điểm chung ngữ nghĩa nào.  | Ảnh phong cảnh núi \+ caption về công thức nấu ăn |
| **Caption rỗng / vô nghĩa** | Caption chỉ có ký tự đặc biệt, số ngẫu nhiên, hoặc không thành câu | "\#\#\#", "123abc???" |
| **Ảnh không đọc được** | Ảnh mờ, hỏng, hoặc không nhận diện được nội dung | Ảnh toàn màu đen / nhiễu |
| **Ngôn ngữ không xác định** | Caption bằng ngôn ngữ annotator không đọc được, không thể tra cứu | Caption bằng chữ tượng hình không rõ nguồn |
| **Nội dung bị kiểm duyệt** | Ảnh hoặc text bị che/xóa đến mức không còn đủ thông tin | Caption bị censor toàn bộ từ khóa chính |

### **2.3 Quy tắc bắt buộc khi gán INVALID**

**Dành cho annotator (người):**

* Phải ghi rõ lý do trong ô **Note** — không được để trống  
* Không được gán INVALID chỉ vì nội dung khó hiểu hoặc cần suy nghĩ lâu  
* Nếu có thể đoán được quan hệ giữa caption và ảnh dù mơ hồ → **không gán INVALID**, chuyển sang `Unclear`  
* Mỗi tuần tổng hợp tỉ lệ INVALID — nếu vượt 10% tổng batch cần review lại

**Dành cho LLM:**

* Phải trả về structured output có trường `reason` khi gán INVALID  
* Không được gán INVALID cho nội dung chính trị/nhạy cảm chỉ vì ngại xử lý  
* Ngưỡng tin cậy: nếu confidence \< 0.4 và không xác định được quan hệ caption–ảnh → gán INVALID, còn không → chọn nhãn gần nhất và ghi chú

## **3\. Taxonomy Sarcasm — Định nghĩa lại cho MXH Tiếng Việt**

### **Nhãn 1 — SARCASTIC**

| Loại | Tên | Định nghĩa | Đặc trưng trên MXH Việt | Dấu hiệu  nhận biết | Ví dụ Việt |
| :---: | ----- | ----- | ----- | ----- | ----- |
| **1.1** | Verbal | Text nói ngược hoàn toàn với ý thật, không cần ảnh, emoji để nhận ra conflict | Dùng từ khen trong ngữ cảnh chê: *"hay nhỉ", "tuyệt thật", "giỏi ghê"*; câu cảm thán ngược chiều trước tình huống tệ | Từ ngữ tích cực \+ tình huống tiêu cực rõ ràng (hoặc ngược lại); chỉ đọc text là đủ nhận ra, không cần ngữ cảnh ngoài | *"Hay thật, vừa nhận lương xong đã hết tiền rồi"* / *"Giỏi nhỉ, deadline 8h sáng mà 7h59 mới bắt đầu làm"* |
| **1.2** | Image–Text Conflict | Text và ảnh mâu thuẫn trực tiếp \- text khen nhưng ảnh xấu, hoặc text mô tả tích cực nhưng ảnh cho thấy thực tế ngược lại | Meme kết quả thực tế ↔ caption kỳ vọng; ảnh check-in ↔ caption sự thật phía sau; ảnh trước/sau ↔ caption phóng đại | Đọc text một mình: bình thường; xem ảnh một mình: bình thường; đọc cả hai: mâu thuẫn rõ ràng | Text: *"Nhà mình sau khi dọn dẹp"* ↔ Ảnh: phòng vẫn bừa bộn y chang |
| **1.3** | Emoji–Text Conflict | Emoji phủ nhận hoặc đảo ngược cảm xúc của text; bỏ emoji đi thì câu mất nghĩa mỉa mai | 🤡 sau lời khen \= tự nhạo/chỉ trích ngầm; 😭 sau tin vui \= phàn nàn; 🙂 đơn độc \= passive-aggressive; 💀 \= cường điệu hài hước | Emoji thuộc nhóm cảm xúc trái chiều với tone text; bỏ emoji đi câu trở nên trung tính hoặc tích cực | *"Hôm nay sếp khen mình làm việc tốt lắm 🤡"* / *"Tăng lương 2% sau 3 năm cống hiến 🥰"* |
| **1.4** | Contextual | Text bề ngoài bình thường, không có từ ngữ đảo nghĩa, nhưng người đọc nhận ra mỉa mai nhờ biết ngữ cảnh xã hội/văn hóa/sự kiện thực tế | Bình luận thời sự ngầm chỉ trích; dùng *"đúng rồi", "chắc chắn", "tất nhiên"* trước điều thực tế bác bỏ; meme về kẹt xe, giá nhà, lương thấp | Câu có vẻ trung lập/đồng ý nhưng ngữ cảnh thực tế bác bỏ hoàn toàn; cần biết "chuyện đang xảy ra" mới hiểu | *"Tất nhiên rồi, mua nhà Sài Gòn bằng lương công nhân viên chức dễ lắm mà"* |
| **1.5** | Self- deprecating | Người đăng tự chế giễu chính mình bằng cách gán danh hiệu/phẩm chất trái ngược hoàn toàn với hành vi thực tế | Tự phong danh hiệu hài hước: *"chuyên gia", "người trưởng thành", "người có kế hoạch"*; kể chuyện tự dìm theo cấu trúc kỳ vọng ↔ thực tế | Chủ thể là người đăng (ngôi thứ nhất); danh hiệu/lời tự khen ↔ hành vi thực tế trái ngược; không nhằm chỉ trích người khác | *"Mình là chuyên gia tài chính — chuyên tiêu hết tiền trước ngày 10"* / *"Tôi là người ngủ sớm. Định nghĩa: trước 2 giờ sáng"* |
| **1.6** | Hyperbolic | Dùng cường điệu quá mức theo hướng tích cực hoặc tiêu cực để truyền tải ý phê phán hoặc chế giễu về tình huống thực tế | *"chỉ có", "chỉ mới", "nhẹ thôi"* trước con số/tình huống thực ra rất tệ; *"tuyệt vời", "hoàn hảo"* cho điều rõ ràng là bình thường hoặc tệ; hay gặp trong bình luận thời tiết, giá cả, giao thông | Từ phóng đại không khớp với mức độ thực tế; nếu hiểu theo nghĩa literal thì câu vô lý hoặc buồn cười | *"Thời tiết Sài Gòn dễ chịu quá, chỉ có 40 độ thôi mà"* / *"Giá xăng tăng nhẹ thôi, chỉ thêm 3k/lít"* |
| **1.7** | Multimodal | Mỉa mai chỉ xuất hiện khi kết hợp đồng thời ≥2 phương thức; đọc riêng lẻ từng phương thức không đủ để nhận ra conflict | Meme ghép ảnh \+ caption cần đọc cả hai mới "vỡ ra"; facial expression trái ngược caption; caption trung tính \+ ảnh meme đã mang sẵn nghĩa mỉa mai trong văn hóa Việt | Test: che text → không thấy mỉa mai; che ảnh → không thấy mỉa mai; xem cả hai → mỉa mai rõ | Text: *"Cảm giác khi nói 'để em tham khảo đã ạ'"* \+ ảnh mặt nạ mèo che khuôn mặt ranh mãnh |
| **0** | Non- sarcastic | Text, ảnh, emoji và ngữ cảnh hoàn toàn nhất quán — nghĩa literal đúng với thực tế, không có conflict ẩn | Chia sẻ thật lòng, mô tả trực tiếp sự việc, cảm xúc thật; emoji khớp tone text; không có ẩn ý ngầm | Tất cả phương thức cùng chiều; không cần "đọc giữa dòng" | Ảnh cơm ngon \+ *"Hôm nay ăn cơm ngon quá"* \+ 😋 / *"Mình vừa được nhận vào công ty mơ ước 🥹"* \+ ảnh offer letter thật |
| **\-1** | Invalid | Caption và ảnh không liên quan nhau, hoặc không đủ thông tin để xác định bất kỳ nhãn nào | Caption rỗng/vô nghĩa; ảnh hỏng/không đọc được; caption và ảnh về hai chủ đề hoàn toàn khác nhau; ngôn ngữ không xác định | Không tìm được bất kỳ quan hệ ngữ nghĩa nào giữa caption và ảnh dù cố gắng suy luận | Ảnh phong cảnh núi \+ caption công thức nấu ăn không liên quan / Caption toàn ký tự đặc biệt "\#\#\#???" |

### **Nhãn 0 — NON-SARCASTIC**

**Định nghĩa:** Text, ảnh, emoji và ngữ cảnh **hoàn toàn nhất quán** với nhau. Nghĩa literal của câu đúng với thực tế được thể hiện — không có conflict ẩn nào.

| Dấu hiệu | Ví dụ |
| ----- | ----- |
| Text mô tả đúng ảnh | Ảnh cơm ngon \+ *"Hôm nay ăn cơm ngon quá"* |
| Emoji tương ứng cảm xúc | *"Buồn quá 😢"* — emoji khớp nội dung |
| Phát biểu trung lập / thông tin | *"Hôm nay trời mưa ở Hà Nội"* |
| Chia sẻ thật lòng, không có ẩn ý | *"Mình vừa được nhận vào công ty mơ ước 🥹"* \+ ảnh offer letter thật |

## 

## **4\. Cây quyết định gán nhãn**

**Bước 1: Caption và ảnh có liên quan nhau không?**  
    → Không / Không xác định được  →  INVALID (-1) \+ ghi note lý do  
    → Có  →  Bước 2

**Bước 2: Có conflict giữa các phương thức không?**  
    → Không  →  Non-sarcastic (0)  
    → Có  →  Bước 3

**Bước 3: Conflict đến từ đâu?**  
    ├── Chỉ từ ngôn ngữ (text tự mâu thuẫn)        →  1.1 Verbal  
    ├── Text ↔ Ảnh mâu thuẫn                        →  1.2 Image–Text  
    ├── Emoji đảo ngược cảm xúc text                →  1.3 Emoji–Text  
    ├── Cần ngữ cảnh xã hội/thực tế bên ngoài       →  1.4 Contextual  
    ├── Chủ thể tự chế giễu bản thân                →  1.5 Self-deprecating  
    ├── Phóng đại quá mức để chỉ trích              →  1.6 Hyperbolic  
    └── Phải kết hợp ≥2 phương thức mới hiểu được  →  1.7 Multimodal

Lưu ý: Một sample có thể gán NHIỀU loại con (1.x) cùng lúc.

## **5\. Các trường hợp dễ nhầm**

| Tình huống | Không phải INVALID | Lý do |
| ----- | ----- | ----- |
| Caption chính trị gay gắt | Non-sarcastic (0) | Phát biểu thẳng, không có conflict ẩn |
| Caption khó hiểu nhưng liên quan ảnh | Unclear → chọn nhãn gần nhất | Vẫn có quan hệ ngữ nghĩa |
| Caption tiếng Anh/tiếng lóng khó dịch | Cố gắng dịch \+ ghi note | Không gán INVALID chỉ vì khó |
| Ảnh meme nổi tiếng không có caption | Xét theo nghĩa meme gốc | Ngữ cảnh meme đã là caption ngầm |

