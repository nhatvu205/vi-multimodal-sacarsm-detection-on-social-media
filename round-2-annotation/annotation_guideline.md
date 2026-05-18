# **HƯỚNG DẪN CHI TIẾT CÁC BƯỚC GÁN NHÃN FINE-GRAINED**

## 

## **BƯỚC 1: GÁN NHÃN TEXT-ONLY (T)**

**Mục tiêu:** Xác định xem nội dung chữ có chứa yếu tố mỉa mai khi đứng độc lập hay không.

* **Cách làm:** Che hoàn toàn ảnh. Đọc Text 2 lần:  
  1. *Lần 1:* Đọc theo nghĩa đen (Literal).  
  2. *Lần 2:* Tìm kiếm các "tín hiệu mỉa mai" (Sarcastic Cues).  
* **Các kịch bản gán T=1:**  
  1. **Mâu thuẫn nội hàm:** Trong cùng một câu có các từ đối lập (Ví dụ: "*Sáng ra đã được sếp tặng cho 'cơn mưa' lời khen về việc đi muộn, vui quá là vui.*").  
  2. **Cường điệu hóa (Hyperbole):** Dùng các từ cực đoan để nói về việc bình thường (Ví dụ: "*Đỉnh nóc kịch trần*", "*Hết nước chấm*", "*Tuyệt tác nhân loại*").  
  3. **Sử dụng Emoji đảo ngược:** Câu khẳng định tích cực đi kèm emoji tiêu cực hoặc mỉa mai (Ví dụ: "*Bài thi được 2 điểm, mình giỏi thật sự 🤡*").  
  4. **Khen ngợi giả tạo:** Dùng các tính từ cực tốt cho các tình huống hiển nhiên là tệ (Ví dụ: "*Trời mưa ngập đường, đi lại thật là thuận tiện và sạch sẽ*").  
* **Lưu ý:** Nếu văn bản là một câu hỏi thông thường, một lời chào, hoặc một câu cảm thán không rõ sắc thái (Ví dụ: "*Alo*", "*Hay quá*", "*Cạn lời*") \-\> **Gán T=0**.

## **BƯỚC 2: GÁN NHÃN IMAGE-ONLY (I)**

**Mục tiêu:** Xác định bức ảnh có tự thân chứa đựng sự mỉa mai/châm biếm mà không cần chữ giải thích hay không.

* **Cách làm:** Che phần văn bản và emoji. Chỉ nhìn vào khung hình.  
* **Các kịch bản gán I=1:**  
  1. **Ảnh Meme:** Các hình ảnh nhân vật nổi tiếng với biểu cảm đặc trưng (ếch Pepe, gấu lầy, các template meme thịnh hành) vốn dĩ đã mang tính châm biếm.  
  2. **Sự đối lập trực quan (Visual Juxtaposition):** Ảnh chụp một biển báo "Cấm đổ rác" đặt trên một đống rác lớn; ảnh một chiếc xe cứu hỏa đang bốc cháy.  
  3. **Chữ trong ảnh (Text-in-image):** Nếu trong ảnh có bảng hiệu, tin nhắn, hoặc chú thích (caption) mang tính mỉa mai thì tính vào I=1.  
  4. **Tình huống trớ trêu:** Ảnh chụp kết quả một hành động thất bại nhưng được trình bày như một thành tựu (Ví dụ: Nấu ăn cháy đen nhưng bày biện trang trọng).  
* **Lưu ý:** Một bức ảnh selfie đẹp, một bức ảnh phong cảnh, hoặc một bức ảnh chụp đồ vật bình thường \-\> **Gán I=0**. Đừng tự suy diễn cảm xúc của người trong ảnh nếu không có sự lố bịch rõ rệt.

## **BƯỚC 3: GÁN NHÃN MULTIMODAL (MM)**

**Mục tiêu:** Đây là nhãn quan trọng nhất. Xác định sự kết hợp giữa Text và Image có tạo ra mỉa mai hay không.

* **Cách làm:** Kết nối nghĩa của Bước 1 và Bước 2\.  
* **Công thức kiểm tra (Checklist):**  
  1. **Text (Khen) \+ Image (Tệ) \= MM=1.** (Ví dụ: Chữ khen "Cơm mẹ nấu ngon nhất" \- Ảnh là đĩa cơm cháy đen).  
  2. **Text (Bình thường) \+ Image (Mỉa mai) \= MM=1.** (Ví dụ: Chữ "Ok bạn" \- Ảnh là meme chế giễu).  
  3. **Text (Than vãn) \+ Image (Tích cực/Khoe khoang) \= MM=1.** (Ví dụ: Chữ "Khổ thân mình quá" \- Ảnh là đang đi du lịch sang chảnh). *Đây là kiểu mỉa mai khiêm tốn (Humblebrag).*  
  4. **Bối cảnh thực tế (Common Knowledge):** Nếu cả Text và Image đều trông bình thường nhưng chúng đi ngược lại sự thật hiển nhiên (Ví dụ: Chữ "Hà Nội mùa này không khí trong lành quá" \- Ảnh chụp bầu trời xám xịt bụi mịn).  
* **Quy tắc loại trừ:** Nếu Text và Image hỗ trợ nghĩa cho nhau theo hướng tích cực hoặc cùng tiêu cực một cách nghiêm túc (Ví dụ: Chữ "Buồn quá" \- Ảnh đang khóc thật) \-\> **Gán MM=0**.

## 

## **BẢNG QUYẾT ĐỊNH NHANH (CHO ANNOTATOR)**

| Nếu kết quả là... | Thì nhãn tổng hợp là... | Giải thích ngắn gọn |
| ----- | ----- | ----- |
| **T=0, I=0, MM=1** | **Implicit Sarcasm** | Mỉa mai ngầm định. Chỉ nảy sinh khi ghép cặp. |
| **T=1, I=0, MM=1** | **Text-driven Sarcasm** | Mỉa mai chủ yếu do lời nói. Ảnh là phụ. |
| **T=0, I=1, MM=1** | **Image-driven Sarcasm** | Mỉa mai do hình ảnh. Lời nói chỉ là dẫn chuyện. |
| **T=1, I=1, MM=1** | **Multi-driven Sarcasm** | Cả hai cùng "đá đểu" nhau. |
| **T=1, I=0, MM=0** | **Non-multimodal** | Chữ mỉa mai cái gì đó khác, không liên quan đến ảnh này. |

