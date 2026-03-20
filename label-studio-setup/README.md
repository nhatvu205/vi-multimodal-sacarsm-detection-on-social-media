# Cài đặt Label Studio - Gán nhãn dữ liệu đa phương thức

Hướng dẫn cài đặt Label Studio trên máy cá nhân bằng Docker để gán nhãn dữ liệu gồm ảnh và văn bản.

---

## 0. Yêu cầu

Chỉ cần cài một công cụ duy nhất:

- **Docker Desktop** — https://www.docker.com/products/docker-desktop/

Sau khi cài xong, mở Docker Desktop và đảm bảo nó đang chạy (icon Docker xuất hiện ở thanh taskbar góc dưới bên phải). Nếu Docker Desktop yêu cầu cài WSL 2, đồng ý và khởi động lại máy.

---

## 1. Clone repo và cấu hình môi trường

**Clone repo và di chuyển vào thư mục:**

```powershell
git clone <repository-url>
cd label-studio-setup
```

**Tạo file `.env` từ template:**

```powershell
Copy-Item .env.example .env
```

**Điền thông tin vào file `.env`:**

Mở file `.env` bằng bất kỳ text editor nào. Liên hệ trưởng nhóm để lấy các thông tin cần điền, đặc biệt là:

```
LS_USERNAME=        # Email tài khoản admin
LS_PASSWORD=        # Mật khẩu tài khoản admin
LS_API_KEY=         # Điền sau khi đã đăng nhập lần đầu
PROJECT_ID=         # Điền sau khi trưởng nhóm tạo project
```

> Không commit file `.env` lên Git. File này đã được thêm vào `.gitignore`.

---

## 2. Khởi động Docker và kiểm tra

**Build image cho scripts (chỉ cần làm một lần):**

```powershell
docker compose build scripts
```

**Khởi động toàn bộ hệ thống:**

```powershell
docker compose up -d
```

Lần đầu chạy sẽ tải các Docker image về (~1 GB), mất khoảng 5–10 phút tùy tốc độ mạng.

**Kiểm tra trạng thái container:**

```powershell
docker compose ps
```

Kết quả hợp lệ trông như sau — cột STATUS phải là `running` hoặc `healthy`:

```
NAME               STATUS
labelstudio_app    Up (running)
labelstudio_db     Up (healthy)
```

**Mở trình duyệt và truy cập:** http://localhost:8080

Đăng nhập bằng `LS_USERNAME` và `LS_PASSWORD` đã điền trong `.env`.

---

### Các lỗi thường gặp

**Trình duyệt báo "This site can't be reached"**

Container có thể vẫn đang khởi động. Chờ thêm 30 giây rồi thử lại. Nếu vẫn không được, kiểm tra log:

```powershell
docker compose logs -f label-studio
```

Tìm dòng `Starting development server at http://0.0.0.0:8080/` — nếu dòng này xuất hiện thì server đã sẵn sàng.

Nếu dùng trình duyệt, đảm bảo dùng `http://` chứ không phải `https://`. Một số trình duyệt tự chuyển sang HTTPS và gây lỗi kết nối.

---

**Container không lên hoặc liên tục restart**

```powershell
docker compose logs db
docker compose logs label-studio
```

Nếu database bị lỗi khởi tạo, reset hoàn toàn bằng lệnh sau **(xóa toàn bộ dữ liệu)**:

```powershell
docker compose down -v
docker compose up -d
```

---

**Port 8080 đã bị chiếm bởi ứng dụng khác**

Mở file `.env` và đổi port:

```
LS_PORT=8090
```

Sau đó khởi động lại:

```powershell
docker compose down
docker compose up -d
```

Truy cập bằng địa chỉ mới: http://localhost:8090

---

**Quên mật khẩu**

```powershell
docker compose exec label-studio label-studio reset_password --username admin@example.com
```

Thay `admin@example.com` bằng giá trị `LS_USERNAME` trong file `.env`.

---

**Dừng server (giữ nguyên dữ liệu)**

```powershell
docker compose down
```

---

## 3. Nhập dữ liệu và hướng dẫn gán nhãn

> **Sắp cập nhật.** Phần này sẽ bao gồm hướng dẫn chuẩn bị dữ liệu, nhập task, quy trình gán nhãn và xuất kết quả. Liên hệ trưởng nhóm để được hướng dẫn trực tiếp trong thời gian chờ.
