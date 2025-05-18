# Hướng dẫn Triển khai RT-DETRv2 Siêu Tốc bằng Docker

Đây là hướng dẫn chi tiết các bước để cài đặt, cấu hình và chạy inference RT-DETRv2 (phiên bản PyTorch) trên một máy mới sử dụng Docker, đã được tự động hóa tối đa.

**Điều kiện tiên quyết:**
*   Bạn đã có sẵn mã nguồn của project này.
*   Người dùng cần cài đặt Docker và Docker Compose trên máy.

## 1. Build Docker Image (Chỉ làm 1 lần đầu hoặc khi có cập nhật)

Mở terminal (CMD, PowerShell, Git Bash,...) trên máy của bạn.
Di chuyển (`cd`) vào thư mục gốc của project (nơi chứa `Dockerfile`, `docker-compose.yml`).

Chạy lệnh sau từ thư mục gốc của project:
```bash
docker-compose up -d --build
```
*Lưu ý: Lệnh này sẽ build Docker image. Quá trình này bao gồm cài đặt tất cả thư viện cần thiết và tải model weights, nên có thể mất vài phút ở lần chạy đầu tiên. Các lần chạy sau (nếu không có thay đổi trong `Dockerfile` hoặc image chưa được build) sẽ nhanh hơn nhiều.*
*Thư mục project trên máy host sẽ được mount vào `/workspace` trong container.*

## 2. Chuẩn bị Dữ liệu Đầu vào (Trên Máy Host)

*   Copy hoặc di chuyển các file ảnh (`.jpg`, `.png`, ...) và/hoặc video (`.mp4`, `.avi`, ...) bạn muốn xử lý vào thư mục `input` **trong thư mục gốc của project trên máy host của bạn**.
*   Ví dụ: Nếu project của bạn là `MyRTDETR`, hãy đặt file vào `MyRTDETR/input/my_image.jpg`.
*   Các file này sẽ tự động xuất hiện trong `/workspace/input` bên trong container (do cơ chế mount volume của Docker).

## 3. Chạy Inference

Bạn có hai tùy chọn để chạy inference:

### Tùy chọn A: Chạy bằng Dòng lệnh (Bên trong Container)

1.  **Truy cập Container:**
    Mở một terminal trên máy host (vẫn ở thư mục gốc project), chạy lệnh sau để vào shell bash bên trong container đang chạy:
    ```bash
    docker-compose exec tensorrt-container bash
    ```
    *Bạn sẽ vào bên trong container, thư mục hiện tại là `/workspace`.*

2.  **Thực thi Lệnh Inference:**
    Từ shell bash bên trong container (đang ở `/workspace`), chạy script:
    ```bash
    bash ./run_inference.sh
    ```
    *   **Tùy chọn nâng cao cho `run_inference.sh`:**
        *   Chạy trên CPU: `bash ./run_inference.sh cpu`
        *   Chạy trên CPU với ngưỡng phát hiện 0.3: `bash ./run_inference.sh cpu 0.3`
        *   Chạy trên GPU (mặc định là `cuda:0`) với ngưỡng 0.7: `bash ./run_inference.sh cuda:0 0.7`
        *   Script sẽ tự động quét thư mục `/workspace/input`, xử lý các file ảnh/video hợp lệ, và lưu kết quả vào `/workspace/output`.

### Tùy chọn B: Chạy bằng Giao diện Đồ họa (GUI) (Trên Máy Host)

1.  **Đảm bảo Docker Container đang chạy:**
    Nếu bạn chưa khởi động container, hãy mở terminal trên máy host (trong thư mục gốc project) và chạy:
    ```bash
    docker-compose up -d
    ```
    *(Nếu bạn đã build image ở Bước 1, bạn không cần `--build` ở đây trừ khi có thay đổi trong Dockerfile).*

2.  **Khởi chạy GUI:**
    Mở một terminal **trên máy host của bạn** (ví dụ: CMD, PowerShell).
    Di chuyển (`cd`) vào thư mục gốc của project (nơi chứa file `gui.py`).
    Chạy lệnh sau để khởi động giao diện:
    ```bash
    python gui.py
    ```
    *(Hoặc `python3 gui.py` tùy theo cấu hình Python trên máy bạn).*

3.  **Sử dụng Giao diện:**
    *   Một cửa sổ giao diện sẽ xuất hiện.
    *   Bạn có thể sử dụng các nút "Open Input Directory" và "Open Output Directory" để quản lý file.
    *   Chọn "Device" (CPU hoặc GPU) và "Threshold" mong muốn.
    *   Nhấn nút "Start Inference".
    *   Quá trình xử lý sẽ bắt đầu, log sẽ hiển thị trong giao diện và có thanh tiến trình.
    *   Kết quả sẽ được lưu vào thư mục `output` trên máy host.

## 4. Lấy Kết quả Output (Trên Máy Host)

*   Dù bạn chạy bằng dòng lệnh hay GUI, các file ảnh/video kết quả (đã vẽ bounding box) sẽ nằm trong thư mục `output` **trong thư mục gốc của project trên máy host của bạn** (do đã mount volume).
*   Ví dụ: `MyRTDETR/output/my_image.jpg`.
*   Mở thư mục này để xem hoặc sử dụng kết quả.

## 5. Dọn dẹp (Trên Máy Host)

*   **Thoát Container (nếu đang ở trong shell container):** Gõ `exit` hoặc nhấn `Ctrl+D`.
*   **Dừng và Xóa Container:** Quay lại terminal trên máy host (vẫn đang ở thư mục gốc của project), chạy lệnh:
    ```bash
    docker-compose down
    ```
    *(Lệnh này sẽ dừng và xóa container `tensorrt-container`, nhưng không xóa Docker image đã build hoặc các file code/weights/input/output trên máy host).*