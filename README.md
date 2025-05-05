# Hướng dẫn Triển khai RT-DETRv2 PyTorch bằng Docker

Đây là hướng dẫn chi tiết các bước để cài đặt, cấu hình và chạy inference RT-DETRv2 (phiên bản PyTorch) trên một máy mới sử dụng Docker.

## 1. Chuẩn bị trên Máy Mới (Host Machine)

*   **Cài đặt:** Đảm bảo Docker và Docker Compose đã được cài đặt. Nếu muốn sử dụng GPU NVIDIA, cần cài đặt driver NVIDIA mới nhất và NVIDIA Container Toolkit.
*   **Lấy Code:** Clone repository RT-DETR từ GitHub.
    ```bash
    git clone https://github.com/lyuwenyu/RT-DETR.git
    ```
*   **Di chuyển vào Thư mục:** Mở terminal và `cd` vào thư mục code PyTorch của RT-DETRv2.
    ```bash
    cd RT-DETR/rtdetrv2_pytorch
    ```

## 2. Build và Chạy Docker Container (Host Machine)

*   **Build Image & Chạy Container:** Lệnh này sẽ đọc `Dockerfile` để build image (nếu chưa có) và đọc `docker-compose.yml` để tạo, cấu hình (mount volume `./:/workspace`), và chạy container `tensorrt-container` ở chế độ nền.
    ```bash
    docker-compose up -d --build
    ```
    *Chờ quá trình build và khởi động hoàn tất. Thư mục `rtdetrv2_pytorch` trên máy host sẽ được mount vào `/workspace` trong container.*

## 3. Thiết lập bên trong Container (Chỉ chạy lần đầu)

*   **Truy cập Container:** Mở một shell bash bên trong container đang chạy.
    ```bash
    docker-compose exec tensorrt-container bash
    ```
    *Bạn sẽ vào bên trong container, thư mục hiện tại là `/workspace`.*

*   **Cài đặt Thư viện Hệ thống:** Cài đặt thư viện đồ họa `libgl1-mesa-glx` mà OpenCV cần.
    ```bash
    apt-get update && apt-get install -y libgl1-mesa-glx
    ```
*   **Cài đặt OpenCV Python:** Cần cho việc xử lý video và vẽ kết quả.
    ```bash
    pip install opencv-python
    ```
*   **Tải Pretrained Weights:** Tải trọng số đã huấn luyện sẵn (ví dụ: RT-DETRv2-S).
    ```bash
    mkdir -p weights
    wget https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth -O weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth
    ```
*   **Tạo Thư mục Input/Output:** Tạo các thư mục để chứa dữ liệu vào và kết quả ra.
    ```bash
    mkdir -p input output
    ```

## 4. Chuẩn bị Dữ liệu Đầu vào

*   **Đặt Ảnh/Video vào Thư mục Input:** Copy hoặc di chuyển các file ảnh (`.jpg`, `.png`, ...) và/hoặc video (`.mp4`, `.avi`, ...) bạn muốn xử lý vào thư mục `RT-DETR/rtdetrv2_pytorch/input` **trên máy host của bạn**. Các file này sẽ tự động xuất hiện trong `/workspace/input` bên trong container do cơ chế mount volume.

## 5. Chạy Inference (Bên trong Container)

*   **Thực thi Lệnh:** Từ shell bash bên trong container (đang ở `/workspace`), chạy script inference. Script này sẽ tự động quét thư mục `/workspace/input`, xử lý các file ảnh/video hợp lệ, và lưu kết quả vào `/workspace/output`.
    ```bash
    python -m references.deploy.rtdetrv2_torch \
      -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
      -r weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
      --input-dir ./input \
      --output-dir ./output \
      --device cuda:0 \
      --threshold 0.5
      # Lưu ý:
      # - Thay đổi --device cpu nếu không có GPU hoặc không cấu hình NVIDIA Container Toolkit.
      # - Điều chỉnh --threshold (0.0 đến 1.0) để thay đổi độ nhạy phát hiện.
    ```

## 6. Lấy Kết quả Output

*   **Kiểm tra Thư mục Output:** Các file ảnh/video kết quả (đã vẽ bounding box) sẽ nằm trong thư mục `RT-DETR/rtdetrv2_pytorch/output` **trên máy host của bạn** (do đã mount volume).
*   Mở thư mục này để xem hoặc sử dụng kết quả.

## 7. Dọn dẹp (Host Machine)

*   **Thoát Container:** Bên trong shell container, gõ `exit` hoặc nhấn `Ctrl+D`.
*   **Dừng và Xóa Container:** Quay lại terminal trên máy host (vẫn đang ở thư mục `RT-DETR/rtdetrv2_pytorch`), chạy lệnh:
    ```bash
    docker-compose down
    ```
    *(Lệnh này sẽ dừng và xóa container `tensorrt-container`, nhưng không xóa Docker image đã build hoặc các file code/weights/input/output trên máy host).*