"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import os
import glob
import time
import cv2 # Thêm cv2

from src.core import YAMLConfig

# Danh sách tên lớp COCO (80 lớp)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Sửa hàm draw để dùng cv2 vẽ lên frame (numpy array)
def draw_on_frame(frame, labels, boxes, scores, thrh=0.6):
    img_h, img_w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    line_type = cv2.LINE_AA
    text_color = (255, 255, 255) # White
    bg_color = (0, 0, 0) # Black
    box_color = (0, 0, 255) # Red (BGR)
    box_thickness = 2

    scr = scores[0]
    lab_indices = labels[0][scr > thrh]
    box = boxes[0][scr > thrh]
    scrs = scores[0][scr > thrh]

    for j, b in enumerate(box):
        class_id = lab_indices[j].item()
        class_name = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else f"ID:{class_id}"
        score_text = f"{scrs[j].item():.2f}"
        label_text = f"{class_name} {score_text}"

        # Lấy tọa độ box nguyên
        x1, y1, x2, y2 = map(int, b)
        # Đảm bảo box không ra ngoài ảnh
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)

        # Tính kích thước text để vẽ nền
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        text_x = x1
        text_y = y1 - th - baseline // 2

        # Vẽ nền đen cho text
        cv2.rectangle(frame, (text_x, text_y + baseline), (text_x + tw, text_y - th - baseline//2), bg_color, -1)
        # Vẽ text trắng
        cv2.putText(frame, label_text, (text_x, text_y), font, font_scale, text_color, font_thickness, line_type)
        # Vẽ bounding box đỏ
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

    return frame # Trả về frame đã vẽ

# Hàm xử lý ảnh (tách ra từ main cũ)
def process_image_folder(model, transforms, args):
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory for images: {args.output_dir}")

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.input_path, ext)))

    if not image_files:
        print(f"No images found in {args.input_path}")
        return

    print(f"Found {len(image_files)} images in {args.input_path}")

    total_time = 0
    processed_count = 0
    for img_path in image_files:
        try:
            print(f"Processing Image: {img_path}")
            start_time = time.time()

            im_pil = Image.open(img_path).convert('RGB')
            w, h = im_pil.size
            orig_size = torch.tensor([[w, h]]).to(args.device)

            im_data = transforms(im_pil)[None].to(args.device)

            output = model(im_data, orig_size)
            labels, boxes, scores = output

            end_time = time.time()
            processing_time = end_time - start_time
            total_time += processing_time
            print(f"  Inference time: {processing_time:.4f} seconds")

            base_filename = os.path.basename(img_path)
            output_path = os.path.join(args.output_dir, base_filename)

            # Chuyển im_pil sang numpy để dùng hàm draw_on_frame (BGR)
            frame_to_draw = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
            processed_frame = draw_on_frame(frame_to_draw, labels, boxes, scores, thrh=args.threshold)
            cv2.imwrite(output_path, processed_frame)
            print(f"  Saved result to: {output_path}")
            processed_count += 1

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    avg_time = total_time / processed_count if processed_count else 0
    print(f"\nFinished processing {processed_count} images from the folder.")

# Hàm xử lý video
def process_video(model, transforms, args):
    print(f"Processing Video: {args.input_path}")
    cap = cv2.VideoCapture(args.input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {args.input_path}")
        return

    # Lấy thông tin video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video Info: {frame_width}x{frame_height} @ {fps:.2f} FPS, ~{frame_count} frames")

    # Tạo thư mục output và video writer
    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = os.path.basename(args.input_path)
    output_filename = os.path.splitext(base_filename)[0] + '_output.mp4'
    output_path = os.path.join(args.output_dir, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec cho MP4
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    print(f"  Output video will be saved to: {output_path}")

    frame_idx = 0
    total_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # Hết video

        start_time = time.time()
        frame_idx += 1

        # Chuyển BGR frame sang RGB PIL Image để transform
        im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        orig_size = torch.tensor([[frame_width, frame_height]]).to(args.device)
        im_data = transforms(im_pil)[None].to(args.device)

        # Inference
        output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Vẽ kết quả lên frame gốc (BGR)
        processed_frame = draw_on_frame(frame, labels, boxes, scores, thrh=args.threshold)

        # Ghi frame vào video output
        writer.write(processed_frame)

        end_time = time.time()
        processing_time = end_time - start_time
        total_time += processing_time
        print(f"  Processed frame {frame_idx}/{frame_count} - Time: {processing_time:.4f}s", end='\r')

    print(f"\nFinished processing video.")
    avg_time = total_time / frame_idx if frame_idx else 0
    print(f"Average inference time per frame: {avg_time:.4f} seconds")

    # Giải phóng tài nguyên
    cap.release()
    writer.release()
    print(f"Output video saved: {output_path}")

def main(args):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True) # Use weights_only=True if possible
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    cfg.model.load_state_dict(state)
    print("Model state loaded successfully.")

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        @torch.no_grad()
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)
    model.eval()
    print(f"Model moved to device: {args.device}")

    # Định nghĩa transforms
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    # Tạo thư mục output
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Lấy danh sách tất cả các file trong thư mục input
    input_files = glob.glob(os.path.join(args.input_dir, '*.*')) # Lấy tất cả file

    if not input_files:
        print(f"No files found in {args.input_dir}")
        return

    print(f"Found {len(input_files)} files in {args.input_dir}. Processing...")

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

    # Lặp qua từng file và xử lý
    for file_path in input_files:
        _, file_ext = os.path.splitext(file_path)
        file_ext = file_ext.lower()

        if file_ext in image_extensions:
            # Tạo args tạm thời cho hàm xử lý ảnh
            img_args = argparse.Namespace(**vars(args)) # Copy args
            img_args.input_path = file_path # Đặt input_path là file ảnh hiện tại
            process_single_image(model, transforms, img_args) # Gọi hàm xử lý ảnh đơn lẻ

        elif file_ext in video_extensions:
            # Tạo args tạm thời cho hàm xử lý video
            vid_args = argparse.Namespace(**vars(args)) # Copy args
            vid_args.input_path = file_path # Đặt input_path là file video hiện tại
            process_video(model, transforms, vid_args)

        else:
            print(f"Skipping unsupported file type: {file_path}")

    print("\nFinished processing all supported files.")

# Hàm xử lý ảnh đơn lẻ (refactor từ process_image_folder)
def process_single_image(model, transforms, args):
    img_path = args.input_path # Lấy đường dẫn từ args
    try:
        print(f"Processing Image: {img_path}")
        start_time = time.time()

        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(args.device)

        im_data = transforms(im_pil)[None].to(args.device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        end_time = time.time()
        processing_time = end_time - start_time
        print(f"  Inference time: {processing_time:.4f} seconds")

        base_filename = os.path.basename(img_path)
        output_path = os.path.join(args.output_dir, base_filename)

        frame_to_draw = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
        processed_frame = draw_on_frame(frame_to_draw, labels, boxes, scores, thrh=args.threshold)
        cv2.imwrite(output_path, processed_frame)
        print(f"  Saved result to: {output_path}")

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to model config file')
    parser.add_argument('-r', '--resume', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--input-dir', type=str, default='./input', help='Directory containing input images and/or videos') # Sửa help text
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save output images/videos')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='Device to use (e.g., cpu, cuda:0)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for drawing boxes')
    args = parser.parse_args()

    if 'cuda' in args.device and not torch.cuda.is_available():
        print(f"CUDA device '{args.device}' requested but CUDA not available. Using CPU instead.")
        args.device = 'cpu'

    main(args)
