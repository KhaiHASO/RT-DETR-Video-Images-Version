# tensorrt:23.01-py3 (8.5.2.2)
FROM nvcr.io/nvidia/tensorrt:23.01-py3

WORKDIR /workspace

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install opencv-python && \
    pip install Pillow

# Create directories for weights, input, and output
RUN mkdir -p /workspace/weights /workspace/input /workspace/output

# Download pre-trained weights
RUN wget https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth -O /workspace/weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth

# Copy the run script and make it executable (optional, can be done on host)
# COPY run_inference.sh .
# RUN chmod +x run_inference.sh

CMD ["/bin/bash"]
