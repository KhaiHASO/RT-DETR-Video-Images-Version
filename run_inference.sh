#!/bin/bash

# Default values
DEFAULT_DEVICE="cuda:0"
DEFAULT_THRESHOLD="0.5"

# Use provided arguments or defaults
DEVICE="${1:-$DEFAULT_DEVICE}"
THRESHOLD="${2:-$DEFAULT_THRESHOLD}"

echo "Running inference with device: $DEVICE and threshold: $THRESHOLD"

python -m references.deploy.rtdetrv2_torch \
  -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml \
  -r weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth \
  --input-dir ./input \
  --output-dir ./output \
  --device "$DEVICE" \
  --threshold "$THRESHOLD"

echo "Inference complete. Output is in ./output directory." 