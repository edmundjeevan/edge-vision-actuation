# Benchmark Log - Edge Vision Actuation Project

Hardware: NVIDIA Jetson Orin Nano Super (8GB), MAXN SUPER power mode
Camera: Waveshare IMX219-83 stereo (single sensor used, 1280x720)
Model: YOLOv8-nano (pretrained, COCO 80 classes)

| Date       | Backend            | Precision | Resolution | Avg FPS | Frames | Notes |
|------------|---------------------|-----------|------------|---------|--------|-------|
| 2026-08-02 | PyTorch (.pt)       | FP32      | 1280x720   | 13.9    | 278    | Baseline, GStreamer/Argus pipeline via l4t-ml container |
| 2026-08-03 | TensorRT (.engine)  | FP16      | 1280x720   | 17.6    | 352    | +27% over baseline. Camera/ISP pipeline now likely secondary bottleneck alongside inference |

## Export command
yolo export model=yolov8n.pt format=engine half=True device=0

Export time: 432s (TensorRT engine build), one-time cost per model/device pair.

## Observations
- TensorRT FP16 gain (27%) is smaller than pure-inference speedups typically reported for TensorRT
  on this model class, suggesting the end-to-end pipeline (camera capture, ISP warmup, colour
  conversion via nvvidconv/videoconvert) is a meaningful part of total frame time, not just the
  neural network forward pass.
- Next optimization candidates: reduce input resolution, profile each pipeline stage individually
  (camera capture vs preprocessing vs inference vs draw/encode) to find the actual bottleneck.
