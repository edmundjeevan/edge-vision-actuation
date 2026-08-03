"""Live object detection on the Jetson's CSI camera using pretrained YOLOv8-nano.

Reads frames via GStreamer (nvarguscamerasrc -> OpenCV), runs YOLOv8 inference,
draws bounding boxes, and saves annotated frames + a short demo clip.

Run inside the PyTorch jetson-container:
    python3 live_detect.py --sensor-id 0 --warmup 60 --duration 20
"""
import argparse
import time

import cv2
from ultralytics import YOLO


def gst_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method=0 ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor-id", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=60, help="frames to skip while ISP auto-exposure settles")
    ap.add_argument("--duration", type=int, default=20, help="seconds to capture after warmup")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", default="/projects/edge-vision-actuation/demo")
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)

    print("Loading YOLOv8-nano TensorRT engine...")
    model = YOLO("yolov8n.engine")

    pipeline = gst_pipeline(args.sensor_id, args.width, args.height, args.fps)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise SystemExit("Failed to open camera via GStreamer pipeline")

    print("Waiting 2s for pipeline to start streaming...")
    time.sleep(2)

    print(f"Warming up ISP for {args.warmup} frames...")
    fails = 0
    for i in range(args.warmup):
        ok, _ = cap.read()
        if not ok:
            fails += 1
            time.sleep(0.05)
    if fails:
        print(f"  {fails} of {args.warmup} warmup frames failed to read")

    print(f"Recording {args.duration}s of detections...")
    writer = None
    t0, frame_count, last_fps_print = time.time(), 0, time.time()

    while time.time() - t0 < args.duration:
        ok, frame = cap.read()
        if not ok:
            continue

        results = model(frame, conf=args.conf, verbose=False)
        annotated = results[0].plot()

        if writer is None:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(f"{args.out}/detection_demo.mp4", fourcc, args.fps, (w, h))
        writer.write(annotated)

        frame_count += 1
        if time.time() - last_fps_print > 2:
            elapsed = time.time() - t0
            print(f"  {frame_count} frames, {frame_count/elapsed:.1f} FPS")
            last_fps_print = time.time()

    cv2.imwrite(f"{args.out}/last_frame.jpg", annotated)

    cap.release()
    if writer:
        writer.release()

    print(f"Done. Saved: {args.out}/detection_demo.mp4 and {args.out}/last_frame.jpg")
    print(f"Total frames: {frame_count}, avg FPS: {frame_count/args.duration:.1f}")


if __name__ == "__main__":
    main()
