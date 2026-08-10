<img width="4000" height="3008" alt="irobot" src="https://github.com/user-attachments/assets/f2f7bfa9-6e85-4bdd-b2c5-d169262183dd" />
# Edge Vision Actuation

Real-time object detection on an NVIDIA Jetson Orin Nano Super, from a physical CSI camera
through GStreamer/Argus into a PyTorch/TensorRT YOLOv8 pipeline — benchmarked, optimised,
and built as a deployment engineering exercise rather than a tutorial walkthrough.

## Demo

| Baseline (PyTorch, FP32) | Optimised (TensorRT, FP16) |
|---|---|
| `demo/detection_demo_baseline.mp4` | `demo/detection_demo_tensorrt.mp4` |
| 13.9 FPS | 17.6 FPS (+27%) |

## Hardware

- NVIDIA Jetson Orin Nano Super (8GB), MAXN SUPER power mode
- Waveshare IMX219-83 stereo camera module (single sensor used here; stereo depth is a
  planned extension)
- 1TB NVMe SSD (Docker storage, model weights, project files)

## Architecture

```
IMX219 camera (CSI)
      |
nvarguscamerasrc (Argus daemon)
      |
nvvidconv (NV12 -> BGRx)
      |
videoconvert (BGRx -> BGR)
      |
OpenCV appsink  --------->  YOLOv8-nano (PyTorch .pt or TensorRT .engine)
                                   |
                          annotated frame (cv2)
                                   |
                          MP4 writer + still frame
```

## Running it

Built and tested inside the `dustynv/l4t-ml:r36.4.0` container (JetPack 6.2.1, L4T 36.4.7),
which ships a Jetson-correct OpenCV build with GStreamer/Argus support out of the box.

```bash
docker run --rm -it --runtime nvidia --network host \
  --name l4t-ml-dev \
  --privileged \
  -v /run/systemd/resolve/resolv.conf:/etc/resolv.conf \
  -v /path/to/this/repo:/projects \
  -v /tmp/argus_socket:/tmp/argus_socket \
  dustynv/l4t-ml:r36.4.0

pip3 install ultralytics
cd /projects
python3 live_detect.py --sensor-id 0
```

To use the TensorRT-optimised engine instead of the plain PyTorch weights:

```bash
yolo export model=yolov8n.pt format=engine half=True device=0
# then edit live_detect.py's model = YOLO("yolov8n.pt") to point at yolov8n.engine
```

See `BENCHMARKS.md` for full results and notes on what the TensorRT speedup actually reflects.

## Notes on getting this working

Getting a clean camera pipeline running on Jetson inside a container is not plug-and-play.
In order, the real blockers were:

1. **IMX219 driver not enabled by default on Orin** — fixed via `jetson-io.py`, selecting the
   IMX219 dual-camera overlay.
2. **GStreamer plugin dependency chain incomplete in minimal containers** — the NVIDIA-specific
   plugins (`libgstnvarguscamerasrc.so` etc.) need `libgstvideo`, `libgstallocators`, `libEGL`,
   and `libGLESv2` all present; generic pip-installed OpenCV lacks GStreamer support entirely.
   `dustynv/l4t-ml` solves this by shipping a correctly-built OpenCV from the start.
3. **Argus daemon socket not shared into the container** — needs an explicit
   `-v /tmp/argus_socket:/tmp/argus_socket` mount, in addition to `--privileged` for device access.
4. **NvMap memory exhaustion from repeated failed camera sessions** — resolved with a clean
   reboot; a stuck Argus session from earlier attempts will silently starve later ones of
   camera memory.
5. **OpenCV `appsink` needs the pipeline given time to reach PLAYING state** — the original
   attempt read frames immediately after opening `VideoCapture`, before the pipeline had
   started producing them. A short sleep after opening fixed frame reads reliably.

## Roadmap

- [x] Phase 1: PyTorch -> TensorRT object detection on live camera feed
- [ ] Phase 2: Closed-loop pan-tilt tracking (servo-driven)
- [ ] Phase 3: LeRobot imitation-learning policy on a low-cost robot arm (SO-ARM101)
- [ ] Stereo depth from the second IMX219 sensor

## Author

Edmund Jeevan D'Souza — [edmund.tech](https://edmund.tech) | [LinkedIn](https://www.linkedin.com/in/edmundjeevan/)
