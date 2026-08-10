<img width="4000" height="3008" alt="irobot" src="https://github.com/user-attachments/assets/f2f7bfa9-6e85-4bdd-b2c5-d169262183dd" />
# Edge Vision Actuation

Real-time object detection and voice-controlled navigation on an NVIDIA Jetson Orin Nano Super,
mounted on a Roomba chassis. Built as a deployment engineering exercise - closing the full loop
from camera and microphone input through to real motor control - rather than a tutorial
walkthrough.

## Demo

| Baseline (PyTorch, FP32) | Optimised (TensorRT, FP16) |
|---|---|
| demo/detection_demo_baseline.mp4 | demo/detection_demo_tensorrt.mp4 |
| 13.9 FPS | 17.6 FPS (+27%) |

**Voice-controlled robot (live hardware):** wake word -> GPU-accelerated Whisper -> command
parsing -> real Roomba motor control over serial, with obstacle (bump) safety confirmed working
end-to-end. See `voice_command.py` and `roomba_control.py`.

## The Rig

![Robot rig - full view](images/rig-full.jpg)

A Jetson Orin Nano Super mounted on a Roomba 530 chassis, driven via a USB serial link to the
Roomba's Open Interface (OI) port. Powered untethered by a 100W USB-C PD power bank through a
barrel-jack trigger cable. Voice commands come in through a USB microphone, processed on-device
(wake word + GPU-accelerated Whisper) and translated into real motor commands over serial.

| Component | Detail |
|---|---|
| Compute | NVIDIA Jetson Orin Nano Super (8GB), MAXN SUPER |
| Base | iRobot Roomba 530 (Open Interface, 115200 baud) |
| Camera | Waveshare IMX219-83 stereo (single sensor in use) |
| Audio | USB sound card (mic + speaker) |
| Power | 100W USB-C PD power bank via 5.5x2.5mm barrel trigger cable |
| Serial link | USB-to-Mini-DIN cable, Roomba OI port |

<p align="center">
  <img src="images/rig-top.jpg" width="32%" alt="Top-mounted Jetson and camera" />
  <img src="images/rig-underside.jpg" width="32%" alt="Roomba OI serial connection" />
  <img src="images/rig-power.jpg" width="32%" alt="Power bank and trigger cable" />
</p>

## Architecture

```
IMX219 camera (CSI)                          USB microphone
      |                                            |
nvarguscamerasrc (Argus daemon)          openWakeWord (wake detection)
      |                                            |
nvvidconv (NV12 -> BGRx)                  speaches (GPU Whisper, CUDA)
      |                                            |
videoconvert (BGRx -> BGR)                 command parser (regex)
      |                                            |
OpenCV appsink  --------->  YOLOv8-nano            |
    (PyTorch .pt or TensorRT .engine)              v
                                            roomba_control.py (OI serial)
                                                    |
                                            Roomba 530 drive motors
```

## Running it - vision pipeline

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

## Running it - voice control

Runs directly on the Jetson host (not in a container) for simplest USB audio device access.
Requires a running `speaches` container for GPU-accelerated transcription:

```bash
docker run -d \
  --name speaches \
  --runtime nvidia \
  --shm-size=2g \
  -p 8000:8000 \
  -v speaches-models:/home/ubuntu/.cache/huggingface/hub \
  cbinckly/speaches:0.9.0-l4t-cuda-12.6.11-arch87
```

Then:

```bash
pip3 install openwakeword pyaudio requests pyserial
python3 voice_command.py --device-index <your-mic-index> --wakeword-threshold 0.7 --live
```

Omit `--live` to dry-run (prints the serial bytes it would send without touching real hardware) -
useful for testing the voice/parsing pipeline before connecting to the robot.

## Notes on getting this working

Getting a clean camera pipeline running on Jetson inside a container is not plug-and-play, and
neither is closing the loop with real hardware. In order, the real blockers were:

1. **IMX219 driver not enabled by default on Orin** - fixed via `jetson-io.py`, selecting the
   IMX219 dual-camera overlay.
2. **GStreamer plugin dependency chain incomplete in minimal containers** - the NVIDIA-specific
   plugins (`libgstnvarguscamerasrc.so` etc.) need `libgstvideo`, `libgstallocators`, `libEGL`,
   and `libGLESv2` all present; generic pip-installed OpenCV lacks GStreamer support entirely.
   `dustynv/l4t-ml` solves this by shipping a correctly-built OpenCV from the start - and
   critically, `pip uninstall opencv-*` inside that container can silently corrupt the system
   OpenCV install by removing shared files. Never uninstall opencv inside it.
3. **Argus daemon socket not shared into the container** - needs an explicit
   `-v /tmp/argus_socket:/tmp/argus_socket` mount, in addition to `--privileged` for device access.
4. **NvMap memory exhaustion from repeated failed camera sessions** - resolved with a clean
   reboot; a stuck Argus session from earlier attempts will silently starve later ones of
   camera memory.
5. **OpenCV `appsink` needs the pipeline given time to reach PLAYING state** - the original
   attempt read frames immediately after opening `VideoCapture`, before the pipeline had
   started producing them. A short sleep after opening fixed frame reads reliably.
6. **CUDA OOM in a Docker container despite plenty of free system RAM** - fixed with
   `--shm-size=2g` on the container; Docker's default shared-memory limit (64MB) is too small
   for CUDA/ctranslate2 allocator behaviour even when host RAM is nowhere near full.
7. **`ctranslate2`'s PyPI wheel has no CUDA support on Jetson/ARM64** - solved by running Whisper
   inference in a separate purpose-built container (`speaches`) rather than in-process.
8. **Roomba OI reverts to Passive mode after any bump/cliff safety event** - this silently
   stranded the voice pipeline, since subsequent drive commands were sent while the robot had
   already dropped out of Safe mode. Fixed by re-asserting Safe mode before every command.
9. **USB audio device index shifts** when other USB devices (like the Roomba's serial adapter)
   are connected - always re-check `pyaudio` device indices after changing what's plugged in.

## Roadmap

- [x] Phase 1: PyTorch -> TensorRT object detection on live camera feed
- [x] Phase 2: Voice-controlled navigation - wake word, GPU Whisper, live Roomba serial control,
      obstacle safety confirmed on real hardware
- [ ] Phase 3: Camera-based obstacle avoidance merged with voice control (in progress)
- [ ] Phase 4: Closed-loop pan-tilt camera tracking (servo-driven)
- [ ] Phase 5: Remote operation via Ring two-way talk as a voice relay
- [ ] Stereo depth from the second IMX219 sensor

## Author

Edmund Jeevan D'Souza - [edmund.tech](https://edmund.tech) | [LinkedIn](https://www.linkedin.com/in/edmundjeevan/)
