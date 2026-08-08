"""Voice command pipeline for the robot: openWakeWord -> Whisper STT -> command parser.

Architecture:
    USB mic (plughw:0,0) --> continuous wake-word listening (cheap, CPU)
                                       |
                              wake word detected
                                       |
                          record N seconds of audio
                                       |
                          Whisper transcribes locally
                                       |
                          parse_command() maps text -> action
                                       |
                          (hook: send to Roomba OI serial, or print for now)

Install (inside the l4t-ml or a dedicated container):
    pip3 install openwakeword pyaudio
    pip3 install faster-whisper   # CUDA-accelerated Whisper on Jetson

First run downloads openWakeWord's pretrained models automatically.
"""
import argparse
import queue
import re
import sys
import time

import numpy as np
import pyaudio

CHUNK = 1280          # openWakeWord expects 80ms frames at 16kHz
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16

# Commands the robot understands. Keep phrasing forgiving - Whisper output
# varies, so match on keywords rather than exact strings.
COMMAND_PATTERNS = {
    "stop":     [r"\bstop\b", r"\bhalt\b", r"\bfreeze\b"],
    "backward": [r"\bback\b", r"\breverse\b", r"\bbackward\b"],
    "left":     [r"\bleft\b"],
    "right":    [r"\bright\b"],
    "dock":     [r"\bdock\b", r"\bcharge\b", r"\bhome\b"],
    "come":     [r"\bcome\b(.*here)?", r"\bfollow\b"],
    "forward":  [r"\bforward\b", r"\bahead\b", r"\bgo\b", r"\bmove\b"],
}


def parse_command(text: str) -> str | None:
    """Map transcribed text to a known robot command, or None if no match."""
    text = text.lower().strip()
    for command, patterns in COMMAND_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, text):
                return command
    return None


def execute_command(command: str, roomba):
    """Send the parsed voice command to the Roomba via the OI serial driver.

    Re-asserts Safe mode before every command, since a bump/cliff event
    silently reverts Roomba to Passive mode (per the OI spec), which
    would otherwise strand subsequent voice commands with no response."""
    print(f"[ACTION] Re-asserting Safe mode...")
    roomba.safe_mode()
    print(f"[ACTION] Executing: {command}")
    action_map = {
        "forward":  roomba.forward,
        "backward": roomba.backward,
        "left":     roomba.turn_left,
        "right":    roomba.turn_right,
        "stop":     roomba.stop,
        "dock":     roomba.dock,
        "come":     roomba.forward,
    }
    action = action_map.get(command)
    if action:
        action()


def record_seconds(pa: pyaudio.PyAudio, seconds: float, device_index: int) -> np.ndarray:
    stream = pa.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
        frames_per_buffer=CHUNK, input_device_index=device_index,
    )
    frames = []
    for _ in range(int(RATE / CHUNK * seconds)):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))
    stream.stop_stream()
    stream.close()
    audio = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-index", type=int, default=None,
                     help="PyAudio input device index; omit to use default")
    ap.add_argument("--wakeword-threshold", type=float, default=0.7)
    ap.add_argument("--record-seconds", type=float, default=3.0,
                     help="How long to record command audio after wake word fires")
    ap.add_argument("--whisper-model", default="tiny",
                     help="tiny, base, small - tiny is fastest, good enough for short commands")
    ap.add_argument("--roomba-port", default="/dev/ttyUSB0")
    ap.add_argument("--dry-run", action="store_true", default=True,
                     help="Roomba commands print instead of sending (default True until hardware arrives)")
    ap.add_argument("--live", dest="dry_run", action="store_false",
                     help="Send real commands to the Roomba over serial")
    args = ap.parse_args()

    print("Loading openWakeWord...")
    from openwakeword.model import Model
    oww_model = Model(inference_framework="onnx")  # pretrained models auto-download

    print("Connecting to speaches (GPU Whisper service on :8000)...")
    import requests

    def transcribe_via_speaches(wav_path: str, model: str = "Systran/faster-whisper-tiny") -> str:
        with open(wav_path, "rb") as f:
            resp = requests.post(
                "http://localhost:8000/v1/audio/transcriptions",
                files={"file": f},
                data={"model": model},
                timeout=10,
            )
        resp.raise_for_status()
        return resp.json().get("text", "").strip()

    print(f"Connecting to Roomba (dry_run={args.dry_run})...")
    from roomba_control import RoombaOI
    roomba = RoombaOI(port=args.roomba_port, dry_run=args.dry_run)
    roomba.start()
    roomba.safe_mode()

    pa = pyaudio.PyAudio()

    if args.device_index is None:
        print("Available input devices:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"  [{i}] {info['name']}")
        print("Pass --device-index N to select one explicitly if the default is wrong.")

    stream = pa.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
        frames_per_buffer=CHUNK, input_device_index=args.device_index,
    )

    print("Listening for wake word... (Ctrl+C to stop)")
    try:
        while True:
            audio_chunk = np.frombuffer(
                stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16
            )
            prediction = oww_model.predict(audio_chunk)

            for wakeword, score in prediction.items():
                if score > args.wakeword_threshold:
                    print(f"\nWake word detected ({wakeword}, score={score:.2f})")
                    stream.stop_stream()
                    stream.close()

                    print(f"Recording command ({args.record_seconds}s)...")
                    audio = record_seconds(pa, args.record_seconds, args.device_index)

                    # Debug: save what was actually captured so we can inspect it
                    import wave
                    debug_path = "/tmp/last_command.wav"
                    with wave.open(debug_path, "wb") as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(pa.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes((audio * 32768).astype(np.int16).tobytes())
                    print(f"  (debug audio saved to {debug_path})")

                    print("Transcribing (GPU via speaches)...")
                    text = transcribe_via_speaches(debug_path)
                    print(f"Heard: \"{text}\"")

                    command = parse_command(text)
                    if command:
                        execute_command(command, roomba)
                    else:
                        print("No matching command recognised.")

                    oww_model.reset()
                    stream = pa.open(
                        format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                        frames_per_buffer=CHUNK, input_device_index=args.device_index,
                    )
                    print("\nListening for wake word...")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        roomba.close()


if __name__ == "__main__":
    main()
