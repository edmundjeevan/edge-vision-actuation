"""Roomba 500-series Open Interface (OI) serial driver.

Implements the core OI commands needed for voice-controlled navigation:
mode control, direct wheel drive, dock/charge, and basic sensor reads.

Reference: iRobot Roomba 500 Open Interface (OI) Specification.
Default baud: 115200. Connector: 7-pin Mini-DIN under the top cover.

DRY-RUN MODE: if no serial port is available (e.g. testing before the
USB-to-Mini-DIN cable arrives), pass dry_run=True and every command just
prints what it would have sent - lets you test the command layer and the
voice pipeline's wiring today, with the real hardware plugged in later
by changing one flag.

Usage:
    roomba = RoombaOI(port="/dev/ttyUSB0", dry_run=False)
    roomba.start()
    roomba.safe_mode()
    roomba.drive_direct(200, 200)   # both wheels forward, mm/s
    roomba.stop()
    roomba.dock()
"""
import struct
import time

try:
    import serial
except ImportError:
    serial = None  # allows dry-run testing without pyserial installed


# --- OI Op-codes (Roomba 500 spec) ---
OI_START        = 128
OI_SAFE         = 131
OI_FULL         = 132
OI_POWER_DOWN   = 133
OI_DRIVE_DIRECT = 145   # takes right velocity, left velocity (mm/s, signed 16-bit)
OI_SEEK_DOCK    = 143   # attempt to dock (same as auto-dock on low battery)
OI_SENSORS      = 142   # request a single sensor packet
OI_LEDS         = 139

# Sensor packet IDs (subset - extend as needed)
SENSOR_BUMPS_WHEELDROPS = 7
SENSOR_CLIFF_LEFT       = 9
SENSOR_CLIFF_RIGHT      = 12
SENSOR_BATTERY_CHARGE   = 25   # mAh, 2 bytes
SENSOR_BATTERY_CAPACITY = 26   # mAh, 2 bytes

# Speed limits (mm/s) - Roomba OI accepts -500 to 500, but keep it gentle
# for a first-pass indoor robot sharing space with people/pets/furniture.
MAX_SPEED = 200
TURN_SPEED = 150


class RoombaOI:
    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200, dry_run: bool = False):
        self.dry_run = dry_run or serial is None
        if self.dry_run:
            print(f"[DRY-RUN] RoombaOI initialised (no real serial port opened; "
                  f"would use {port} @ {baud})")
            self.ser = None
        else:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(1)  # let the port settle

    # --- low-level send ---
    def _send(self, *bytes_):
        if self.dry_run:
            print(f"[DRY-RUN] send bytes: {list(bytes_)}")
            return
        self.ser.write(bytes(bytes_))

    # --- mode control ---
    def start(self):
        """Enter Passive mode - required before anything else."""
        self._send(OI_START)
        time.sleep(0.1)

    def safe_mode(self):
        """Safe mode: full control, but Roomba still stops itself on cliffs/wheel-drop.
        Use this for all early testing."""
        self._send(OI_SAFE)
        time.sleep(0.1)

    def full_mode(self):
        """Full mode: total control, safety behaviours disabled. Only use once
        the pipeline is proven reliable and you have a manual kill-switch plan."""
        self._send(OI_FULL)
        time.sleep(0.1)

    def power_down(self):
        self._send(OI_POWER_DOWN)

    # --- driving ---
    def drive_direct(self, right_mm_s: int, left_mm_s: int):
        """Direct per-wheel velocity control, -500 to 500 mm/s each.
        Positive = forward, negative = backward, independently per wheel
        so this covers driving straight, turning, and spinning in place."""
        right_mm_s = max(-500, min(500, right_mm_s))
        left_mm_s = max(-500, min(500, left_mm_s))
        r_hi, r_lo = struct.pack(">h", right_mm_s)
        l_hi, l_lo = struct.pack(">h", left_mm_s)
        self._send(OI_DRIVE_DIRECT, r_hi, r_lo, l_hi, l_lo)

    def forward(self, speed: int = MAX_SPEED):
        self.drive_direct(speed, speed)

    def backward(self, speed: int = MAX_SPEED):
        self.drive_direct(-speed, -speed)

    def turn_left(self, speed: int = TURN_SPEED):
        """Spin in place, left."""
        self.drive_direct(speed, -speed)

    def turn_right(self, speed: int = TURN_SPEED):
        """Spin in place, right."""
        self.drive_direct(-speed, speed)

    def stop(self):
        self.drive_direct(0, 0)

    # --- docking ---
    def dock(self):
        """Attempt to find and dock with the charging base, same behaviour
        as automatic low-battery docking."""
        self._send(OI_SEEK_DOCK)

    # --- sensors ---
    def read_sensor(self, packet_id: int, num_bytes: int) -> bytes:
        """Request a sensor packet and read the response.
        See the OI spec for packet IDs and byte layouts."""
        if self.dry_run:
            print(f"[DRY-RUN] would request sensor packet {packet_id}, "
                  f"expecting {num_bytes} bytes")
            return b"\x00" * num_bytes
        self._send(OI_SENSORS, packet_id)
        time.sleep(0.05)
        return self.ser.read(num_bytes)

    def battery_percent(self) -> float:
        charge = struct.unpack(">H", self.read_sensor(SENSOR_BATTERY_CHARGE, 2))[0]
        capacity = struct.unpack(">H", self.read_sensor(SENSOR_BATTERY_CAPACITY, 2))[0]
        if capacity == 0:
            return -1.0
        return 100.0 * charge / capacity

    def close(self):
        if not self.dry_run and self.ser:
            self.stop()
            self.ser.close()


# --- Command name -> RoombaOI method mapping, for voice_command.py's execute_command() ---
def build_command_map(roomba: RoombaOI) -> dict:
    """Maps the string commands produced by voice_command.py's parse_command()
    to actual Roomba actions. Import and use this in voice_command.py's
    execute_command() hook."""
    return {
        "forward":  lambda: roomba.forward(),
        "backward": lambda: roomba.backward(),
        "left":     lambda: roomba.turn_left(),
        "right":    lambda: roomba.turn_right(),
        "stop":     lambda: roomba.stop(),
        "dock":     lambda: roomba.dock(),
        "come":     lambda: roomba.forward(),  # placeholder until a real "come to me" behaviour exists
    }


if __name__ == "__main__":
    # Self-test: runs in dry-run mode by default so this is safe to run
    # right now, before the USB cable arrives. Set dry_run=False once
    # /dev/ttyUSB0 exists and you're ready to test on real hardware.
    print("=== RoombaOI dry-run self-test ===")
    roomba = RoombaOI(dry_run=True)
    roomba.start()
    roomba.safe_mode()
    roomba.forward()
    time.sleep(0.5)
    roomba.turn_left()
    time.sleep(0.5)
    roomba.stop()
    print(f"Battery (dry-run, meaningless value): {roomba.battery_percent()}%")
    roomba.dock()
    roomba.close()
    print("=== Self-test complete ===")
