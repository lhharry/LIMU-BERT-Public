import csv
import time

import serial


PORT = "COM12"
BAUD = 115200
SECONDS = 300
#add current time and date to output file name
timestamp = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"inference/rampup_imu_log_{timestamp}.csv"

OUTPUT_COLUMNS = [
    "time_ms",
    "left_accel_x", "left_accel_y", "left_accel_z",
    "left_gyro_x", "left_gyro_y", "left_gyro_z",
    "right_accel_x", "right_accel_y", "right_accel_z",
    "right_gyro_x", "right_gyro_y", "right_gyro_z",
]

# Map incoming MCU keys to CSV column names.
KEY_MAP = {
    "L_ax": "left_accel_x", "L_ay": "left_accel_y", "L_az": "left_accel_z",
    "L_gx": "left_gyro_x", "L_gy": "left_gyro_y", "L_gz": "left_gyro_z",
    "R_ax": "right_accel_x", "R_ay": "right_accel_y", "R_az": "right_accel_z",
    "R_gx": "right_gyro_x", "R_gy": "right_gyro_y", "R_gz": "right_gyro_z",
}


def parse_imu_line(line: str, elapsed_ms: int):
    # Expected format example:
    # L_ax:1.610,L_ay:3.410,...,R_gz:0.125
    values = {"time_ms": elapsed_ms}
    for part in line.split(","):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in KEY_MAP:
            try:
                values[KEY_MAP[key]] = float(value)
            except ValueError:
                return None

    # Only accept full packets to avoid partial/corrupted writes.
    for col in OUTPUT_COLUMNS[1:]:
        if col not in values:
            return None
    return values


kept = 0
seen = 0

with serial.Serial(PORT, BAUD, timeout=1) as ser, open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
    writer.writeheader()

    ser.reset_input_buffer()
    t0 = time.time()
    while time.time() - t0 < SECONDS:
        raw = ser.readline().decode(errors="ignore").strip()
        if not raw:
            continue

        seen += 1
        row = parse_imu_line(raw, int((time.time() - t0) * 1000))
        if row is None:
            continue

        writer.writerow(row)
        kept += 1

print(f"done -> {OUTPUT_FILE}, kept={kept}, seen={seen}")