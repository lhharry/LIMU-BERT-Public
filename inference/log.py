import serial, time

PORT, BAUD, SECONDS = 'COM12', 115200, 300
HEADER = ("time_ms,"
          "left_accel_x,left_accel_y,left_accel_z,"
          "left_gyro_x,left_gyro_y,left_gyro_z,"
          "right_accel_x,right_accel_y,right_accel_z,"
          
          "right_gyro_x,right_gyro_y,right_gyro_z\n")

with serial.Serial(PORT, BAUD, timeout=1) as ser, \
     open('imu_log.csv', 'w', newline='') as f:
    f.write(HEADER)
    ser.reset_input_buffer()
    t0 = time.time()
    while time.time() - t0 < SECONDS:
        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            continue
        if not (line[0].isdigit() or line[0] == '-'):
            continue
        f.write(line + '\n')

print("done -> imu_log.csv")