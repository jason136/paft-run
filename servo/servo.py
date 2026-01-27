import smbus
import time

# ----- I2C setup -----
I2C_BUS = 1
PCA_ADDR = 0x40

bus = smbus.SMBus(I2C_BUS)

# ----- PCA9685 registers -----
MODE1 = 0x00
MODE2 = 0x01
LED0_ON_L = 0x06
PRE_SCALE = 0xFE

# ----- Helper functions -----
def set_pwm_freq(freq_hz):
    prescaleval = 25000000.0
    prescaleval /= 4096.0
    prescaleval /= float(freq_hz)
    prescaleval -= 1.0
    prescale = int(prescaleval + 0.5)
    
    oldmode = bus.read_byte_data(PCA_ADDR, MODE1)
    newmode = (oldmode & 0x7F) | 0x10
    bus.write_byte_data(PCA_ADDR, MODE1, newmode)
    bus.write_byte_data(PCA_ADDR, PRE_SCALE, prescale)
    bus.write_byte_data(PCA_ADDR, MODE1, oldmode)
    time.sleep(0.005)
    bus.write_byte_data(PCA_ADDR, MODE1, oldmode | 0x80)

def set_pwm(channel, on, off):
    bus.write_byte_data(PCA_ADDR, LED0_ON_L + 4*channel, on & 0xFF)
    bus.write_byte_data(PCA_ADDR, LED0_ON_L + 4*channel + 1, on >> 8)
    bus.write_byte_data(PCA_ADDR, LED0_ON_L + 4*channel + 2, off & 0xFF)
    bus.write_byte_data(PCA_ADDR, LED0_ON_L + 4*channel + 3, off >> 8)

def set_servo_speed(channel, speed):
    """
    Set speed for 360° servo.
    speed = -1.0 (full reverse) to 0 (stop) to 1.0 (full forward)
    """
    # Map speed to pulse width (in 12-bit units)
    # typical stop pulse = 307 (1.5 ms)
    # full reverse = ~205 (1 ms)
    # full forward = ~410 (2 ms)
    stop = 312
    min_pulse = 205
    max_pulse = 410
    if speed > 1.0: speed = 1.0
    if speed < -1.0: speed = -1.0
    
    if speed == 0:
        pulse = stop
    elif speed > 0:
        pulse = int(stop + (max_pulse - stop) * speed)
    else:
        pulse = int(stop + (min_pulse - stop) * speed)
    
    set_pwm(channel, 0, pulse)

# ----- Initialization -----
bus.write_byte_data(PCA_ADDR, MODE2, 0x04)
bus.write_byte_data(PCA_ADDR, MODE1, 0x00)
set_pwm_freq(50)

SERVO_CHANNEL = 0

# ----- Test: move servo -----
try:
    while True:
        print("Full forward")
        set_servo_speed(SERVO_CHANNEL, 1.0)
        time.sleep(2)
        
        print("Stop")
        set_servo_speed(SERVO_CHANNEL, 0.0)
        time.sleep(2)
        
        print("Full reverse")
        set_servo_speed(SERVO_CHANNEL, -1.0)
        time.sleep(2)
        
        print("Stop")
        set_servo_speed(SERVO_CHANNEL, 0.0)
        time.sleep(2)
except KeyboardInterrupt:
    set_servo_speed(SERVO_CHANNEL, 0.0)
    print("Exiting, servo stopped")
