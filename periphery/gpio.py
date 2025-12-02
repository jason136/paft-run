from periphery import GPIO
import time

#H-bridge pins
in1 = GPIO(571, "out")
in2 = GPIO(572, "out")
in3 = GPIO(579, "out")
in4 = GPIO(581, "out")

def set_motor(in1, in2, a, b):
    in1.write(a)
    in2.write(b)

try:
    while True:
        # Motor A clockwise
        set_motor(in1, in2, True, False)
        #time.sleep(5)
        print("motor a cw")

        # Motor A brake
        #set_motor(IN1, IN2, True, True)
        #time.sleep(0.5)

        # Motor B clockwise
        set_motor(in3, in4, True, False)
        time.sleep(5)

        # Motor B brake
        #set_motor(IN3, IN4, True, True)
        #time.sleep(0.5)

        # Motor A counter-clockwise
        set_motor(in1, in2, False, True)
        #time.sleep(5)

        # Motor A brake
        #set_motor(IN1, IN2, True, True)
        #time.sleep(0.5)

        # Motor B counter-clockwise
        set_motor(in3, in4, False, True)
        time.sleep(5)

        # Motor B brake
        #set_motor(IN3, IN4, True, True)
        #time.sleep(0.5)

except KeyboardInterrupt:
    for p in [IN1, IN2, IN3, IN4]:
        p.write(False)

finally:
    for p in [IN1, IN2, IN3, IN4]:
        p.close()
