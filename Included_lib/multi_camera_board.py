import os


# from RPi import GPIO as GPIO
class MultiCameraBoard:
    def __init__(self):
        self.selection_gpio_pin = 7
        self.enable1_gpio_pin = 11
        self.enable2_gpio_pin = 12

    def off_camera_socket(self):
        # GPIO.output(self.selection_gpio_pin, GPIO.HIGH)
        # GPIO.output(self.enable1_gpio_pin, GPIO.HIGH)
        # GPIO.output(self.enable2_gpio_pin, GPIO.HIGH)
        pass

    def set_camera_socket_a(self):
        i2c = "i2cset -y 1 0x70 0x00 0x04"
        os.system(i2c)
        # GPIO.output(self.selection_gpio_pin, GPIO.LOW)
        # GPIO.output(self.enable1_gpio_pin, GPIO.LOW)
        # GPIO.output(self.enable2_gpio_pin, GPIO.HIGH)

    def set_camera_socket_b(self):
        i2c = "i2cset -y 1 0x70 0x00 0x05"
        os.system(i2c)
        # GPIO.output(self.selection_gpio_pin, GPIO.HIGH)
        # GPIO.output(self.enable1_gpio_pin, GPIO.LOW)
        # GPIO.output(self.enable2_gpio_pin, GPIO.HIGH)

    def set_camera_socket_c(self):
        i2c = "i2cset -y 1 0x70 0x00 0x06"
        os.system(i2c)
        # GPIO.output(self.selection_gpio_pin, GPIO.LOW)
        # GPIO.output(self.enable1_gpio_pin, GPIO.HIGH)
        # GPIO.output(self.enable2_gpio_pin, GPIO.LOW)

    def set_camera_socket_d(self):
        i2c = "i2cset -y 1 0x70 0x00 0x07"
        os.system(i2c)
        # GPIO.output(self.selection_gpio_pin, GPIO.HIGH)
        # GPIO.output(self.enable1_gpio_pin, GPIO.HIGH)
        # GPIO.output(self.enable2_gpio_pin, GPIO.LOW)
