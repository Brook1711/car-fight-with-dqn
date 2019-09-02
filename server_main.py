#import necessary package
import socket
import time
import sys
#import RPi.GPIO as GPIO
import string 
#define host ip: Rpi's IP
#input_ip = input("host ip:")
#HOST_IP = str(input_ip)

HOST_IP = "192.168.43.203"
HOST_PORT = 8888
print("Starting socket: TCP...")
#1.create socket object:socket=socket.socket(family,type)
socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("TCP server listen @ %s:%d!" %(HOST_IP, HOST_PORT) )
host_addr = (HOST_IP, HOST_PORT)
#2.bind socket to addr:socket.bind(address)
socket_tcp.bind(host_addr)
#3.listen connection request:socket.listen(backlog)
socket_tcp.listen(1)
#4.waite for client:connection,address=socket.accept()
socket_con, (client_ip, client_port) = socket_tcp.accept()
print("Connection accepted from %s." %client_ip)
str='Welcome to RPi TCP server!'
str=str.encode()
#socket_con.send(str)
#5.handle
#GPIO.setmode(GPIO.BOARD)
#GPIO.setup(11,GPIO.OUT)
print("Receiving package...")
print("setting pwm...")
from  donkeycar.parts.actuator import PCA9685
import config as cfg
throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
while True:
    data=socket_con.recv(512)
    data= data.decode()
    if data[0] == 'S':
        steering_pulse_str = data[1:4]
        steering_pulse = int(steering_pulse_str)
        steering_controller.set_pulse(steering_pulse)
    if data[0] == 'T':
        throttle_pulse_str = data[1:4]
        throttle_pulse = int(throttle_pulse_str)
        throttle_controller.set_pulse(throttle_pulse)
    print(data)
