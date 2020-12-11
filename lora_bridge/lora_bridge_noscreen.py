#!/usr/bin/python3

# /usr/local/lib/python3.7/dist-packages/adafruit_rfm9x.py

import board
import busio
import digitalio

# Import the SSD1306 module.
#import adafruit_ssd1306
import adafruit_rfm9x

# import sys
import random
import time


# import socket

import socketio

sio = socketio.Client()


@sio.event
def connect():
    print('connection established')

@sio.event
def objectData(data):
    print('message received with ', data)
    #sio.emit('my response', {'response': 'my response'})

@sio.event
def disconnect():
    print('reconnecting to server')
    sio.connect('http://dicklabyvr.duckdns.org:3000')


sio.connect('http://dicklabyvr.duckdns.org:3000')
print('my sid is', sio.sid)




# Button A
btnA = digitalio.DigitalInOut(board.D5)
btnA.direction = digitalio.Direction.INPUT
btnA.pull = digitalio.Pull.UP

# Button B
btnB = digitalio.DigitalInOut(board.D6)
btnB.direction = digitalio.Direction.INPUT
btnB.pull = digitalio.Pull.UP

# Button C
btnC = digitalio.DigitalInOut(board.D12)
btnC.direction = digitalio.Direction.INPUT
btnC.pull = digitalio.Pull.UP

# Create the I2C interface.
#i2c = busio.I2C(board.SCL, board.SDA)

# 128x32 OLED Display
#reset_pin = digitalio.DigitalInOut(board.D4)
#display = adafruit_ssd1306.SSD1306_I2C(128, 32, i2c, reset=reset_pin)
# Clear the display.
#display.rotation = 2
#display.fill(0)
#display.show()
#width = display.width
#height = display.height


# set the time interval (seconds) for sending packets
transmit_interval = 0

# Define radio parameters.
RADIO_FREQ_MHZ = 915.0  # Frequency of the radio in Mhz. Must match your
# module! Can be a value like 915.0, 433.0, etc.
BAUDRATE = 10000000

# Define pins connected to the chip.
CS = digitalio.DigitalInOut(board.CE1)
RESET = digitalio.DigitalInOut(board.D25)

# Initialize SPI bus.
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)

# Initialze RFM radio

# Attempt to set up the rfm9x Module
try:
    rfm9x = adafruit_rfm9x.RFM9x(spi, CS, RESET, RADIO_FREQ_MHZ)
    #display.text("rfm9x: Detected", 0, 0, 1)
except RuntimeError:
    # Thrown on version mismatch
    #display.text("rfm9x: ERROR", 0, 0, 1)
    print("error")

#display.show()

# enable CRC checking
rfm9x.enable_crc = True

# set node addresses
rfm9x.node = 2
rfm9x.destination = 1
# initialize counter
counter = 0
# send a broadcast message from my_node with ID = counter


# initialize counter
counter = 0

# set timer
time_now = time.monotonic()

# Wait to receive packets.
print("Waiting for packets...")

while True:
    # Look for a new packet: only accept if addresses to my_node
    packet = rfm9x.receive(with_header=True)
    # If no packet was received during the timeout then None is returned.
    if packet is not None:
        # Received a packet!
        # Print out the raw bytes of the packet:
        #print("Received (raw header):", [hex(x) for x in packet[0:4]])
        #print("Received (raw payload): {0}".format(packet[4:]))
        #packet_text = str(packet, "ascii")
        #packet_text = str(packet, "UTF-8")
        #print("Packet text: ", packet_text)
        #print("Received RSSI: {0}".format(rfm9x.last_rssi))
        counter = counter + 1


        #######################
        # send to nodejs server
        #######################

        # parse payload
        # data = "id:{}.dotX:{}.dotY:{}".format(label, dotX, dotY)
        packet_text = str(packet, "UTF-8")
        label, dotX, dotY = packet_text.split('.')
        #print(label)

        label = label.strip()

        # send
        def sendData():
            data = {
                'id': label,
                'x': dotX,
                'y': dotY
            }

            sio.emit('objectData', data)

        sendData()


    #print(counter)

    # ticks counter
    if time.monotonic() - time_now > 1:
        time_now = time.monotonic()
        #print(counter)
        counter = 0
