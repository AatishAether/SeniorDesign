import cv2
import os
import pickle
import pandas as pd

import re
import imagiz
import zlib
import struct
import time
import io
import sys

import socket
import time
from threading import Thread


class SocketManager(object):

    def _init_(self):
        print("Connecting...")

    def send_to_gui(self, sock, NewFrame):
            # Sending the frames over to the C# GUI
            cv2.imwrite("C:\\Users\\david\\Desktop\\DTWpipe\\frameToSend.png", NewFrame)
            myFrameToSend = open("C:\\Users\\david\\Desktop\\DTWpipe\\frameToSend.png", 'rb')
            image_data = myFrameToSend.read()
            sock.sendall(image_data)
            myFrameToSend.close()
