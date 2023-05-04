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
        print("Connecting")

    def image_to_gui(self, sock, NewFrame):
            # Sending the frames over to the C# GUI
            cv2.imwrite("C:\\Users\\david\\Desktop\\DTWpipe\\frameToSend.png", NewFrame)
            f = open("C:\\Users\\david\\Desktop\\DTWpipe\\frameToSend.png", 'rb')
            image_data = f.read()
            sock.sendall(image_data)
            f.close()

    def new_action(self):
            ActionFile = open("C:\\Users\\david\\Desktop\\DTWpipe\\action.txt", 'r')
            new_action = ActionFile.read(1)
            ActionFile.close()
            ActionFile = open("C:\\Users\\david\\Desktop\\DTWpipe\\action.txt", 'w'). close()
            return new_action

    def text_to_file(self, sign_detected):
            f = open("C:\\Users\\david\\Desktop\\DTWpipe\\translated_text.txt", "w")
            f.writelines("Reading Sign: " + sign_detected)
            f.close()

    def text_to_gui(self, sign_detected, sock):
                sock.send(bytes(sign_detected,'UTF-8'))
        #sock.send(sign_detected.encode())
