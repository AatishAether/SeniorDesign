import cv2
import mediapipe as mp
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

from utils.dataset_utils import load_dataset, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager
from my_socket import SocketManager

from imutils.video import VideoStream
import socket
import time
from threading import Thread


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_address = ('172.16.206.245', 1234)
# sock.connect(server_address)
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_address = ('192.168.56.1', 1234)
# sock.connect(server_address)
# sock.setblocking(False)

image_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ("172.16.206.245", 1234)
image_sock.connect(server_address)
image_sock.setblocking(0)

# text_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_address = ("172.16.206.245", 5678)
# text_sock.connect(server_address)
# text_sock.setblocking(False)

def main():
    global image_sock
    # Create dataset of the videos where landmarks have not been extracted yet
    print("Reading Dataset...")
    dataset = load_dataset()

    # Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
    ##I think this is cause a huge bog down. If we can save the reference signs dataFrame
    ##This can save time
    print("Loading Signs...")
    if not (os.path.exists("./referenceSigns.pickle")):
        reference_signs = load_reference_signs(dataset)

        reference_signs.to_pickle('./referenceSigns.pickle')

    else:
        reference_signs = pd.read_pickle('./referenceSigns.pickle')



    print("Creating Sign Recorder object")
    # Object that stores mediapipe results and computes sign similarities
    sign_recorder = SignRecorder(reference_signs)

    # Object that sends images and text to the gui
    my_socket = SocketManager()

    # Object that draws keypoints & displays results
    webcam_manager = WebcamManager()

    # Turn on the webcam
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # Set up the Mediapipe environment
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Process results
            sign_detected, is_recording = sign_recorder.process_results(results)

            # Update the frame (draw landmarks & display result)
            newFrame = webcam_manager.update(frame, results, sign_detected, is_recording)

            my_socket.image_to_gui(image_sock, newFrame)
            my_socket.text_to_file(sign_detected)

            Action = my_socket.new_action()


            if Action == ("R"):  # Record pressing r
                image_sock.setblocking(1)
                sign_recorder.record()
                ret, frame = cap.read()
            elif Action == ("Q"):  # Break pressing q
                break
            elif Action == ('P'): ##Print to file
                features = sign_recorder.recorded_sign.lh_embedding
                #features = str(features).replace('[','')
                #features = features.replace(']','')
                #features = features.replace("'",'')
                #features = list(features.split(","))
                #features = list(map(float,features))
                with open("features6.pickle", "wb") as f:
                    pickle.dump(features,f)

            continue

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
