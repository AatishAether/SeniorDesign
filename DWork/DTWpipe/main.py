import cv2
import mediapipe as mp
import os
import pickle
import pandas as pd
from pathlib import Path

import re
import imagiz
import zlib
import struct
import time
import io

from utils.dataset_utils import load_dataset, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager

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

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (str(socket.gethostbyname(socket.gethostname())), 1234)
sock.connect(server_address)
sock.setblocking(False)

def main():
    # change current working directory
    os.chdir(Path(__file__).parent)

    # Create dataset of the videos where landmarks have not been extracted yet
    print("Reading Dataset...")
    dataset = load_dataset()
    print("DSSIZE", len(dataset))

    # Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
    ##I think this is cause a huge bog down. If we can save the reference signs dataFrame
    ##This can save time
    print("Loading Signs...")
    if not (os.path.exists("./referenceSigns.pickle")):
        print('pickling signs')
        reference_signs = load_reference_signs(dataset)
        reference_signs.to_pickle('./referenceSigns.pickle')

    else:
        print('unpickling signs')
        reference_signs = pd.read_pickle('./referenceSigns.pickle')

    print("Creating Sign Recorder object")
    # Object that stores mediapipe results and computes sign similarities
    print('REF SIGNS:', reference_signs)
    sign_recorder = SignRecorder(reference_signs)
    # exit(0)

    # Object that draws keypoints & displays results
    webcam_manager = WebcamManager()

    # Turn on the webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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

            cv2.imwrite("C:\\Users\\EEG\\Documents\\GitHub\\SeniorDesign\\DWork\\DTWpipe\\frameToSend.png", newFrame)
            f = open("C:\\Users\\EEG\\Documents\\GitHub\\SeniorDesign\\DWork\\DTWpipe\\frameToSend.png", 'rb')
            image_data = f.read()
            sock.sendall(image_data)
            f.close()

            print("here")

            ActionFile = open("C:\\Users\\EEG\\Documents\\GitHub\\SeniorDesign\\DWork\\DTWpipe\\action.txt", 'r')
            Action = ActionFile.read(1)
            ActionFile.close()
            ActionFile = open("C:\\Users\\EEG\\Documents\\GitHub\\SeniorDesign\\DWork\\DTWpipe\\action.txt", 'w'). close()

            # f = open("C:\\Users\\EEG\\Documents\\GitHub\\SeniorDesign\\DWork\\DTWpipe\\translated_text.txt", "w")
            # f.write("Text: " + repr(sign_detected) + str(type(sign_detected)))
            # f.close()

            if Action == ("R"):  # Record pressing r
                print("ACTION R")
                sign_recorder.record()
            elif Action == ("Q"):  # Break pressing q
                break
            elif Action == ('P'): ##Print to file
                print("ACTION P")
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
