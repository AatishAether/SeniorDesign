import cv2
import mediapipe as mp
import os
import pickle
import pandas as pd

import re

from utils.dataset_utils import load_dataset, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager

from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time
from threading import Thread

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
#mp_hands = mp.solutions.hands

#imageHub = imagezmq.ImageHub()

def main():
    # print("Trying to start sender....")
    # # sender = imagezmq.ImageSender(connect_to="tcp://127.0.0.1:6008")
    # # rpiName = socket.gethostname()
    # # time.sleep(2.0)
    #
    #
    # # Create dataset of the videos where landmarks have not been extracted yet
    # print("Reading Dataset...")

    sock = socket.socket()
    sock.connect(("localhost", 5555))

        #filename1 = "./image.bin"   # save data as bin file
        #bin_file = image.tofile(filename1)

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

    # Object that draws keypoints & displays results
    webcam_manager = WebcamManager()

    # Turn on the webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Set up the Mediapipe environment
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Process results
            sign_detected, is_recording = sign_recorder.process_results(results)

            # Update the frame (draw landmarks & display result)
            newFrame = webcam_manager.update(frame, results, sign_detected, is_recording)

            bin_file = frame.tofile("C:\\Users\\david\\Desktop\\DTWpipe\\data.bin")

            with open("C:\\Users\\david\\Desktop\\DTWpipe\\data.bin", "rb") as fd:
                buf = fd.read(1024)
                sock.send(buf)
                buf = fd.read(1024)

            #(rpiName,frame) = imageHub.recv_image()
            # sender.send_image(rpiName,frame)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord("r"):  # Record pressing r
                sign_recorder.record()
            elif pressedKey == ord("q"):  # Break pressing q
                break
            elif pressedKey == ord('p'): ##Print to file
                features = sign_recorder.recorded_sign.lh_embedding
                #features = str(features).replace('[','')
                #features = features.replace(']','')
                #features = features.replace("'",'')
                #features = list(features.split(","))
                #features = list(map(float,features))
                with open("features6.pickle", "wb") as f:
                    pickle.dump(features,f)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
