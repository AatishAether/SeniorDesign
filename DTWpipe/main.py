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
from my_socket import SocketManager

import socket
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
#mp_hands = mp.solutions.hands

image_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (str(socket.gethostbyname(socket.gethostname())), 1234)
image_sock.connect(server_address)
image_sock.setblocking(0)

def main():
    # Create dataset of the videos where landmarks have not been extracted yet
    print("Reading Dataset...")
    n,dataset = load_dataset()

    # Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
    ##I think this is cause a huge bog down. If we can save the reference signs dataFrame
    ##This can save time
    print("Loading Signs...")
    if not (os.path.exists("./referenceSigns.pickle")) or (n > 0):
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

            my_socket.image_to_gui(image_sock, newFrame)
            # webcam_manager.update(frame, results, sign_detected, is_recording)
            if(sign_detected != ""):
                # print(sign_detected)
                my_socket.text_to_file(sign_detected)

            Action = my_socket.new_action()

            if Action == ("R"):  # Record pressing r
                # image_sock.setblocking(1)
                # sign_recorder.record()
                # # ret, frame = cap.read()
                # print(sign_detected)
                # with open('action.txt', 'a') as f:
                #     f.write('Hello!')
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # host = socket.gethostname()

                # set port number
                port = 12345

                # connect to the server
                s.connect((socket.gethostbyname(socket.gethostname()), 12345))
                s.setblocking(0)
                message = "Hello from Python!"
                s.send(message.encode())

                # wait for a response from the server
                response = s.recv(1024)
                print(response.decode())
                s.close()
            elif Action == ("Q"):  # Break pressing q
                break
            elif Action == ('P'):  ##Print to file
                features = sign_recorder.recorded_sign.lh_embedding
                # features = str(features).replace('[','')
                # features = features.replace(']','')
                # features = features.replace("'",'')
                # features = list(features.split(","))
                # features = list(map(float,features))
                with open("features6.pickle", "wb") as f:
                    pickle.dump(features, f)

            continue

        cap.release()
        cv2.destroyAllWindows()

        #     pressedKey = cv2.waitKey(1) & 0xFF
        #     if pressedKey == ord("r"):  # Record pressing r
        #         sign_recorder.record()
        #     elif pressedKey == ord("q"):  # Break pressing q
        #         break
        #     elif pressedKey == ord('p'): ##Print to file
        #         features = sign_recorder.recorded_sign.lh_embedding
        #         #features = str(features).replace('[','')
        #         #features = features.replace(']','')
        #         #features = features.replace("'",'')
        #         #features = list(features.split(","))
        #         #features = list(map(float,features))
        #         with open("openHand.pickle", "wb") as f:
        #             pickle.dump(features,f)
        #
        # cap.release()
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
