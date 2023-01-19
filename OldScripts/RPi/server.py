from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import argparse
import imutils
import cv2

imageHub = imagezmq.ImageHub()

lastActive = {}
lastActiveCheck = datetime.now()

ESTIMATED_NUM_PIS = 1
ACTIVE_CHECK_PERIOD = 10
ACTIVE_CHECK_SECONDS = ESTIMATED_NUM_PIS * ACTIVE_CHECK_PERIOD

while True:
    (rpiName,frame) = imageHub.recv_image()
    imageHub.send_reply(b"OK")

    if rpiName not in lastActive.keys():
        print("[INFO] receiving data from {}...".format(rpiName))

    lastActive[rpiName] = datetime.now()

#    frame = imutils.resize(frame,width=480)
    (h,w) = frame.shape[:2]

    #frameDict[rpiName] = frame

    cv2.imshow("Frame",frame)

    key = cv2.waitKey(1) & 0xFF

    if(datetime.now() - lastActiveCheck).seconds > ACTIVE_CHECK_SECONDS:
        for(rpiName,ts) in list(lastActive.items()):
            if (datetime.now() - ts).seconds > ACTIVE_CHECK_SECONDS:
                print("[INFO] lost connection to {}".format(rpiName))
                lastActive.pop(rpiName)
                #frameDict.pop(rpiName)

        lastActiveCheck = datetime.now()

    if key == ord("q"):
        break

cv2.destroyAllWindows()
