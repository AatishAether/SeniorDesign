from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time

ap = argparse.ArgumentParser()
ap.add_argument("-s","--serverip", required=True,
            help="ip address of server the client will connect")
arg = vars(ap.parse_args())

sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(arg["serverip"]))

#get host name
#initialize videostream
#warm up camera sensor
rpiName = socket.gethostname()
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    sender.send_image(rpiName,frame)

