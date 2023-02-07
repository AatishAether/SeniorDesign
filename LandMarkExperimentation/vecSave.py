import cv2
import mediapipe as mp
import numpy
import numpy as np

def vecSave(d):
    sDict = d.copy()
    angles = []
    for key in d:
        dp = abs(numpy.dot(d[key], sDict[key]))
        mult = abs(numpy.dot(d[key])) * abs(numpy.dot(sDict[key]))
        angle = numpy.arccos(dp/mult)
        angles.append(angle)
    return angles

