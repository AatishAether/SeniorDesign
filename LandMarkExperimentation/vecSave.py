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


# The following is new code that really doesn't do much of what it should, but I am saving it as a starting point. 
# def vecSave(d, num_frames_diff):
#     saved_vectors = []
#     for key in d:
#         k = key + num_frames_diff
#         if k in d:
#             u = d[key]
#             v = d[k]
#             saved_vectors.append((u, v))
#     print(saved_vectors)
#     return saved_vectors
