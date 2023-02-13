# --Here is the newest code, which does work to calculate the angles. 
import numpy as np

def vecSave(vecList):
    # saved_vectors = []
    # for key in d:
    #     k = key + num_frames_diff
    #     if k in d:
    #         u = d[key]
    #         v = d[k]
    #         saved_vectors.append((u, v))
    # print(saved_vectors)
    # print(vecList[len(vecList) - 2])
    # print(vecList[len(vecList) - 1])
    #
    # print(vecList[len(vecList) - 2][0])
    # print(vecList[len(vecList) - 2][4])

    dict1 = {
        0: vecList[len(vecList) - 2][0],
        1: vecList[len(vecList) - 2][1],
        2: vecList[len(vecList) - 2][2],
        3: vecList[len(vecList) - 2][3],
        4: vecList[len(vecList) - 2][4]
    }
    dict2 = {
        0: vecList[len(vecList) - 1][0],
        1: vecList[len(vecList) - 1][1],
        2: vecList[len(vecList) - 1][2],
        3: vecList[len(vecList) - 1][3],
        4: vecList[len(vecList) - 1][4]
    }

    angles = []

    for key in dict1:
        dp = np.dot(dict1[key], dict2[key])

        mult = np.linalg.norm(dict1[key]) * np.linalg.norm(dict2[key])
        angle = np.arccos(dp/mult)


        if angle * 100 > 5:
            print("Check Image for " + str(key) + "!")
            return 1
        else:
            return 2

        # angles.append(angle * 100)

    # return angles
    
# --This is the oldest code--
# import cv2
# import mediapipe as mp
# import numpy
# import numpy as np

# def vecSave(d):
#     sDict = d.copy()
#     angles = []
#     for key in d:
#         dp = abs(numpy.dot(d[key], sDict[key]))
#         mult = abs(numpy.dot(d[key])) * abs(numpy.dot(sDict[key]))
#         angle = numpy.arccos(dp/mult)
#         angles.append(angle)
#     return angles


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
