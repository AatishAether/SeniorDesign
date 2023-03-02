import numpy as np

returns = []

def vecSave(vecList):
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

    angle_threshold = 5
    for key in dict1:
        vec1 = dict1[key]
        vec2 = dict2[key]
        dp = np.dot(vec1, vec2)
        mag1 = np.linalg.norm(vec1)
        mag2 = np.linalg.norm(vec2)
        cos_angle = dp / (mag1 * mag2)
        angle = np.arccos(cos_angle) * 180 / np.pi  # convert to degrees
        if angle > angle_threshold:
            returns.append("Check Image for " + str(key) + "!")
            return 1
            # print(f"Check Image for finger {key}! Angle: {angle:.2f} degrees")



        # dp = np.dot(dict1[key], dict2[key])
        #
        # mult = np.linalg.norm(dict1[key]) * np.linalg.norm(dict2[key])
        # angle = np.arccos(dp/mult)
        #
        #
        # if angle * 100 > 10:
        #     print("Check Image for " + str(key) + "!")
        #     return 1
        # else:
        #     return 2
        #     # angles.append(angle * 100)

        # angles.append(angle * 100)
        # return 2 if all(angle <= 10 for angle in angles) else 1

    # return angles
