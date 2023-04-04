import numpy as np

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

    angles = []

    for key in dict1:
        dp = np.dot(dict1[key], dict2[key])

        mult = np.linalg.norm(dict1[key]) * np.linalg.norm(dict2[key])
        angle = np.arccos(dp/mult)


        if angle * 100 > 10:
            print("Check Image for " + str(key) + "!")
            return 1
        else:
            return 2

        # angles.append(angle * 100)

    # return angles



