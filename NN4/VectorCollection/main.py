import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import pandas as pd

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FPS, 40)

file = open('ASL_Alph_Vectorized.csv', 'a')
frame_count = 1200
with file:
    header = ['Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Thumb_TIP', 'Index_MCP', 'Index_PIP', 'Index_DIP', 'Index_TIP', 'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_TIP', 'Ring_MCP',
                    'Ring_PIP', 'Ring_DIP', 'Ring_TIP', 'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_TIP', 'Letter', 'Class_Index', 'ImagePath']
    writer = csv.DictWriter(file, fieldnames=header, quoting=csv.QUOTE_NONE, escapechar=' ')
    if (os.stat('ASL_Alph_Vectorized.csv').st_size == 0):
        writer.writeheader()
    Class = np.array(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "nothing", "O", "P", "Q", "R", "S",
          "space", "T", "U", "V", "W", "X", "Y", "Z"])
    while vid.isOpened() & (frame_count < 2400):
        frame, image = vid.read()
        imcopy = image
        # cv2.imshow('vectorizer', image)
        mpdraw = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mppose = mp.solutions.hands

        hand = mppose.Hands(static_image_mode=True, model_complexity=0, min_detection_confidence=.2)

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hand.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks is not None:
            print("true")
            for hand_landmarks in results.multi_hand_landmarks:
                mpdraw.draw_landmarks(
                    image,
                    hand_landmarks,
                    mppose.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            cv2.imshow('Vectorizer', image)

            entry2 = 'F'
            Ind = np.where(Class == entry2)[0][0]
            print(Ind)
            frame_count += 1
            if(frame_count % 10 == 7):
                path = ".\\ASL_Data\ASL_Test\\" + entry2 + "\\" + entry2 + str(frame_count) + ".jpg"
                cv2.imwrite(".\\ASL_Data\ASL_Test" + "\\" + entry2 + "\\" + entry2 + str(frame_count) + ".jpg", imcopy)
            elif(frame_count % 10 == 9):
                path = ".\\ASL_Data\ASL_Validate\\" + entry2 + "\\" + entry2 + str(frame_count) + ".jpg"
                cv2.imwrite(".\\ASL_Data\ASL_Validate" + "\\" + entry2 + "\\" + entry2 + str(frame_count) + ".jpg", imcopy)
            else:
                path = ".\\ASL_Data\ASL_Train\\" + entry2 + "\\" + entry2 + str(frame_count) + ".jpg"
                cv2.imwrite(".\\ASL_Data\ASL_Train" + "\\" + entry2 + "\\" + entry2 + str(frame_count) + ".jpg", imcopy)
            dict = {
                'Thumb_CMC': [results.multi_hand_landmarks[0].landmark[1].x,results.multi_hand_landmarks[0].landmark[1].y,results.multi_hand_landmarks[0].landmark[1].z],
                 'Thumb_MCP': [results.multi_hand_landmarks[0].landmark[2].x,results.multi_hand_landmarks[0].landmark[2].y,results.multi_hand_landmarks[0].landmark[2].z],
                 'Thumb_IP': [results.multi_hand_landmarks[0].landmark[3].x,results.multi_hand_landmarks[0].landmark[3].y,results.multi_hand_landmarks[0].landmark[3].z],
                 'Thumb_TIP': [results.multi_hand_landmarks[0].landmark[4].x,results.multi_hand_landmarks[0].landmark[4].y,results.multi_hand_landmarks[0].landmark[4].z],
                 'Index_MCP': [results.multi_hand_landmarks[0].landmark[5].x,results.multi_hand_landmarks[0].landmark[5].y,results.multi_hand_landmarks[0].landmark[5].z],
                 'Index_PIP': [results.multi_hand_landmarks[0].landmark[6].x,results.multi_hand_landmarks[0].landmark[6].y,results.multi_hand_landmarks[0].landmark[6].z],
                 'Index_DIP': [results.multi_hand_landmarks[0].landmark[7].x,results.multi_hand_landmarks[0].landmark[7].y,results.multi_hand_landmarks[0].landmark[7].z],
                 'Index_TIP': [results.multi_hand_landmarks[0].landmark[8].x,results.multi_hand_landmarks[0].landmark[8].y,results.multi_hand_landmarks[0].landmark[8].z],
                 'Middle_MCP': [results.multi_hand_landmarks[0].landmark[9].x,results.multi_hand_landmarks[0].landmark[9].y,results.multi_hand_landmarks[0].landmark[9].z],
                 'Middle_PIP': [results.multi_hand_landmarks[0].landmark[10].x,results.multi_hand_landmarks[0].landmark[10].y,results.multi_hand_landmarks[0].landmark[10].z],
                 'Middle_DIP': [results.multi_hand_landmarks[0].landmark[11].x,results.multi_hand_landmarks[0].landmark[11].y,results.multi_hand_landmarks[0].landmark[11].z],
                 'Middle_TIP': [results.multi_hand_landmarks[0].landmark[12].x,results.multi_hand_landmarks[0].landmark[12].y,results.multi_hand_landmarks[0].landmark[12].z],
                 'Ring_MCP': [results.multi_hand_landmarks[0].landmark[13].x,results.multi_hand_landmarks[0].landmark[13].y,results.multi_hand_landmarks[0].landmark[13].z],
                 'Ring_PIP': [results.multi_hand_landmarks[0].landmark[14].x,results.multi_hand_landmarks[0].landmark[14].y,results.multi_hand_landmarks[0].landmark[14].z],
                 'Ring_DIP': [results.multi_hand_landmarks[0].landmark[15].x,results.multi_hand_landmarks[0].landmark[15].y,results.multi_hand_landmarks[0].landmark[15].z],
                 'Ring_TIP': [results.multi_hand_landmarks[0].landmark[16].x,results.multi_hand_landmarks[0].landmark[16].y,results.multi_hand_landmarks[0].landmark[16].z],
                 'Pinky_MCP': [results.multi_hand_landmarks[0].landmark[17].x,results.multi_hand_landmarks[0].landmark[17].y,results.multi_hand_landmarks[0].landmark[17].z],
                 'Pinky_PIP': [results.multi_hand_landmarks[0].landmark[18].x,results.multi_hand_landmarks[0].landmark[18].y,results.multi_hand_landmarks[0].landmark[18].z],
                 'Pinky_DIP': [results.multi_hand_landmarks[0].landmark[19].x,results.multi_hand_landmarks[0].landmark[19].y,results.multi_hand_landmarks[0].landmark[19].z],
                 'Pinky_TIP': [results.multi_hand_landmarks[0].landmark[20].x,results.multi_hand_landmarks[0].landmark[20].y,results.multi_hand_landmarks[0].landmark[20].z],
                 'Letter': entry2,
                 'Class_Index': Ind,
                 'ImagePath': path
            }
            DataF = pd.Series(dict).to_frame().T
            print(DataF)

            pd.DataFrame.to_csv(DataF, file, header=False, encoding='utf-8', index=False)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# with open('ASL_Alph_Vectorized.csv', 'r') as file2:
#     reader = csv.reader(file2)
#     header = next(reader)  # Skip the header row
#     row_after_header = next(reader, None)  # Read the row after the header
#     if row_after_header:
#         # Process the row or access the cell values as needed
#         first_cell_value = row_after_header[0]
# print(first_cell_value)
vid.release()
cv2.destroyAllWindows()