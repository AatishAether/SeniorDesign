import cv2
import mediapipe as mp
import csv
from PIL import Image
from numpy import asarray

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
from vecSave import vecSave

cap = cv2.VideoCapture(0)


def mediaPipe(img, frame):
    count = 0
    vec_list = []

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:


      # while cap.isOpened():
      while True:
        # success, image = cap.read()
        success = img
        image = frame

        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img = Image.open('img.png')
        darr = asarray(img)

        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            mp_drawing.draw_landmarks(
                darr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # print(results.multi_hand_landmarks[0].landmark[16].x)

            # 4 is the thumb tip
            # 8 is the index tip
            # 12 is the middle finger tip
            # 16 is the ring finger tip
            # 20 is the pinky tip

            data = [
                [results.multi_hand_landmarks[0].landmark[4].x, results.multi_hand_landmarks[0].landmark[4].y, results.multi_hand_landmarks[0].landmark[4].z],
                [results.multi_hand_landmarks[0].landmark[8].x, results.multi_hand_landmarks[0].landmark[8].y, results.multi_hand_landmarks[0].landmark[8].z],
                [results.multi_hand_landmarks[0].landmark[12].x, results.multi_hand_landmarks[0].landmark[12].y, results.multi_hand_landmarks[0].landmark[12].z],
                [results.multi_hand_landmarks[0].landmark[16].x, results.multi_hand_landmarks[0].landmark[16].y, results.multi_hand_landmarks[0].landmark[16].z],
                [results.multi_hand_landmarks[0].landmark[20].x, results.multi_hand_landmarks[0].landmark[20].y, results.multi_hand_landmarks[0].landmark[20].z]
            ]

            # print(data[0])

            vecDict = {
                0: data[0],
                1: data[1],
                2: data[2],
                3: data[3],
                4: data[4]
            }

            filename = 'vectors.csv'

            if(count % 7 == 0):
                print(count)
                vec_list.append(data)
                with open('vectors.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['frame', 'vector'])
                    for i, vec in enumerate(vec_list):
                        writer.writerow([i, vec])

                print()
                if(len(vec_list) > 1):
                    if vecSave(vec_list) == 1:
                        cv2.imwrite('frame' + str(count) + '.jpg', image)
                        cv2.imwrite('blank' + str(count) + '.jpg', darr)

            count += 1

        # Flip the image horizontally for a selfie-view display.
        # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        # cv2.imshow('Test', cv2.flip(darr, 1))
        # if cv2.waitKey(5) & 0xFF == 27:
        #   break

    cap.release()