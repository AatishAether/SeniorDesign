import cv2
import mediapipe as mp
import csv
from PIL import Image
from numpy import asarray
import tkinter as tk
from PIL import ImageTk

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
from vSave import vecSave

cap = cv2.VideoCapture(0)
count = 0
vec_list = []

root = tk.Tk()
root.title("Video Stream")

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()

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

        img = Image.fromarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        photo = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)

        img2 = Image.open('img.png')
        darr = asarray(img2)

        if results.multi_hand_landmarks is not None:
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
                    [results.multi_hand_landmarks[0].landmark[4].x, results.multi_hand_landmarks[0].landmark[4].y,
                     results.multi_hand_landmarks[0].landmark[4].z],
                    [results.multi_hand_landmarks[0].landmark[8].x, results.multi_hand_landmarks[0].landmark[8].y,
                     results.multi_hand_landmarks[0].landmark[8].z],
                    [results.multi_hand_landmarks[0].landmark[12].x, results.multi_hand_landmarks[0].landmark[12].y,
                     results.multi_hand_landmarks[0].landmark[12].z],
                    [results.multi_hand_landmarks[0].landmark[16].x, results.multi_hand_landmarks[0].landmark[16].y,
                     results.multi_hand_landmarks[0].landmark[16].z],
                    [results.multi_hand_landmarks[0].landmark[20].x, results.multi_hand_landmarks[0].landmark[20].y,
                     results.multi_hand_landmarks[0].landmark[20].z]

                ]
                filename = 'vectors.csv'

                if (count % 7 == 0):
                    print(count)
                    vec_list.append(data)
                    with open('vectors.csv', 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['frame', 'vector'])
                        for i, vec in enumerate(vec_list):
                            writer.writerow([i, vec])

                    print()
                    if (len(vec_list) > 1):
                        if vecSave(vec_list) == 1:
                            # cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            cv2.imwrite('frame' + str(count) + '.jpg', image)
                            cv2.imwrite('blank' + str(count) + '.jpg', darr)

                # vecSave(vecDict, count)
                count += 1

                img_tk = ImageTk.PhotoImage(Image.fromarray(image))
                canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
                root.update()
        else:
            img_tk = ImageTk.PhotoImage(Image.fromarray(image))
            canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
            root.update()
    root.mainloop()

root.destroy()
cap.release()
