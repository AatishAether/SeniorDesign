import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

x = 0;
y = 0;
z = 0;

cap = cv2.VideoCapture(0)
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
    if results.multi_hand_landmarks is not None:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        for landmark in hand_landmarks.landmark:
            x = landmark.x
            y = landmark.y
            z = landmark.z

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
        #print(data)
        if ((results.multi_hand_landmarks[0].landmark[4].x > results.multi_hand_landmarks[0].landmark[8].x) and (
                results.multi_hand_landmarks[0].landmark[8].x > results.multi_hand_landmarks[0].landmark[12].x) and (
                results.multi_hand_landmarks[0].landmark[12].x > results.multi_hand_landmarks[0].landmark[16].x) and (
                results.multi_hand_landmarks[0].landmark[16].x > results.multi_hand_landmarks[0].landmark[20].x)):
            if ((results.multi_hand_landmarks[0].landmark[4].y < results.multi_hand_landmarks[0].landmark[8].y) and (
                    results.multi_hand_landmarks[0].landmark[8].y < results.multi_hand_landmarks[0].landmark[12].y) and (
                    results.multi_hand_landmarks[0].landmark[12].y < results.multi_hand_landmarks[0].landmark[16].y) and (
                    results.multi_hand_landmarks[0].landmark[16].y > results.multi_hand_landmarks[0].landmark[20].y)):
                if ((results.multi_hand_landmarks[0].landmark[4].z < results.multi_hand_landmarks[0].landmark[8].z) and (
                        results.multi_hand_landmarks[0].landmark[8].z > results.multi_hand_landmarks[0].landmark[12].z) and (
                        results.multi_hand_landmarks[0].landmark[12].z > results.multi_hand_landmarks[0].landmark[16].z) and (
                        results.multi_hand_landmarks[0].landmark[16].z > results.multi_hand_landmarks[0].landmark[20].z)):
                    print("A")
                else:
                    print("Y")
            elif ((results.multi_hand_landmarks[0].landmark[4].y > results.multi_hand_landmarks[0].landmark[8].y) and (
                    results.multi_hand_landmarks[0].landmark[8].y > results.multi_hand_landmarks[0].landmark[12].y) and (
                    results.multi_hand_landmarks[0].landmark[12].y < results.multi_hand_landmarks[0].landmark[16].y) and (
                    results.multi_hand_landmarks[0].landmark[16].y < results.multi_hand_landmarks[0].landmark[20].y)):
                print("F")
            elif ((results.multi_hand_landmarks[0].landmark[4].y < results.multi_hand_landmarks[0].landmark[8].y) and (
                    results.multi_hand_landmarks[0].landmark[8].y > results.multi_hand_landmarks[0].landmark[12].y) and (
                    results.multi_hand_landmarks[0].landmark[12].y > results.multi_hand_landmarks[0].landmark[16].y) and (
                    results.multi_hand_landmarks[0].landmark[16].y > results.multi_hand_landmarks[0].landmark[20].y)):
                print("I")
            elif ((results.multi_hand_landmarks[0].landmark[4].y > results.multi_hand_landmarks[0].landmark[8].y) and (
                    results.multi_hand_landmarks[0].landmark[8].y < results.multi_hand_landmarks[0].landmark[12].y) and (
                    results.multi_hand_landmarks[0].landmark[12].y < results.multi_hand_landmarks[0].landmark[16].y) and (
                    results.multi_hand_landmarks[0].landmark[16].y > results.multi_hand_landmarks[0].landmark[20].y)):
                print("L")

        elif ((results.multi_hand_landmarks[0].landmark[4].x < results.multi_hand_landmarks[0].landmark[8].x) and (
                results.multi_hand_landmarks[0].landmark[8].x > results.multi_hand_landmarks[0].landmark[12].x) and (
                results.multi_hand_landmarks[0].landmark[12].x > results.multi_hand_landmarks[0].landmark[16].x) and (
                results.multi_hand_landmarks[0].landmark[16].x > results.multi_hand_landmarks[0].landmark[20].x)):
            if ((results.multi_hand_landmarks[0].landmark[4].y > results.multi_hand_landmarks[0].landmark[8].y) and (
                    results.multi_hand_landmarks[0].landmark[8].y > results.multi_hand_landmarks[0].landmark[12].y) and (
                    results.multi_hand_landmarks[0].landmark[12].y > results.multi_hand_landmarks[0].landmark[16].y) and (
                    results.multi_hand_landmarks[0].landmark[16].y > results.multi_hand_landmarks[0].landmark[20].y)):
                print("E")
            elif ((results.multi_hand_landmarks[0].landmark[4].y < results.multi_hand_landmarks[0].landmark[8].y) and (
                    results.multi_hand_landmarks[0].landmark[8].y > results.multi_hand_landmarks[0].landmark[12].y) and (
                    results.multi_hand_landmarks[0].landmark[12].y < results.multi_hand_landmarks[0].landmark[16].y) and (
                    results.multi_hand_landmarks[0].landmark[16].y > results.multi_hand_landmarks[0].landmark[20].y)):
                print("N")
            elif ((results.multi_hand_landmarks[0].landmark[4].y > results.multi_hand_landmarks[0].landmark[8].y) and (
                    results.multi_hand_landmarks[0].landmark[8].y > results.multi_hand_landmarks[0].landmark[12].y) and (
                    results.multi_hand_landmarks[0].landmark[12].y < results.multi_hand_landmarks[0].landmark[16].y) and (
                    results.multi_hand_landmarks[0].landmark[16].y < results.multi_hand_landmarks[0].landmark[20].y)):
                if ((results.multi_hand_landmarks[0].landmark[4].z < results.multi_hand_landmarks[0].landmark[8].z) and (
                        results.multi_hand_landmarks[0].landmark[8].z > results.multi_hand_landmarks[0].landmark[12].z) and (
                        results.multi_hand_landmarks[0].landmark[12].z > results.multi_hand_landmarks[0].landmark[16].z) and (
                        results.multi_hand_landmarks[0].landmark[16].z < results.multi_hand_landmarks[0].landmark[20].z)):
                    print("B")
                elif ((results.multi_hand_landmarks[0].landmark[4].z > results.multi_hand_landmarks[0].landmark[8].z) and (
                        results.multi_hand_landmarks[0].landmark[8].z < results.multi_hand_landmarks[0].landmark[12].z) and (
                        results.multi_hand_landmarks[0].landmark[12].z < results.multi_hand_landmarks[0].landmark[16].z) and (
                        results.multi_hand_landmarks[0].landmark[16].z > results.multi_hand_landmarks[0].landmark[20].z)):
                    print("Q")
                else:
                    print("U")
            elif ((results.multi_hand_landmarks[0].landmark[4].y > results.multi_hand_landmarks[0].landmark[8].y) and (
                    results.multi_hand_landmarks[0].landmark[8].y < results.multi_hand_landmarks[0].landmark[12].y) and (
                    results.multi_hand_landmarks[0].landmark[12].y < results.multi_hand_landmarks[0].landmark[16].y) and (
                    results.multi_hand_landmarks[0].landmark[16].y < results.multi_hand_landmarks[0].landmark[20].y)):
                if ((results.multi_hand_landmarks[0].landmark[4].z > results.multi_hand_landmarks[0].landmark[8].z) and (
                        results.multi_hand_landmarks[0].landmark[8].z < results.multi_hand_landmarks[0].landmark[12].z) and (
                        results.multi_hand_landmarks[0].landmark[12].z < results.multi_hand_landmarks[0].landmark[16].z) and (
                        results.multi_hand_landmarks[0].landmark[16].z > results.multi_hand_landmarks[0].landmark[20].z)):
                    print("G")
                elif ((results.multi_hand_landmarks[0].landmark[4].z < results.multi_hand_landmarks[0].landmark[8].z) and (
                        results.multi_hand_landmarks[0].landmark[8].z > results.multi_hand_landmarks[0].landmark[12].z) and (
                        results.multi_hand_landmarks[0].landmark[12].z > results.multi_hand_landmarks[0].landmark[16].z) and (
                        results.multi_hand_landmarks[0].landmark[16].z > results.multi_hand_landmarks[0].landmark[20].z)):
                    print("O")
                else:
                    print("X")
            elif ((results.multi_hand_landmarks[0].landmark[4].y < results.multi_hand_landmarks[0].landmark[8].y) and (
                    results.multi_hand_landmarks[0].landmark[8].y < results.multi_hand_landmarks[0].landmark[12].y) and (
                    results.multi_hand_landmarks[0].landmark[12].y < results.multi_hand_landmarks[0].landmark[16].y) and (
                    results.multi_hand_landmarks[0].landmark[16].y > results.multi_hand_landmarks[0].landmark[20].y)):
                if ((results.multi_hand_landmarks[0].landmark[4].z < results.multi_hand_landmarks[0].landmark[8].z) and (
                        results.multi_hand_landmarks[0].landmark[8].z > results.multi_hand_landmarks[0].landmark[12].z) and (
                        results.multi_hand_landmarks[0].landmark[12].z > results.multi_hand_landmarks[0].landmark[16].z) and (
                        results.multi_hand_landmarks[0].landmark[16].z > results.multi_hand_landmarks[0].landmark[20].z)):
                    print("M")
                else:
                    print("S")

        elif ((results.multi_hand_landmarks[0].landmark[4].x > results.multi_hand_landmarks[0].landmark[8].x) and (
                results.multi_hand_landmarks[0].landmark[8].x < results.multi_hand_landmarks[0].landmark[12].x) and (
                results.multi_hand_landmarks[0].landmark[12].x > results.multi_hand_landmarks[0].landmark[16].x) and (
                results.multi_hand_landmarks[0].landmark[16].x > results.multi_hand_landmarks[0].landmark[20].x)):
            if ((results.multi_hand_landmarks[0].landmark[4].y > results.multi_hand_landmarks[0].landmark[8].y) and (
                    results.multi_hand_landmarks[0].landmark[8].y > results.multi_hand_landmarks[0].landmark[12].y) and (
                    results.multi_hand_landmarks[0].landmark[12].y < results.multi_hand_landmarks[0].landmark[16].y) and (
                    results.multi_hand_landmarks[0].landmark[16].y < results.multi_hand_landmarks[0].landmark[20].y)):
                print("C")
            else:
                print("D")

        elif ((results.multi_hand_landmarks[0].landmark[4].x < results.multi_hand_landmarks[0].landmark[8].x) and (
                results.multi_hand_landmarks[0].landmark[8].x < results.multi_hand_landmarks[0].landmark[12].x) and (
                results.multi_hand_landmarks[0].landmark[12].x > results.multi_hand_landmarks[0].landmark[16].x) and (
                results.multi_hand_landmarks[0].landmark[16].x > results.multi_hand_landmarks[0].landmark[20].x)):
            if ((results.multi_hand_landmarks[0].landmark[4].y > results.multi_hand_landmarks[0].landmark[12].y) and (
                    results.multi_hand_landmarks[0].landmark[8].y < results.multi_hand_landmarks[0].landmark[12].y) and (
                    results.multi_hand_landmarks[0].landmark[12].y < results.multi_hand_landmarks[0].landmark[16].y) and (
                    results.multi_hand_landmarks[0].landmark[16].y < results.multi_hand_landmarks[0].landmark[20].y)):
                print("H")
            elif ((results.multi_hand_landmarks[0].landmark[4].y < results.multi_hand_landmarks[0].landmark[12].y) and (
                    results.multi_hand_landmarks[0].landmark[8].y < results.multi_hand_landmarks[0].landmark[12].y) and (
                    results.multi_hand_landmarks[0].landmark[12].y < results.multi_hand_landmarks[0].landmark[16].y) and (
                    results.multi_hand_landmarks[0].landmark[16].y > results.multi_hand_landmarks[0].landmark[20].y)):
                print("P")
            elif ((results.multi_hand_landmarks[0].landmark[4].y > results.multi_hand_landmarks[0].landmark[8].y) and (
                    results.multi_hand_landmarks[0].landmark[8].y > results.multi_hand_landmarks[0].landmark[12].y) and (
                    results.multi_hand_landmarks[0].landmark[12].y < results.multi_hand_landmarks[0].landmark[16].y) and (
                    results.multi_hand_landmarks[0].landmark[16].y < results.multi_hand_landmarks[0].landmark[20].y)):
                print("R")

        elif ((results.multi_hand_landmarks[0].landmark[4].x < results.multi_hand_landmarks[0].landmark[8].x) and (
                results.multi_hand_landmarks[0].landmark[8].x > results.multi_hand_landmarks[0].landmark[12].x) and (
                results.multi_hand_landmarks[0].landmark[12].x < results.multi_hand_landmarks[0].landmark[16].x) and (
                results.multi_hand_landmarks[0].landmark[16].x > results.multi_hand_landmarks[0].landmark[20].x)):
            if ((results.multi_hand_landmarks[0].landmark[4].z > results.multi_hand_landmarks[0].landmark[12].z) and (
                    results.multi_hand_landmarks[0].landmark[8].z > results.multi_hand_landmarks[0].landmark[12].z) and (
                    results.multi_hand_landmarks[0].landmark[12].z < results.multi_hand_landmarks[0].landmark[16].z) and (
                    results.multi_hand_landmarks[0].landmark[16].z > results.multi_hand_landmarks[0].landmark[20].z)):
                print("K")
            elif ((results.multi_hand_landmarks[0].landmark[4].z < results.multi_hand_landmarks[0].landmark[12].z) and (
                    results.multi_hand_landmarks[0].landmark[8].z > results.multi_hand_landmarks[0].landmark[12].z) and (
                    results.multi_hand_landmarks[0].landmark[12].z < results.multi_hand_landmarks[0].landmark[16].z) and (
                    results.multi_hand_landmarks[0].landmark[16].z > results.multi_hand_landmarks[0].landmark[20].z)):
                print("V")

        elif ((results.multi_hand_landmarks[0].landmark[4].x < results.multi_hand_landmarks[0].landmark[8].x) and (
                results.multi_hand_landmarks[0].landmark[8].x > results.multi_hand_landmarks[0].landmark[12].x) and (
                results.multi_hand_landmarks[0].landmark[12].x > results.multi_hand_landmarks[0].landmark[16].x) and (
                results.multi_hand_landmarks[0].landmark[16].x < results.multi_hand_landmarks[0].landmark[20].x)):
            print("W")

      #print("x is: ", str(x), "y is: ", str(y), "z is ", str(z))
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
