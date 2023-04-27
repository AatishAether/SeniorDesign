import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = ['./test5.jpg']
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    #cv2.imwrite(
    #    '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    ## Draw hand world landmarks.
    #if not results.multi_hand_world_landmarks:
    #  continue
    #for hand_world_landmarks in results.multi_hand_world_landmarks:
    #  mp_drawing.plot_landmarks(
    #    hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

#Crop image around wrist
x = int(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].x * image.shape[1])
y = int(results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].y * image.shape[0])

w = 150
h = 150
offset = 140

crop_img = image[y-h-offset:y+h-offset, x-w:x+w]
cv2.imshow("cropped", crop_img)
cv2.imwrite("test5_cropped.jpg", crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

