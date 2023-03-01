import cv2
import mediapipe as mp


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
    mp_drawing_styles = mp.solutions.drawing_styles

    # Draw left hand connections
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks = results.left_hand_landmarks,
        connections=mp_hands.HAND_CONNECTIONS,
        land_drawSpec = mp_drawing_styles.get_default_hand_landmarks_style(),
        conn_drawSpec = mp_drawing_styles.get_default_hand_connections_style())

    mp_drawing.draw_landmarks(
        image,
        hand_landmarks = results.right_hand_landmarks,
        connections=mp_hands.HAND_CONNECTIONS,
        land_drawSpec = mp_drawing_styles.get_default_hand_landmarks_style(),
        conn_drawSpec = mp_drawing_styles.get_default_hand_connections_style()
    )
    return image
