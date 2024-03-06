import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class ImgVec:
    def __init__(self):
        self.images = []

    def add_image(self, image):
        self.images.append(image)

    def disp_imgs(self):
        num_img = len(self.images)
        if(num_img == 0):
            print("No images. ")
            return
        fig, axes = plt.subplots(1, num_img, figsize = (15, 3))
        for i, img in enumerate(self.images):
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f'Image {i + 1}')
        plt.tight_layout()
        plt.show()

tlist = []

class Vectorizer:
    def __init__(self):
        self.image = None
        self.imcopy = None
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_draw_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=.1)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True, model_complexity=1, min_detection_confidence=.1)

        self.mp_face = mp.solutions.face_mesh
        self.face = self.mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=.2)
        # self.pose.context.use_gpu = True
    def vec_img(self, image):
        self.image = image
        self.imcopy = image
        if(self.image is not None and self.imcopy is not None):
            self.image.flags.writeable = False
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(self.image)
            results_h = self.hands.process(self.image)
            results_f = self.face.process(self.imcopy)

            self.image.flags.writeable = True
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            if results_f.multi_face_landmarks:
                tinterlist = []
                print(len(self.mp_face.FACEMESH_CONTOURS))
                for f_l in results_f.multi_face_landmarks:
                    # print(f_l.landmark[0].x)
                    for l in f_l.landmark:
                        # print(f_l.landmark[i])
                        tinterlist.append([l.x, l.y, l.z])

                    self.mp_draw.draw_landmarks(
                        self.imcopy,
                        f_l,
                        self.mp_face.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None, 
                        connection_drawing_spec=self.mp_draw_styles.get_default_face_mesh_contours_style()
                    )
                print(tinterlist)

            if results_h.multi_hand_landmarks:
                if len(results_h.multi_hand_landmarks) > 1:
                    print("True")
                else:
                    print("False")
                for h_l in results_h.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        self.image,
                        h_l,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw_styles.get_default_hand_landmarks_style(),
                        self.mp_draw_styles.get_default_hand_connections_style()
                    )

            if results.pose_landmarks:
                # for landmark in results.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    self.image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_draw_styles.get_default_pose_landmarks_style()
                    # self.mp_draw_styles.get_default_pose_connections_style()
                )

            cv2.imshow('Vectorizer', self.image)
            cv2.imshow('Vectorize Face', self.imcopy)





dp = 'D:\\RWTH-Phoenix\\phoenix2014-release\\phoenix-2014-multisigner\\features\\fullFrame-210x260px\\train\\01April_2010_Thursday_heute_default-0\\1\\'
file = os.listdir(dp)

i = 0

vec_img = Vectorizer()

for img in file:
    img_use = cv2.imread(os.path.join(dp, img))
    if (i < 40):
        # cv2.imshow('Image', img_use)
        vec_img.vec_img(img_use)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    i += 1