from typing import List

import numpy as np

from models.hand_model import HandModel


class SignModel(object):
    def __init__(
        self,
        pose_list: List[List[float]],
        left_hand_list: List[List[float]],
        right_hand_list: List[List[float]]
    ):
        """
        Params
            x_hand_list: List of all landmarks for each frame of a video
        Args
            has_x_hand: bool; True if x hand is detected in the video, otherwise False
            xh_embedding: ndarray; Array of shape (n_frame, nb_connections * nb_connections)
        """
        self.has_pose = np.sum(pose_list) != 0
        self.has_left_hand = np.sum(left_hand_list) != 0
        self.has_right_hand = np.sum(right_hand_list) != 0

        #Features of each hand.
        #I think of it as self referencing rather than world coords
        ##The HandModel creates a featureList of angles of
        ###Each landmark relative to all the other landmarks
        ###This way it's not referencing where the hand landmarks are on screen but rather to themselves
        self.lh_embedding = self._get_embedding_from_landmark_list(left_hand_list)
        self.rh_embedding = self._get_embedding_from_landmark_list(right_hand_list)

    @staticmethod
    def _get_embedding_from_landmark_list(
        hand_list: List[List[float]],
    ) -> List[List[float]]:
        """
        Params
            hand_list: List of all landmarks for each frame of a video
        Return
            Array of shape (n_frame, nb_connections * nb_connections) containing
            the feature_vectors of the hand for each frame
        """
        embedding = []
        for frame_idx in range(len(hand_list)):
            if np.sum(hand_list[frame_idx]) == 0: ##If empty frame data
                continue
            
            hand_gesture = HandModel(hand_list[frame_idx])
            embedding.append(hand_gesture.feature_vector)
        return embedding
