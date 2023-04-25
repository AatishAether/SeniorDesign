import pandas as pd
import numpy as np
from collections import Counter

from utils.dtw import dtw_distances
from models.sign_model import SignModel
from utils.landmark_utils import extract_landmarks


class SignRecorder(object):
    def __init__(self, reference_signs: pd.DataFrame, seq_len=40):
        # Variables for recording
        self.is_recording = False
        self.seq_len = seq_len

        # List of results stored each frame
        self.recorded_results = []

        # DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        self.reference_signs = reference_signs
        #self.recorded_sign = SignModel()
        

    def record(self):
        """
        Initialize sign_distances & start recording
        """

        ##Reset distance to all signs to 0
        self.reference_signs["distance"].values[:] = 0
        self.is_recording = True

    def process_results(self, results) :
        """
        If the SignRecorder is in the recording state:
            it stores the landmarks during seq_len frames and then computes the sign distances
        :param results: mediapipe output
        :return: Return the word predicted (blank text if there is no distances)
                & the recording state
        """
        if self.is_recording:
            if len(self.recorded_results) < self.seq_len:
                ##We don't have an entire time series for DTW yet
                self.recorded_results.append(results)
            else:
                #Once we do, compute distances of ALL reference signs
                self.compute_distances()                
                print(self.reference_signs)

        if np.sum(self.reference_signs["distance"].values) == 0:
            return "", self.is_recording
        return self._get_sign_predicted(), self.is_recording

    def get_features(self,results):
        pose_list,left_hand_list, right_hand_list = [],[],[]

        for result in results:
            pose,left_hand,right_hand = extract_landmarks(results)
            left_hand_list.append(left_hand)
            right_hand_list.append(right_hand)
            pose_list.append(pose)
        return pose_list,left_hand_list,right_hand_list



            
    def compute_distances(self):
        """
        Updates the distance column of the reference_signs
        and resets recording variables
        """
        pose_list, left_hand_list, right_hand_list = [], [],[]
        for results in self.recorded_results:
            ##Flattens the landmarks to a 1D array
            pose, left_hand, right_hand = extract_landmarks(results)
            left_hand_list.append(left_hand)
            right_hand_list.append(right_hand)
            pose_list.append(pose)
            

        # Create a SignModel object with the landmarks gathered during recording
        self.recorded_sign = SignModel(pose_list,left_hand_list, right_hand_list)
        #print(recorded_sign.lh_embedding)

        # Compute sign similarity with DTW (ascending order)
        self.reference_signs = dtw_distances(self.recorded_sign, self.reference_signs)

        
        # Reset variables
        self.recorded_results = []
        self.is_recording = False
        

    def _get_sign_predicted(self, batch_size=5, threshold=0.5):
        """
        Method that outputs the sign that appears the most in the list of closest
        reference signs, only if its propo
        rtion within the batch is greater than the threshold

        :param batch_size: Size of the batch of reference signs that will be compared to the recorded sign
        :param threshold: If the proportion of the most represented sign in the batch is greater than threshold,
                        we output the sign_name
                          If not,
                        we output "Sign not found"
        :return: The name of the predicted sign
        """
        # Get the list (of size batch_size) of the most similar reference signs
        sign_names = self.reference_signs.iloc[:batch_size]["name"].values
        # Count the occurrences of each sign and sort them by descending order
        sign_counter = Counter(sign_names).most_common()

        predicted_sign, count = sign_counter[0]
        if count / batch_size < threshold:
            return "Unknown Sign"
        return sign_counter[0][0]
