import cv2
import mediapipe as mp
import json
import numpy as np
import sys
import os

from utils.angles import calculate_angle
from utils.config import EXERCISE_CONFIG, KEYPOINT_MAP

class RealTimeTracker:
    def __init__(self, exercise_name):
        self.exercise_name = exercise_name
        self.config = EXERCISE_CONFIG.get(exercise_name)
        if not self.config:
            print(f"FATAL ERROR: Exercise '{exercise_name}' not found.")
            sys.exit(1)
            
        self.feature_defs = self.config["FEATURES"]
        self.rep_logic = self.config["REP_LOGIC"]
        self.primary_feature_names = self.rep_logic["PRIMARY_FEATURES"]
        self.rep_strategy = self.rep_logic.get("REP_STRATEGY", "average")
        
        self.frame_buffer_threshold = 10
        self.accuracy = 0.0
        self.total_reps = 0
        self.state = "up"

        self.rep_counters = {}
        for feature in self.primary_feature_names:
            self.rep_counters[feature] = {
                "count": 0, "state": "up", "up_frames": 0, "down_frames": 0
            }

        self.reference_data = self._load_reference_data()
        self.ref_down_angles, self.ref_up_angles = self._get_ref_angles()

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)

    def _load_reference_data(self):
        ref_file = os.path.join("reference_data", f"{self.exercise_name}_ref.json")
        try:
            with open(ref_file, "r") as f: return json.load(f)
        except Exception as e:
            print(f"FATAL ERROR: Could not load or parse '{ref_file}'. Error: {e}\nPlease recalibrate the exercise.")
            sys.exit(1)

    def _get_ref_angles(self):
        down_angles, up_angles = [], []
        # Check if reference data is in the new split format
        if 'downward' not in self.reference_data or 'upward' not in self.reference_data:
             print(f"FATAL ERROR: Reference file for '{self.exercise_name}' is outdated. Please recalibrate.")
             sys.exit(1)

        for frame in self.reference_data['downward']:
            angles = [frame['features'][list(self.feature_defs.keys()).index(pf)] for pf in self.primary_feature_names]
            down_angles.append(np.mean(angles))
        for frame in self.reference_data['upward']:
            angles = [frame['features'][list(self.feature_defs.keys()).index(pf)] for pf in self.primary_feature_names]
            up_angles.append(np.mean(angles))
        return np.array(down_angles), np.array(up_angles)

    def _get_synced_reference_frame(self, user_primary_angle):
        if self.state == "down":
            search_angles = self.ref_down_angles
            search_data = self.reference_data['downward']
        else:
            search_angles = self.ref_up_angles
            search_data = self.reference_data['upward']
        if not search_angles.any(): return None
        closest_index = np.argmin(np.abs(search_angles - user_primary_angle))
        return search_data[closest_index]

    def _get_robust_angle(self, user_angles_dict, landmarks):
        if len(self.primary_feature_names) == 1:
            return user_angles_dict.get(self.primary_feature_names[0])
        if len(self.primary_feature_names) == 2:
            name1, name2 = self.primary_feature_names[0], self.primary_feature_names[1]
            angle1, angle2 = user_angles_dict.get(name1), user_angles_dict.get(name2)
            kps1 = self.feature_defs[name1]['keypoints']
            vis1 = np.mean([landmarks[KEYPOINT_MAP[kp]].visibility for kp in kps1])
            kps2 = self.feature_defs[name2]['keypoints']
            vis2 = np.mean([landmarks[KEYPOINT_MAP[kp]].visibility for kp in kps2])
            if vis1 > vis2 + 0.2: return angle1
            if vis2 > vis1 + 0.2: return angle2
        primary_angles = [user_angles_dict[name] for name in self.primary_feature_names]
        return np.mean(primary_angles)

    def _update_rep_counter(self, user_angles_dict, robust_angle):
        up_thresh, down_thresh = self.rep_logic["UP_THRESHOLD"], self.rep_logic["DOWN_THRESHOLD"]
        if self.rep_strategy == 'independent':
            for feature, counter in self.rep_counters.items():
                angle = user_angles_dict.get(feature, up_thresh)
                if angle > up_thresh:
                    counter["up_frames"] += 1; counter["down_frames"] = 0
                    if counter["up_frames"] >= self.frame_buffer_threshold and counter["state"] == "down":
                        counter["count"] += 1; counter["state"] = "up"
                elif angle < down_thresh:
                    counter["down_frames"] += 1; counter["up_frames"] = 0
                    if counter["down_frames"] >= self.frame_buffer_threshold:
                        counter["state"] = "down"
            self.total_reps = sum(c['count'] for c in self.rep_counters.values())
        elif self.rep_strategy == 'average':
            counter = list(self.rep_counters.values())[0]
            if robust_angle > up_thresh:
                counter["up_frames"] += 1; counter["down_frames"] = 0
                if counter["up_frames"] >= self.frame_buffer_threshold and self.state == "down":
                    counter["count"] += 1; self.state = "up"
            elif robust_angle < down_thresh:
                counter["down_frames"] += 1; counter["up_frames"] = 0
                if counter["down_frames"] >= self.frame_buffer_threshold:
                    self.state = "down"
            self.total_reps = counter['count']

    def _calculate_accuracy(self, user_angles_dict, ref_frame):
        ref_angles_list = ref_frame['features']
        ref_angles_dict = {name: ref_angles_list[i] for i, name in enumerate(self.feature_defs.keys())}
        total_error, total_weight = 0, 0
        for name, user_angle in user_angles_dict.items():
            feature_config = self.feature_defs[name]
            weight = feature_config.get('weight', 1)
            error = abs(user_angle - ref_angles_dict[name])
            total_error += error * weight
            total_weight += weight
        if total_weight == 0: self.accuracy = 100.0; return
        weighted_average_error = total_error / total_weight
        self.accuracy = max(0, 100 - weighted_average_error)

# MODIFIED: Added a try-except block to catch and report the exact error
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("FATAL ERROR: Cannot open camera.")
            return
        
        print(f"--- Starting tracker for {self.rep_logic['NAME']}. Press 'q' to quit. ---")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            current_angle_for_ui = 0
            if results.pose_landmarks:
                # NEW: Wrap the entire detection block in a try-except
                try:
                    user_landmarks = results.pose_landmarks.landmark
                    user_angles_dict = {}
                    for name, feature_data in self.feature_defs.items():
                        kps = feature_data['keypoints']
                        p1, p2, p3 = KEYPOINT_MAP[kps[0]], KEYPOINT_MAP[kps[1]], KEYPOINT_MAP[kps[2]]
                        angle = calculate_angle([user_landmarks[p1].x, user_landmarks[p1].y], 
                                                [user_landmarks[p2].x, user_landmarks[p2].y], 
                                                [user_landmarks[p3].x, user_landmarks[p3].y])
                        user_angles_dict[name] = angle
                    
                    primary_angles_for_sync = [user_angles_dict[name] for name in self.primary_feature_names]
                    if self.rep_strategy == 'independent' and primary_angles_for_sync:
                        sync_angle = min(primary_angles_for_sync)
                        active_limb_index = np.argmin(primary_angles_for_sync)
                        active_limb_name = self.primary_feature_names[active_limb_index]
                        self.state = self.rep_counters[active_limb_name]['state']
                    else:
                        sync_angle = np.mean(primary_angles_for_sync)
                    
                    robust_angle = self._get_robust_angle(user_angles_dict, user_landmarks)
                    if robust_angle is not None:
                        self._update_rep_counter(user_angles_dict, robust_angle)
                        current_angle_for_ui = robust_angle

                        ref_frame = self._get_synced_reference_frame(sync_angle)
                        if ref_frame:
                            # self._draw_ghost(image, ref_frame['landmarks'])
                            self._calculate_accuracy(user_angles_dict, ref_frame)
                    
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
                
                except Exception as e:
                    # This will print the exact error to your console
                    print(f"ERROR during processing: {e}")
                    import traceback
                    traceback.print_exc() # This prints the full error trace
            
            self._draw_ui(image, current_angle_for_ui)
            cv2.imshow('Virtual PT', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    # MODIFIED: A new clean UI with an accuracy bar
    def _draw_ui(self, image, current_angle):
        h, w, _ = image.shape
        # --- Rep Counter and Accuracy Score ---
        cv2.putText(image, f"Reps: {self.total_reps}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(image, f"Accuracy: {self.accuracy:.1f}%", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        # --- Accuracy Bar ---
        bar_x, bar_y, bar_height = 50, 120, 30
        bar_width = w - 100
        # Determine bar color
        if self.accuracy > 85: bar_color = (0, 255, 0) # Green
        elif self.accuracy > 60: bar_color = (0, 255, 255) # Yellow
        else: bar_color = (0, 0, 255) # Red
        # Calculate fill width
        fill_width = int((self.accuracy / 100) * bar_width)
        # Draw bar background and fill
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 3)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), bar_color, -1)

    def _draw_ghost(self, image, ghost_landmarks_data):
        # This method is no longer called but is kept for potential future use
        try:
            h, w, _ = image.shape
            for connection in self.mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(ghost_landmarks_data) and end_idx < len(ghost_landmarks_data):
                    start_pos = (int(ghost_landmarks_data[start_idx]['x'] * w), int(ghost_landmarks_data[start_idx]['y'] * h))
                    end_pos = (int(ghost_landmarks_data[end_idx]['x'] * w), int(ghost_landmarks_data[end_idx]['y'] * h))
                    cv2.line(image, start_pos, end_pos, (255, 255, 255), 5)
        except Exception as e:
            print(f"WARNING: Could not draw ghost overlay. Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <exercise_name>")
        print("Available exercises:", ", ".join(EXERCISE_CONFIG.keys()))
        sys.exit(1)
    
    tracker = RealTimeTracker(sys.argv[1].lower())
    tracker.run()