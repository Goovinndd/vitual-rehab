import sys
import os
import json
import re
import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QListWidget, QLabel,
                             QLineEdit, QFileDialog, QMessageBox, QDialog,
                             QFormLayout, QComboBox, QListWidgetItem, QTextEdit,
                             QProgressDialog, QSpinBox, QDialogButtonBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtCore import Qt, QPoint

# --- Add utils to path ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from angles import calculate_angle
from config import EXERCISE_CONFIG, KEYPOINT_MAP

# --- Configuration Writing Utility ---
def write_config_file(config_data):
    # This helper function needs to be updated to handle the new FEATURES structure
    def custom_encoder(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "utils", "config.py")
    content = "import mediapipe as mp\n\n"
    content += "KEYPOINT_MAP = " + json.dumps(KEYPOINT_MAP, indent=4) + "\n\n"
    # Use a custom encoder to handle potential numpy integers from QSpinBox
    content += "EXERCISE_CONFIG = " + json.dumps(config_data, indent=4, default=custom_encoder)
    content = content.replace('"', "'")
    with open(CONFIG_FILE_PATH, "w") as f:
        f.write(content)

# --- (ThresholdDialog, ClickableLabel, MainWindow remain the same as before) ---
# --- Dialog for manually editing thresholds ---
class ThresholdDialog(QDialog):
    def __init__(self, detected_up, detected_down, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Adjust Thresholds")
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(QLabel("Auto-detected thresholds are shown below.\nAdjust them to sensible values for your exercise."))
        form_layout = QFormLayout()
        self.up_spinbox = QSpinBox()
        self.up_spinbox.setRange(0, 180)
        self.up_spinbox.setValue(int(detected_up))
        self.down_spinbox = QSpinBox()
        self.down_spinbox.setRange(0, 180)
        self.down_spinbox.setValue(int(detected_down))
        form_layout.addRow("UP Threshold (Extended Position):", self.up_spinbox)
        form_layout.addRow("DOWN Threshold (Contracted Position):", self.down_spinbox)
        self.layout.addLayout(form_layout)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)
    def get_values(self):
        return self.up_spinbox.value(), self.down_spinbox.value()

# --- Clickable Label for Skeleton Diagram ---
class ClickableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.landmark_positions = {
    'LEFT_ANKLE': (0.306, 0.9047),'LEFT_EAR': (0.3046, 0.1173),'LEFT_ELBOW': (0.3893, 0.3651),'LEFT_EYE': (0.2814, 0.0704),
    'LEFT_EYE_INNER': (0.2678, 0.0689),'LEFT_EYE_OUTER': (0.2923, 0.0704),'LEFT_FOOT_INDEX': (0.3429, 0.9648),'LEFT_HEEL': (0.2883, 0.9575),
    'LEFT_HIP': (0.3046, 0.5308),'LEFT_INDEX': (0.4713, 0.2522),'LEFT_KNEE': (0.3333, 0.7111),'LEFT_PINKY': (0.4863, 0.3167),
    'LEFT_SHOULDER': (0.332, 0.2669),'LEFT_THUMB': (0.4426, 0.2683),'LEFT_WRIST': (0.444, 0.3211),'MOUTH_LEFT': (0.2678, 0.1613),
    'MOUTH_RIGHT': (0.2336, 0.1628),'NOSE': (0.25, 0.1217),'RIGHT_ANKLE': (0.1954, 0.9091),'RIGHT_EAR': (0.194, 0.1202),
    'RIGHT_ELBOW': (0.112, 0.3651),'RIGHT_EYE': (0.2186, 0.0718),'RIGHT_EYE_INNER': (0.235, 0.0733),'RIGHT_EYE_OUTER': (0.2077, 0.0733),
    'RIGHT_FOOT_INDEX': (0.153, 0.9633),'RIGHT_HEEL': (0.2117, 0.9575),'RIGHT_HIP': (0.194, 0.5308),'RIGHT_INDEX': (0.0287, 0.2507),
    'RIGHT_KNEE': (0.168, 0.7111),'RIGHT_PINKY': (0.0109, 0.3196),'RIGHT_SHOULDER': (0.1667, 0.2669),'RIGHT_THUMB': (0.056, 0.2683),
    'RIGHT_WRIST': (0.0546, 0.3196),
}
        self.setPixmap(QPixmap("utils/skeleton.png"))
        self.setScaledContents(True)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if len(self.points) < 3:
                x, y = event.pos().x(), event.pos().y()
                norm_x, norm_y = x / self.width(), y / self.height()
                if not self.landmark_positions: return
                closest_landmark = min(self.landmark_positions.keys(),
                                       key=lambda k: ((self.landmark_positions[k][0] - norm_x)**2 +
                                                      (self.landmark_positions[k][1] - norm_y)**2)**0.5)
                self.points.append(closest_landmark)
                self.update()
        elif event.button() == Qt.RightButton:
            if self.points: self.points.pop()
            self.update()
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.landmark_positions: return
        painter = QPainter(self)
        pen = QPen(QColor(0, 255, 0), 3)
        painter.setPen(pen)
        for point_name in self.points:
            if point_name in self.landmark_positions:
                pos = self.landmark_positions[point_name]
                painter.drawEllipse(QPoint(int(pos[0] * self.width()), int(pos[1] * self.height())), 5, 5)
        if len(self.points) >= 2:
            p1, p2 = self.landmark_positions[self.points[0]], self.landmark_positions[self.points[1]]
            painter.drawLine(int(p1[0] * self.width()), int(p1[1] * self.height()),
                             int(p2[0] * self.width()), int(p2[1] * self.height()))
        if len(self.points) == 3:
            p2, p3 = self.landmark_positions[self.points[1]], self.landmark_positions[self.points[2]]
            painter.drawLine(int(p2[0] * self.width()), int(p2[1] * self.height()),
                             int(p3[0] * self.width()), int(p3[1] * self.height()))
    def clear_points(self):
        self.points, self.update()

# MODIFIED: The ExerciseDialog is updated to handle weights
class ExerciseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Exercise")
        self.layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        self.ex_name_input = QLineEdit()
        self.display_name_input = QLineEdit()
        form_layout.addRow("Exercise ID (e.g., 'overhead_press'):", self.ex_name_input)
        form_layout.addRow("Display Name (e.g., 'Overhead Press'):", self.display_name_input)
        self.layout.addLayout(form_layout)

        self.angle_layout = QHBoxLayout()
        self.skeleton_label = ClickableLabel()
        
        self.angle_controls = QVBoxLayout()
        self.angle_controls.addWidget(QLabel("1. Left-click 3 joints to define an angle."))
        self.angle_controls.addWidget(QLabel("2. Right-click to undo the last point."))
        
        # NEW: Angle creation form layout
        angle_form = QFormLayout()
        self.angle_name_input = QLineEdit()
        self.angle_weight_spinbox = QSpinBox() # NEW: Spinbox for weight input
        self.angle_weight_spinbox.setRange(1, 5)
        self.angle_weight_spinbox.setValue(1)
        angle_form.addRow("Angle Name (e.g., 'L_ELBOW'):", self.angle_name_input)
        angle_form.addRow("Accuracy Weight (1-5):", self.angle_weight_spinbox)
        self.angle_controls.addLayout(angle_form)

        self.add_angle_button = QPushButton("Add Angle to List")
        self.add_angle_button.clicked.connect(self.add_angle)
        self.angle_controls.addWidget(self.add_angle_button)
        
        self.angle_list = QListWidget()
        self.angle_controls.addWidget(QLabel("Defined Angles:"))
        self.angle_controls.addWidget(self.angle_list)
        
        self.primary_feature_combo = QComboBox()
        self.angle_controls.addWidget(QLabel("Select Primary Angle for Rep Counting:"))
        self.angle_controls.addWidget(self.primary_feature_combo)
        
        self.angle_layout.addWidget(self.skeleton_label, 2)
        self.angle_layout.addLayout(self.angle_controls, 1)
        self.layout.addLayout(self.angle_layout)
        
        self.calibrate_button = QPushButton("Define and Calibrate from Video")
        self.calibrate_button.clicked.connect(self.accept)
        self.layout.addWidget(self.calibrate_button)
        
        self.angles = {}

    # MODIFIED: add_angle now saves the weight
    def add_angle(self):
        if len(self.skeleton_label.points) != 3:
            QMessageBox.warning(self, "Warning", "Please select exactly 3 points.")
            return
        angle_name = self.angle_name_input.text().strip().upper()
        if not angle_name:
            QMessageBox.warning(self, "Warning", "Please provide a name for the angle.")
            return
        
        weight = self.angle_weight_spinbox.value() # NEW: Get weight from spinbox
        angle_def = tuple(self.skeleton_label.points)
        
        # MODIFIED: Store in the new dictionary format
        self.angles[angle_name] = {'keypoints': angle_def, 'weight': weight}

        # MODIFIED: Update the list display to show the weight
        self.angle_list.addItem(f"{angle_name} (Weight: {weight})")
        self.primary_feature_combo.addItem(angle_name)
        
        self.skeleton_label.clear_points()
        self.angle_name_input.clear()
        self.angle_weight_spinbox.setValue(1)

    def get_data(self):
        if not self.ex_name_input.text() or not self.display_name_input.text() or not self.angles:
            return None
        return {
            "ex_name": self.ex_name_input.text().lower().replace(" ", "_"),
            "display_name": self.display_name_input.text(),
            "features": self.angles,
            "primary_feature": self.primary_feature_combo.currentText()
        }
        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Virtual PT Calibration Tool")
        self.setGeometry(100, 100, 900, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.list_label = QLabel("Configured Exercises")
        self.exercise_list = QListWidget()
        self.add_button = QPushButton("Add New Exercise")
        self.add_button.clicked.connect(self.add_exercise)
        self.layout.addWidget(self.list_label)
        self.layout.addWidget(self.exercise_list)
        self.layout.addWidget(self.add_button)
        self.load_exercises()

    def load_exercises(self):
        self.exercise_list.clear()
        for ex_name, data in EXERCISE_CONFIG.items():
            item = QListWidgetItem(f"{data['REP_LOGIC']['NAME']} ({ex_name})")
            self.exercise_list.addItem(item)
    
    def add_exercise(self):
        dialog = ExerciseDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()
            if data: self.process_calibration(data)
            else: QMessageBox.critical(self, "Error", "Incomplete information. Exercise not added.")

    def process_calibration(self, data):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Reference Video", "", "Video Files (*.mp4 *.avi)")
        if not file_name: return
        
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        cap = cv2.VideoCapture(file_name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = QProgressDialog("Processing Video...", "Cancel", 0, total_frames, self)
        progress.setWindowModality(Qt.WindowModal)
        
        reference_data, frame_num = [], 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_num += 1
            progress.setValue(frame_num)
            if progress.wasCanceled(): break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                features = []
                # MODIFIED: Extract keypoints from the new dictionary structure
                for name, feature_data in data['features'].items():
                    kps = feature_data['keypoints']
                    p1, p2, p3 = KEYPOINT_MAP[kps[0]], KEYPOINT_MAP[kps[1]], KEYPOINT_MAP[kps[2]]
                    angle = calculate_angle([landmarks[p1].x, landmarks[p1].y], [landmarks[p2].x, landmarks[p2].y], [landmarks[p3].x, landmarks[p3].y])
                    features.append(angle)
                raw_landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility} for lm in landmarks]
                reference_data.append({'landmarks': raw_landmarks, 'features': features})
        
        progress.setValue(total_frames)
        cap.release()
        if not reference_data:
            QMessageBox.critical(self, "Error", "Could not detect any pose in the selected video.")
            return

        primary_feature_index = list(data['features'].keys()).index(data['primary_feature'])
        primary_angles = [frame['features'][primary_feature_index] for frame in reference_data]
        
        detected_up, detected_down = np.max(primary_angles), np.min(primary_angles)

        threshold_dialog = ThresholdDialog(detected_up, detected_down, self)
        if threshold_dialog.exec_() == QDialog.Accepted:
            final_up, final_down = threshold_dialog.get_values()
        else: return
        
        peak_frame_index = np.argmin(primary_angles)
        downward_motion_data = reference_data[:peak_frame_index + 1]
        upward_motion_data = reference_data[peak_frame_index:]
        split_reference_data = {'downward': downward_motion_data, 'upward': upward_motion_data}

        EXERCISE_CONFIG[data['ex_name']] = {
            'FEATURES': data['features'],
            'REP_LOGIC': {
                'PRIMARY_FEATURES': [data['primary_feature']],
                'REP_STRATEGY': 'average',
                'UP_THRESHOLD': final_up,
                'DOWN_THRESHOLD': final_down,
                'NAME': data['display_name']
            },
            'FORM_RULES': []
        }
        write_config_file(EXERCISE_CONFIG)

        os.makedirs("reference_data", exist_ok=True)
        json_path = os.path.join("reference_data", f"{data['ex_name']}_ref.json")
        with open(json_path, "w") as f: json.dump(split_reference_data, f)
        
        QMessageBox.information(self, "Success", f"Exercise '{data['display_name']}' was calibrated and saved successfully!")
        self.load_exercises()

if __name__ == "__main__":
    if not os.path.exists("utils/skeleton.png"):
        QMessageBox.critical(None, "Error", "Missing 'utils/skeleton.png'. Please ensure it exists.")
    else:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())