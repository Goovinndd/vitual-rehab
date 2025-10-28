import sys
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QPushButton, QListWidget, QLabel, 
                             QTextEdit, QFileDialog)
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QPixmap
from PyQt5.QtCore import Qt, QPoint

# UPDATED: The full list of 33 MediaPipe Pose landmarks
LANDMARK_NAMES = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER",
    "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT",
    "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
    "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL",
    "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
]

# UPDATED: The full set of connections for the skeleton
POSE_CONNECTIONS = [
    ("NOSE", "RIGHT_EYE_INNER"), ("RIGHT_EYE_INNER", "RIGHT_EYE"),
    ("RIGHT_EYE", "RIGHT_EYE_OUTER"), ("RIGHT_EYE_OUTER", "RIGHT_EAR"),
    ("NOSE", "LEFT_EYE_INNER"), ("LEFT_EYE_INNER", "LEFT_EYE"),
    ("LEFT_EYE", "LEFT_EYE_OUTER"), ("LEFT_EYE_OUTER", "LEFT_EAR"),
    ("MOUTH_RIGHT", "MOUTH_LEFT"),
    ("RIGHT_SHOULDER", "LEFT_SHOULDER"), ("RIGHT_HIP", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"), ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_WRIST", "RIGHT_PINKY"), ("RIGHT_WRIST", "RIGHT_INDEX"),
    ("RIGHT_WRIST", "RIGHT_THUMB"), ("RIGHT_PINKY", "RIGHT_INDEX"),
    ("LEFT_WRIST", "LEFT_PINKY"), ("LEFT_WRIST", "LEFT_INDEX"),
    ("LEFT_WRIST", "LEFT_THUMB"), ("LEFT_PINKY", "LEFT_INDEX"),
    ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_ANKLE", "RIGHT_HEEL"), ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
    ("RIGHT_HEEL", "RIGHT_FOOT_INDEX"),
    ("LEFT_ANKLE", "LEFT_HEEL"), ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    ("LEFT_HEEL", "LEFT_FOOT_INDEX"),
]

class ImageCanvas(QWidget):
    """A widget for drawing and dragging points over a background image."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = None
        self.placed_landmarks = {}
        self.dragged_landmark = None
        self.setMinimumSize(400, 600)
        self.setStyleSheet("background-color: #333; border: 1px solid #555;")

    def set_image(self, file_path):
        self.pixmap = QPixmap(file_path)
        self.placed_landmarks = {} # Clear points on new image
        self.update()

    def add_landmark(self, name, position):
        if name not in self.placed_landmarks:
            self.placed_landmarks[name] = position
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw background image
        if self.pixmap:
            painter.drawPixmap(self.rect(), self.pixmap)

        # Draw connections
        pen = QPen(QColor(0, 255, 255, 200), 3) # Cyan, semi-transparent
        painter.setPen(pen)
        for start_lm, end_lm in POSE_CONNECTIONS:
            if start_lm in self.placed_landmarks and end_lm in self.placed_landmarks:
                p1 = self.placed_landmarks[start_lm]
                p2 = self.placed_landmarks[end_lm]
                painter.drawLine(p1, p2)
        
        # Draw joints
        for name, pos in self.placed_landmarks.items():
            painter.setPen(QColor(255, 255, 0)) # Yellow border
            painter.setBrush(QColor(255, 165, 0, 200)) # Orange fill
            painter.drawEllipse(pos, 8, 8)

    def mousePressEvent(self, event):
        if not self.pixmap: return
        for name, pos in self.placed_landmarks.items():
            if (event.pos() - pos).manhattanLength() < 10:
                self.dragged_landmark = name
                return
        self.dragged_landmark = None

    def mouseMoveEvent(self, event):
        if self.dragged_landmark:
            self.placed_landmarks[self.dragged_landmark] = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self.dragged_landmark = None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coordinate Mapper Tool")
        self.setGeometry(200, 200, 1000, 700)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        # Left Panel: Controls
        self.controls_layout = QVBoxLayout()
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        
        self.landmark_list = QListWidget()
        for name in LANDMARK_NAMES:
            self.landmark_list.addItem(name)
        
        self.instructions = QLabel("1. Load an image.\n2. Select a joint from the list.\n3. Click on the image to place it.\n4. Drag points to adjust.")
        self.instructions.setWordWrap(True)

        self.generate_button = QPushButton("Generate Code")
        self.generate_button.clicked.connect(self.generate_code)

        self.code_output = QTextEdit()
        self.code_output.setReadOnly(True)
        self.code_output.setFont(QFont("Courier", 10))

        self.controls_layout.addWidget(QLabel("<h2>Controls</h2>"))
        self.controls_layout.addWidget(self.load_image_button)
        self.controls_layout.addWidget(self.landmark_list)
        self.controls_layout.addWidget(self.instructions)
        self.controls_layout.addWidget(self.generate_button)
        self.controls_layout.addWidget(QLabel("<b>Copy this code into visual_calibrator.py:</b>"))
        self.controls_layout.addWidget(self.code_output)

        # Right Panel: Canvas
        self.canvas = ImageCanvas()
        self.canvas.mousePressEvent = self.canvas_mouse_press # Override

        self.layout.addLayout(self.controls_layout, 1)
        self.layout.addWidget(self.canvas, 3)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.canvas.set_image(file_path)

    def canvas_mouse_press(self, event):
        # First, check for dragging
        for name, pos in self.canvas.placed_landmarks.items():
            if (event.pos() - pos).manhattanLength() < 10:
                self.canvas.dragged_landmark = name
                return
        self.canvas.dragged_landmark = None
        
        # If not dragging, check for placing
        current_item = self.landmark_list.currentItem()
        if current_item and self.canvas.pixmap:
            self.canvas.add_landmark(current_item.text(), event.pos())

    def generate_code(self):
        w, h = self.canvas.width(), self.canvas.height()
        if not w or not h or not self.canvas.placed_landmarks:
            self.code_output.setText("# No landmarks placed yet.")
            return

        normalized_landmarks = {}
        for name, pos in self.canvas.placed_landmarks.items():
            norm_x = round(pos.x() / w, 4)
            norm_y = round(pos.y() / h, 4)
            normalized_landmarks[name] = (norm_x, norm_y)

        code_str = "self.landmark_positions = {\n"
        for name, coords in sorted(normalized_landmarks.items()):
            code_str += f"    '{name}': {coords},\n"
        code_str += "}"
        
        self.code_output.setText(code_str)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())



            self.landmark_positions = {
    'LEFT_ANKLE': (0.306, 0.9047),
    'LEFT_EAR': (0.3046, 0.1173),
    'LEFT_ELBOW': (0.3893, 0.3651),
    'LEFT_EYE': (0.2814, 0.0704),
    'LEFT_EYE_INNER': (0.2678, 0.0689),
    'LEFT_EYE_OUTER': (0.2923, 0.0704),
    'LEFT_FOOT_INDEX': (0.3429, 0.9648),
    'LEFT_HEEL': (0.2883, 0.9575),
    'LEFT_HIP': (0.3046, 0.5308),
    'LEFT_INDEX': (0.4713, 0.2522),
    'LEFT_KNEE': (0.3333, 0.7111),
    'LEFT_PINKY': (0.4863, 0.3167),
    'LEFT_SHOULDER': (0.332, 0.2669),
    'LEFT_THUMB': (0.4426, 0.2683),
    'LEFT_WRIST': (0.444, 0.3211),
    'MOUTH_LEFT': (0.2678, 0.1613),
    'MOUTH_RIGHT': (0.2336, 0.1628),
    'NOSE': (0.25, 0.1217),
    'RIGHT_ANKLE': (0.1954, 0.9091),
    'RIGHT_EAR': (0.194, 0.1202),
    'RIGHT_ELBOW': (0.112, 0.3651),
    'RIGHT_EYE': (0.2186, 0.0718),
    'RIGHT_EYE_INNER': (0.235, 0.0733),
    'RIGHT_EYE_OUTER': (0.2077, 0.0733),
    'RIGHT_FOOT_INDEX': (0.153, 0.9633),
    'RIGHT_HEEL': (0.2117, 0.9575),
    'RIGHT_HIP': (0.194, 0.5308),
    'RIGHT_INDEX': (0.0287, 0.2507),
    'RIGHT_KNEE': (0.168, 0.7111),
    'RIGHT_PINKY': (0.0109, 0.3196),
    'RIGHT_SHOULDER': (0.1667, 0.2669),
    'RIGHT_THUMB': (0.056, 0.2683),
    'RIGHT_WRIST': (0.0546, 0.3196),
}