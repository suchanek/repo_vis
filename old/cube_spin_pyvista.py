import sys
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import QTimer
import numpy as np

class CubeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spinning Cube")
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Initialize PyVista plotter with pyvistaqt
        self.plotter = BackgroundPlotter(
            show=False,
            title="Spinning Cube",
            window_size=(800, 400),
            line_smoothing=True,
            point_smoothing=True,
            polygon_smoothing=True
        )
        layout.addWidget(self.plotter)

        # Create cube
        self.cube = pv.Cube()
        self.cube_actor = self.plotter.add_mesh(
            self.cube,
            color='red',
            show_edges=True,
            edge_color='black'
        )

        # Camera setup
        self.plotter.camera_position = 'iso'
        self.plotter.camera.distance = 5.0

        # Animation variables
        self.angle = 0
        self.spinning = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_spin)

        # Buttons
        spin_five_button = QPushButton("Spin for 5 Seconds")
        spin_five_button.clicked.connect(self.spin_five_seconds)
        layout.addWidget(spin_five_button)

        start_button = QPushButton("Start Spin")
        start_button.clicked.connect(self.start_spin)
        layout.addWidget(start_button)

        stop_button = QPushButton("Stop Spin")
        stop_button.clicked.connect(self.stop_spin)
        layout.addWidget(stop_button)

    def update_spin(self):
        if self.spinning:
            self.angle += 1.0
            if self.angle >= 360:
                self.angle -= 360

            # Rotate cube around Z-axis
            self.cube_actor.RotateZ(1.0)
            self.plotter.update()

    def spin_five_seconds(self):
        self.spinning = True
        self.timer.start(16)  # ~60 FPS
        QTimer.singleShot(5000, self.stop_spin)

    def start_spin(self):
        self.spinning = True
        self.timer.start(16)  # ~60 FPS

    def stop_spin(self):
        self.spinning = False
        self.timer.stop()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pv.set_jupyter_backend(None)  # Ensure PyVista doesn't try to use Jupyter backend
    window = CubeWindow()
    window.show()
    sys.exit(app.exec_())