import PIL
from PIL import ImageQt
import cv2
from PyQt6 import QtWidgets, QtGui
from PyQt6.QtWidgets import QFileDialog, QLabel, QVBoxLayout, QPushButton, QHBoxLayout, QMessageBox,QLineEdit
from PyQt6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtGui import QPixmap, QPainter, QImage, QPolygonF
from PyQt6.QtCore import Qt, pyqtSignal
import sys
import os
import rasterio
from pathlib import Path
import numpy as np

class InteractiveView(QtWidgets.QGraphicsView):
    
    polygonFinished = pyqtSignal(list)   # <-- NEW SIGNAL

    def __init__(self, scene):
        super().__init__(scene)

        self.points = []
        self.temp_items = []
        self.cropped_pixmap = None

        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )

    # ---- ZOOM ----
    def wheelEvent(self, event):
        zoom = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom, zoom)
        else:
            self.scale(1/zoom, 1/zoom)

    # ---- DRAW POINTS WITH SHIFT + CLICK ----
    def mousePressEvent(self, event):

        if event.button() == Qt.MouseButton.LeftButton and \
           event.modifiers() & Qt.KeyboardModifier.ShiftModifier:

            scene_pos = self.mapToScene(event.position().toPoint())
            self.points.append(scene_pos)

            # draw marker
            dot = self.scene().addEllipse(
                scene_pos.x()-2, scene_pos.y()-2, 4, 4
            )
            self.temp_items.append(dot)

            pen = QtGui.QPen(Qt.GlobalColor.red)
            pen.setWidth(2)

            # draw line to previous point
            if len(self.points) > 1:
                p1 = self.points[-2]
                p2 = self.points[-1]
                line = self.scene().addLine(p1.x(), p1.y(), p2.x(), p2.y(), pen)
                self.temp_items.append(line)

            return  # <-- VERY IMPORTANT

        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Return and len(self.points) >= 3:

            # Emit points to ImageViewer
            self.polygonFinished.emit(self.points.copy())   
            # Clear temporary drawing
            for item in self.temp_items:
                self.scene().removeItem(item)

            self.points.clear()
            self.temp_items.clear()

            return

        super().keyPressEvent(event)

class ImageViewer(QtWidgets.QWidget):

    def __init__(self, image_folder_path : Path = None, label_folder_path : Path = None):
        super().__init__()
    
        self.image_folder_path = image_folder_path
        self.label_folder_path = label_folder_path
        self.images = []
        self.label_files = []
        self.current_image_index = 0
        self.labels_shown = False
        self.cropped_images = []

        # For drawing polygon
        self.points = []
        self.temp_lines = []

        self.supported_image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif"]

        self.image_count = 0
        self.label_file_count = 0

        self.image_view_layout = QVBoxLayout()

        self.image_count_label = QLabel(f"Image Count: {self.image_count}", alignment=Qt.AlignmentFlag.AlignCenter)
        self.image_count_label.setMaximumHeight(30)
        self.label_file_count_label = QLabel(f"Label File Count: {self.label_file_count}", alignment=Qt.AlignmentFlag.AlignCenter)
        self.label_file_count_label.setMaximumHeight(30)

        self.image_name_label = QLineEdit(f"Image Name: ", alignment=Qt.AlignmentFlag.AlignCenter)
        self.image_name_label.setMaximumHeight(30)
        self.image_name_layout = QHBoxLayout()
        self.image_name_layout.addWidget(self.image_name_label)
    

        self.main_layout = QVBoxLayout()

        self.scene = QGraphicsScene()
        self.view = InteractiveView(self.scene)
        self.view.polygonFinished.connect(self.crop_polygon)

        self.main_layout.addLayout(self.image_name_layout)

        self.image_view_layout.addWidget(self.view)

        self.cropped_pixmap = None  # Store cropped pixmap for pasting

        self.image_label_count_layout = QHBoxLayout()
        self.image_label_count_layout.addWidget(self.label_file_count_label)
        self.image_label_count_layout.addWidget(self.image_count_label)

        self.main_layout.addLayout(self.image_label_count_layout)
        self.main_layout.addLayout(self.image_view_layout)

        self.open_image_button = QPushButton("Open Image Folder")
        self.open_image_button.clicked.connect(self.open_image_folder)
            
        self.open_label_button = QPushButton("Open Label Folder")
        self.open_label_button.clicked.connect(self.open_label_folder)

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.open_image_button)
        self.button_layout.addWidget(self.open_label_button)
        self.main_layout.addLayout(self.button_layout)

        self.show_label_images_button_layout = QHBoxLayout()
        self.show_label_images_button = QPushButton("Show Labels on Images")
        self.show_label_images_button_layout.addWidget(self.show_label_images_button)
        self.show_label_images_button.clicked.connect(self.show_labels_on_images)
        self.main_layout.addLayout(self.show_label_images_button_layout)

         # Navigation buttons

        self.left_parse_images_button = QPushButton("<<")
        self.left_parse_images_button.clicked.connect(self.iterate_images_left)
        self.right_parse_images_button = QPushButton(">>")
        self.right_parse_images_button.clicked.connect(self.iterate_images_right)

        self.image_navigation_layout = QHBoxLayout()
        self.image_navigation_layout.addWidget(self.left_parse_images_button)
        self.image_navigation_layout.addWidget(self.right_parse_images_button)
        self.main_layout.addLayout(self.image_navigation_layout)

        self.setLayout(self.main_layout)
        self.setWindowTitle("Image and Label Viewer")
        self.resize(800, 600)


    def crop_polygon(self, points):
        # Convert points to polygon
        poly = QtGui.QPolygonF(points)
        path = QtGui.QPainterPath()
        path.addPolygon(poly)

        # Load current pixmap as QImage
        img = QtGui.QPixmap(self.images[self.current_image_index]).toImage()

        # Create writable transparent image
        cropped = QtGui.QImage(img.width(), img.height(), QtGui.QImage.Format.Format_ARGB32)
        cropped.fill(Qt.GlobalColor.transparent)



        # Convert to QPixmap
        self.cropped_pixmap = QtGui.QPixmap.fromImage(cropped)
        self.cropped_pixmap.save("cropped_polygon.png")  # Save for verification
    

    def show_labels_on_images(self, keep: bool = False) -> None:
        
        if not self.image_folder_path or not self.label_folder_path:
            QMessageBox.warning(self, "Folders Not Selected", "Please select both image and label folders.")
            return

        if self.image_count == 0 or self.label_file_count == 0:
            QMessageBox.warning(self, "No Images or Labels", "No images or labels found in the selected folders.")
            return
        
        if self.labels_shown and not keep:
            # Reload original image without labels
            image_path = self.image_folder_path / self.images[self.current_image_index]
            pixmap = QPixmap(str(image_path))
            self.pix_item.setPixmap(pixmap)
            self.show_label_images_button.setText("Show Labels on Images")
            self.labels_shown = False
            return

        # Load current image
        image_path = self.image_folder_path / self.images[self.current_image_index]
        pixmap = QPixmap(str(image_path))
        painter = QtGui.QPainter(pixmap)
        pen = QtGui.QPen(Qt.GlobalColor.white)
        pen.setWidth(2)
        painter.setPen(pen)

        self.show_label_images_button.setText("Stop Showing Labels")

        # Load corresponding label file
        label_file_name = os.path.splitext(self.images[self.current_image_index])[0] + ".txt"
        label_file_path = self.label_folder_path / label_file_name

        if not label_file_path.exists():
            QMessageBox.information(self, "Label File Not Found", f"No label file found for {self.images[self.current_image_index]}.")
            painter.end()
            return

        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, width, height = map(float, parts)
                img_width = pixmap.width()
                img_height = pixmap.height()

                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height

                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                painter.drawRect(x1, y1, int(width), int(height))

        painter.end()

        self.pix_item.setPixmap(pixmap)

        self.labels_shown = True

    def open_image_folder(self) -> None:
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder_path:
            # check if files are images
            files = os.listdir(folder_path)
            
            self.images = [f for f in files if os.path.splitext(f)[1].lower() in self.supported_image_extensions]

            self.image_folder_path = Path(folder_path)
            self.image_count = len(self.images)
            self.update_image_count_display()

        if self.image_count == 0:
            QMessageBox.warning(self, "No Images Found", "The selected folder does not contain any supported image files.")
            return

        self.open_image()
        self.current_image_index = 0

    def open_image(self) -> None:
        image_path = self.image_folder_path / self.images[self.current_image_index]
        pixmap = QPixmap(str(image_path))

        self.scene.clear()
        self.pix_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pix_item)
        self.view.fitInView(self.pix_item, Qt.AspectRatioMode.KeepAspectRatio)

        self.image_name_label.setText(f"Image Name: {self.images[self.current_image_index]}")

    def open_label_folder(self) -> None:
        folder_path = QFileDialog.getExistingDirectory(self, "Select Label Folder")
        if folder_path:
            self.label_folder_path = Path(folder_path)

            files = os.listdir(folder_path)
            files = [f for f in files if os.path.splitext(f)[1].lower() == ".txt"]

            self.label_file_count = len(files)
            self.update_label_file_count_display()

    def update_image_count_display(self) -> None:
        self.image_count_label.setText(f"Image Count: {self.image_count}")

    def update_label_file_count_display(self) -> None:
        self.label_file_count_label.setText(f"Label File Count: {self.label_file_count}")

    def iterate_images_left(self) -> None:
        if self.image_count == 0:
            return
        
        if self.current_image_index == 0:
            self.current_image_index = self.image_count - 1
        else:
            self.current_image_index -= 1

        self.open_image()
        if self.labels_shown:
            self.show_labels_on_images(True)

    def iterate_images_right(self) -> None:
        if self.image_count == 0:
            return
        
        if self.current_image_index == self.image_count - 1:
            self.current_image_index = 0
        else:
            self.current_image_index += 1

        self.open_image()
        if self.labels_shown:
            self.show_labels_on_images(True)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec())