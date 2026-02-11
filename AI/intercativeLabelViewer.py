import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
import glob
from PIL import Image
import io
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button, RadioButtons
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import os

# ------------------------------
# Paths
# ------------------------------
image_dir = "/home/samuel/test/MasterThesis/Orthomosaics/small/translated/translated_250x_250y/processed_output/image_tiles"
label_dir = "/home/samuel/test/MasterThesis/Orthomosaics/small/translated/translated_250x_250y/labels_txt"
nir_output_dir = "/home/samuel/test/MasterThesis/Orthomosaics/small/translated/translated_250x_250y/processed_output/nir"

# Create output directory if it doesn't exist
Path(nir_output_dir).mkdir(parents=True, exist_ok=True)

# Get all .tif files in the image directory
tif_files = sorted(glob.glob(f"{image_dir}/*.tif"))

class InteractiveBBoxViewer:
    def __init__(self, tif_files, image_dir, label_dir, nir_output_dir):
        self.tif_files = tif_files
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.nir_output_dir = nir_output_dir
        
        self.current_idx = 0
        self.bboxes = []  # List of (polygon_patch, bbox_data, selected)
        self.deleted_bboxes = []  # Track deleted bboxes per tile
        self.img = None
        self.h = 0
        self.w = 0
        
        # Drawing state
        self.mode = 'select'  # 'select', 'draw_rect', or 'draw_rotated'
        self.drawing_points = []
        self.drawing_markers = []
        self.temp_shape = None
        self.new_class = 0
        self.rotation_angle = 0  # For rotated rectangles
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        plt.subplots_adjust(bottom=0.15, right=0.85)
        
        # Create mode selector
        ax_radio = plt.axes([0.87, 0.65, 0.12, 0.2])
        self.radio = RadioButtons(ax_radio, ('Select Mode', 'Draw Rectangle', 'Draw Rotated Box'), active=0)
        self.radio.on_clicked(self.change_mode)
        
        # Create class input for drawing
        self.class_text = self.fig.text(0.87, 0.60, 'Class ID: 0', fontsize=10)
        
        # Create rotation angle input (only for rotated mode)
        self.angle_text = self.fig.text(0.87, 0.55, 'Angle: 0°', fontsize=10)
        
        # Create buttons
        ax_prev = plt.axes([0.05, 0.05, 0.08, 0.05])
        ax_next = plt.axes([0.14, 0.05, 0.08, 0.05])
        ax_delete = plt.axes([0.25, 0.05, 0.1, 0.05])
        ax_save = plt.axes([0.36, 0.05, 0.1, 0.05])
        ax_reset = plt.axes([0.47, 0.05, 0.08, 0.05])
        ax_cancel = plt.axes([0.56, 0.05, 0.08, 0.05])

        ax_delete_entry = plt.axes([0.85, 0.05, 0.1, 0.05])

        ax_class_up = plt.axes([0.87, 0.50, 0.05, 0.04])
        ax_class_down = plt.axes([0.93, 0.50, 0.05, 0.04])
        ax_angle_up = plt.axes([0.87, 0.45, 0.05, 0.04])
        ax_angle_down = plt.axes([0.93, 0.45, 0.05, 0.04])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_delete = Button(ax_delete, 'Delete Selected')
        self.btn_save = Button(ax_save, 'Save Changes')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_cancel = Button(ax_cancel, 'Cancel')

        self.btn_delete_entry = Button(ax_delete_entry, 'Delete Entry')

        self.btn_class_up = Button(ax_class_up, '+')
        self.btn_class_down = Button(ax_class_down, '-')
        self.btn_angle_up = Button(ax_angle_up, '+')
        self.btn_angle_down = Button(ax_angle_down, '-')
        
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        self.btn_delete.on_clicked(self.delete_selected)
        self.btn_save.on_clicked(self.save_labels)
        self.btn_reset.on_clicked(self.reset_image)
        self.btn_cancel.on_clicked(self.cancel_drawing)
        self.btn_class_up.on_clicked(self.increment_class)
        self.btn_class_down.on_clicked(self.decrement_class)
        self.btn_angle_up.on_clicked(self.increment_angle)
        self.btn_angle_down.on_clicked(self.decrement_angle)
        self.btn_delete_entry.on_clicked(self.delete_selected_entry)
        
        # Initially hide angle controls
        self.btn_angle_up.ax.set_visible(False)
        self.btn_angle_down.ax.set_visible(False)
        self.angle_text.set_visible(False)
        
        # Initially disable drawing buttons
        self.btn_cancel.ax.set_visible(False)
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Load first image
        self.load_image(0)
        
        plt.show()

    def delete_selected_entry(self, event):
        os.remove(f"{self.label_dir}/{Path(self.tif_files[self.current_idx]).stem}.txt")
        os.remove(f"{self.image_dir}/{Path(self.tif_files[self.current_idx]).stem}.tif")
        os.remove(f"{self.nir_output_dir}/{Path(self.tif_files[self.current_idx]).stem}_NIR.png")

        # load next image after deletion
        if self.current_idx >= len(self.tif_files) - 1:
            self.load_image(0)
        else:
            self.load_image(self.current_idx + 1)
        
    
    def change_mode(self, label):
        """Switch between select and draw modes"""
        if label == 'Select Mode':
            self.mode = 'select'
            self.cancel_drawing(None)
            self.btn_cancel.ax.set_visible(False)
            self.btn_angle_up.ax.set_visible(False)
            self.btn_angle_down.ax.set_visible(False)
            self.angle_text.set_visible(False)
        elif label == 'Draw Rectangle':
            self.mode = 'draw_rect'
            self.btn_cancel.ax.set_visible(True)
            self.btn_angle_up.ax.set_visible(False)
            self.btn_angle_down.ax.set_visible(False)
            self.angle_text.set_visible(False)
        else:  # Draw Rotated Box
            self.mode = 'draw_rotated'
            self.btn_cancel.ax.set_visible(True)
            self.btn_angle_up.ax.set_visible(True)
            self.btn_angle_down.ax.set_visible(True)
            self.angle_text.set_visible(True)
        self.update_title()
        self.fig.canvas.draw()
    
    def increment_class(self, event):
        """Increment class ID"""
        self.new_class += 1
        self.class_text.set_text(f'Class ID: {self.new_class}')
        self.fig.canvas.draw()
    
    def decrement_class(self, event):
        """Decrement class ID"""
        self.new_class = max(0, self.new_class - 1)
        self.class_text.set_text(f'Class ID: {self.new_class}')
        self.fig.canvas.draw()
    
    def increment_angle(self, event):
        """Increment rotation angle"""
        self.rotation_angle = (self.rotation_angle + 15) % 360
        self.angle_text.set_text(f'Angle: {self.rotation_angle}°')
        self.fig.canvas.draw()
    
    def decrement_angle(self, event):
        """Decrement rotation angle"""
        self.rotation_angle = (self.rotation_angle - 15) % 360
        self.angle_text.set_text(f'Angle: {self.rotation_angle}°')
        self.fig.canvas.draw()
    
    def update_title(self):
        """Update the title based on current mode"""
        tile_name = Path(self.tif_files[self.current_idx]).stem
        if self.mode == 'select':
            title = f"{tile_name} - SELECT MODE: Click on bbox to select/deselect"
        elif self.mode == 'draw_rect':
            title = f"{tile_name} - DRAW RECTANGLE: Click and drag to draw axis-aligned rectangle"
        else:
            title = f"{tile_name} - DRAW ROTATED BOX: Click and drag, adjust angle with +/- buttons"
        self.ax.set_title(title, fontsize=12, fontweight='bold')
    
    def rotate_rectangle(self, center, width, height, angle_deg):
        """Create rotated rectangle points"""
        angle = np.radians(angle_deg)
        
        # Rectangle corners relative to center
        corners = np.array([
            [-width/2, -height/2],
            [width/2, -height/2],
            [width/2, height/2],
            [-width/2, height/2]
        ])
        
        # Rotation matrix
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # Rotate corners
        rotated = corners @ R.T
        
        # Translate to center
        rotated += center
        
        return rotated
    
    def load_image(self, idx):
        """Load and display image with bounding boxes"""
        self.current_idx = idx
        image_path = self.tif_files[idx]
        tile_name = Path(image_path).stem
        label_path = f"{self.label_dir}/{tile_name}.txt"
        nir_png_path = f"{self.nir_output_dir}/{tile_name}_NIR.png"
        
        print(f"Loading {tile_name}... ({idx+1}/{len(self.tif_files)})")
        
        # Save tif as png
        try:
            with rasterio.open(image_path) as src:
                img = src.read([7])
                img = np.transpose(img, (1, 2, 0))
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(nir_png_path, img)
        except Exception as e:
            print(f"Error reading {image_path}: {e}")
            return
        
        # Load image
        self.img = cv2.imread(nir_png_path)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.h, self.w = self.img.shape[:2]
        
        # Clear previous plot
        self.ax.clear()
        self.ax.imshow(self.img)
        self.update_title()
        self.ax.axis("off")
        
        # Reset state
        self.bboxes = []
        self.deleted_bboxes = []
        self.cancel_drawing(None)
        
        # Load and draw bounding boxes
        if Path(label_path).exists():
            with open(label_path, "r") as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    
                    cls = int(parts[0])
                    x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = map(float, parts[1:9])
                    
                    # Convert normalized → pixel coordinates
                    pts = np.array([
                        [x_1 * self.w, y_1 * self.h],
                        [x_2 * self.w, y_2 * self.h],
                        [x_3 * self.w, y_3 * self.h],
                        [x_4 * self.w, y_4 * self.h]
                    ])
                    
                    # Create polygon patch
                    poly = Polygon(pts, closed=True, fill=False, 
                                 edgecolor='green', linewidth=2, picker=True)
                    self.ax.add_patch(poly)
                    
                    # Store bbox data
                    bbox_data = {
                        'line': line.strip(),
                        'class': cls,
                        'coords': (x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4),
                        'pts': pts,
                        'line_idx': line_idx
                    }
                    self.bboxes.append([poly, bbox_data, False])  # [patch, data, selected]
                    
        else:
            print(f"Warning: Label file not found: {label_path}")
        
        self.fig.canvas.draw()
    
    def on_click(self, event):
        """Handle click events"""
        if event.inaxes != self.ax:
            return
        
        if self.mode == 'select':
            # Select mode: click on bbox to select/deselect
            for i, (poly, bbox_data, selected) in enumerate(self.bboxes):
                if poly.contains_point((event.x, event.y)):
                    # Toggle selection
                    self.bboxes[i][2] = not selected
                    
                    # Update appearance
                    if self.bboxes[i][2]:  # Now selected
                        poly.set_edgecolor('red')
                        poly.set_linewidth(3)
                    else:  # Now deselected
                        poly.set_edgecolor('green')
                        poly.set_linewidth(2)
                    
                    self.fig.canvas.draw()
                    break
        
        elif self.mode in ['draw_rect', 'draw_rotated']:
            # Draw mode: start drawing rectangle
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                if len(self.drawing_points) == 0:
                    # First click - store starting point
                    self.drawing_points.append([x, y])
                    marker, = self.ax.plot(x, y, 'ro', markersize=8)
                    self.drawing_markers.append(marker)
                    print(f"Starting point: ({x:.1f}, {y:.1f})")
                    self.fig.canvas.draw()
                elif len(self.drawing_points) == 1:
                    # Second click - finish rectangle
                    self.drawing_points.append([x, y])
                    self.finish_rectangle()
    
    def on_motion(self, event):
        """Handle mouse motion for drawing preview"""
        if self.mode not in ['draw_rect', 'draw_rotated'] or len(self.drawing_points) != 1:
            return
        
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        
        # Remove previous preview
        if self.temp_shape:
            self.temp_shape.remove()
        
        # Calculate rectangle from start point to current position
        x1, y1 = self.drawing_points[0]
        x2, y2 = event.xdata, event.ydata
        
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        if self.mode == 'draw_rect':
            # Axis-aligned rectangle
            pts = np.array([
                [min(x1, x2), min(y1, y2)],
                [max(x1, x2), min(y1, y2)],
                [max(x1, x2), max(y1, y2)],
                [min(x1, x2), max(y1, y2)]
            ])
        else:  # draw_rotated
            # Rotated rectangle
            pts = self.rotate_rectangle(center, width, height, self.rotation_angle)
        
        self.temp_shape = Polygon(pts, closed=True, fill=False,
                                 edgecolor='blue', linewidth=2, linestyle='--')
        self.ax.add_patch(self.temp_shape)
        self.fig.canvas.draw()
    
    def finish_rectangle(self):
        """Finish drawing the rectangle and add it as a bbox"""
        if len(self.drawing_points) != 2:
            return
        
        x1, y1 = self.drawing_points[0]
        x2, y2 = self.drawing_points[1]
        
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        if width < 5 or height < 5:
            print("Rectangle too small, cancelled")
            self.cancel_drawing(None)
            return
        
        if self.mode == 'draw_rect':
            # Axis-aligned rectangle (YOLO OBB format with 0 rotation)
            pts = np.array([
                [min(x1, x2), min(y1, y2)],
                [max(x1, x2), min(y1, y2)],
                [max(x1, x2), max(y1, y2)],
                [min(x1, x2), max(y1, y2)]
            ])
        else:  # draw_rotated
            # Rotated rectangle
            pts = self.rotate_rectangle(center, width, height, self.rotation_angle)
        
        # Normalize coordinates for YOLO format
        normalized_coords = []
        for pt in pts:
            normalized_coords.extend([pt[0] / self.w, pt[1] / self.h])
        
        # Create the polygon patch
        poly = Polygon(pts, closed=True, fill=False,
                      edgecolor='green', linewidth=2, picker=True)
        self.ax.add_patch(poly)
        
        # Create label line
        x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = normalized_coords
        label_line = f"{self.new_class} {x_1:.6f} {y_1:.6f} {x_2:.6f} {y_2:.6f} {x_3:.6f} {y_3:.6f} {x_4:.6f} {y_4:.6f}"
        
        # Store bbox data
        bbox_data = {
            'line': label_line,
            'class': self.new_class,
            'coords': (x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4),
            'pts': pts,
            'line_idx': -1  # New bbox, not in original file
        }
        self.bboxes.append([poly, bbox_data, False])
        
        
        angle_str = f" (angle: {self.rotation_angle}°)" if self.mode == 'draw_rotated' else ""
        print(f"Added new bounding box with class {self.new_class}{angle_str}")
        
        # Clear drawing state
        self.cancel_drawing(None)
        self.fig.canvas.draw()
    
    def cancel_drawing(self, event):
        """Cancel current drawing"""
        # Remove temporary elements
        for marker in self.drawing_markers:
            marker.remove()
        if self.temp_shape:
            self.temp_shape.remove()
            self.temp_shape = None
        
        self.drawing_points = []
        self.drawing_markers = []
        
        if event is not None:  # Only draw if called from button
            self.fig.canvas.draw()
    
    def delete_selected(self, event):
        """Delete selected bounding boxes"""
        # Find selected boxes
        to_delete = []
        for i, (poly, bbox_data, selected) in enumerate(self.bboxes):
            if selected:
                to_delete.append(i)
                poly.remove()
                if bbox_data['line_idx'] != -1:  # Original bbox
                    self.deleted_bboxes.append(bbox_data['line_idx'])
        
        # Remove from list (reverse order to maintain indices)
        for i in reversed(to_delete):
            del self.bboxes[i]
        
        print(f"Deleted {len(to_delete)} bounding box(es)")
        self.fig.canvas.draw()
    
    def save_labels(self, event):
        """Save modified labels to file"""
        image_path = self.tif_files[self.current_idx]
        tile_name = Path(image_path).stem
        label_path = f"{self.label_dir}/{tile_name}.txt"
        
        # Collect all remaining labels
        all_labels = []
        
        # Add original labels that weren't deleted
        if Path(label_path).exists():
            with open(label_path, "r") as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                if i not in self.deleted_bboxes:
                    all_labels.append(line.strip())
        
        # Add new labels
        for poly, bbox_data, selected in self.bboxes:
            if bbox_data['line_idx'] == -1:  # New bbox
                all_labels.append(bbox_data['line'])
        
        # Save to file
        with open(label_path, "w") as f:
            for label in all_labels:
                f.write(label + "\n")
        
        print(f"Saved changes to {label_path}")
        print(f"Total labels: {len(all_labels)}")
        print(f"Removed: {len(self.deleted_bboxes)}, Added: {sum(1 for _, bd, _ in self.bboxes if bd['line_idx'] == -1)}")
    
    def reset_image(self, event):
        """Reload current image"""
        self.load_image(self.current_idx)
    
    def prev_image(self, event):
        """Go to previous image"""
        if self.current_idx > 0:
            self.load_image(self.current_idx - 1)
    
    def next_image(self, event):
        """Go to next image"""
        if self.current_idx < len(self.tif_files) - 1:
            self.load_image(self.current_idx + 1)

# Run the interactive viewer
viewer = InteractiveBBoxViewer(tif_files, image_dir, label_dir, nir_output_dir)