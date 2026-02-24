import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
import glob
from PIL import Image
import io
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button, RadioButtons, Slider
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import os

# ------------------------------
# Paths
# ------------------------------
image_dir = "/home/samuel/test/MasterThesis/Orthomosaics/mid/original/original/processed_output/image_tiles"
label_dir = "/home/samuel/test/MasterThesis/Orthomosaics/mid/original/original/labels"
nir_output_dir = "/home/samuel/test/MasterThesis/Orthomosaics/mid/original/original/processed_output/nir"
predictions_dir = "/home/samuel/MasterThesis/runs/obb/inference/val_predictions/labels"
COMBINATION = "NIR_RED_NGRDI"

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
        self.pred_bboxes = []  # List of predicted bboxes (polygon_patch, bbox_data, selected)
        self.img = None
        self.h = 0
        self.w = 0
        self.offset = 20  # pixels for label text offset from bbox  
        
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

        self.confidence_threshold_slider = plt.axes([0.87, 0.40, 0.10, 0.03])
        self.confidence_slider = Slider(self.confidence_threshold_slider, 'Conf Thres', 0.0, 1.0, valinit=0.25, valstep=0.05)
        self.confidence_slider.on_changed(self.update_prediction_visibility)
        
        # Create class input for drawing
        self.class_text = self.fig.text(0.87, 0.60, 'Class ID: 0', fontsize=10)

        self.legend_text = self.fig.text(0.05, 0.12, '', fontsize=10, fontweight='bold')

        # Create legend showing number of images and current image number
        self.legend_text = self.fig.text(0.05, 0.12, '', fontsize=10, fontweight='bold')
        
        # Create rotation angle input (only for rotated mode)
        self.angle_text = self.fig.text(0.87, 0.55, 'Angle: 0°', fontsize=10)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        # Disable conflicting defaults
        self.fig.canvas.mpl_disconnect(
            self.fig.canvas.manager.key_press_handler_id
        )
        
        # Create buttons
        ax_prev         = plt.axes([0.05, 0.05, 0.08, 0.05])
        ax_next         = plt.axes([0.14, 0.05, 0.08, 0.05])
        ax_delete       = plt.axes([0.23, 0.05, 0.10, 0.05])
        ax_accept_pred  = plt.axes([0.34, 0.05, 0.10, 0.05])  # ← NEW
        ax_save         = plt.axes([0.45, 0.05, 0.08, 0.05])
        ax_reset        = plt.axes([0.54, 0.05, 0.07, 0.05])
        ax_cancel       = plt.axes([0.62, 0.05, 0.07, 0.05])
        ax_accept_all_pred    = plt.axes([0.70, 0.05, 0.07, 0.05])  # ← NEW

        ax_delete_entry = plt.axes([0.85, 0.05, 0.10, 0.05])

        ax_class_up   = plt.axes([0.87, 0.50, 0.05, 0.04])
        ax_class_down = plt.axes([0.93, 0.50, 0.05, 0.04])
        ax_angle_up   = plt.axes([0.87, 0.45, 0.05, 0.04])
        ax_angle_down = plt.axes([0.93, 0.45, 0.05, 0.04])
        
        self.btn_prev        = Button(ax_prev,        'Previous')
        self.btn_next        = Button(ax_next,        'Next')
        self.btn_delete      = Button(ax_delete,      'Delete GT')
        self.btn_accept_pred = Button(ax_accept_pred, 'Accept Pred')  # ← NEW
        self.btn_save        = Button(ax_save,        'Save')
        self.btn_reset       = Button(ax_reset,       'Reset')
        self.btn_cancel      = Button(ax_cancel,      'Cancel')
        self.btn_hide_show_pred = Button(plt.axes([0.85, 0.15, 0.10, 0.05]), 'Toggle Pred')  # ← NEW
        self.btn_accept_all_pred = Button(ax_accept_all_pred, 'Accept All Pred')  # ← NEW

        self.btn_delete_entry = Button(ax_delete_entry, 'Delete Entry')

        self.btn_class_up   = Button(ax_class_up,   '+')
        self.btn_class_down = Button(ax_class_down, '-')
        self.btn_angle_up   = Button(ax_angle_up,   '+')
        self.btn_angle_down = Button(ax_angle_down, '-')

        self.gt_count_text = self.fig.text(
            0.05, 0.15,
            'GT Boxes: 0',
            fontsize=10,
            color='red',
            fontweight='bold'
        )
                
        self.btn_prev.on_clicked(self.prev_image)
        self.btn_next.on_clicked(self.next_image)
        self.btn_delete.on_clicked(self.delete_selected)
        self.btn_accept_pred.on_clicked(self.accept_selected_preds)  # ← NEW
        self.btn_save.on_clicked(self.save_labels)
        self.btn_reset.on_clicked(self.reset_image)
        self.btn_cancel.on_clicked(self.cancel_drawing)
        self.btn_class_up.on_clicked(self.increment_class)
        self.btn_class_down.on_clicked(self.decrement_class)
        self.btn_angle_up.on_clicked(self.increment_angle)
        self.btn_angle_down.on_clicked(self.decrement_angle)
        self.btn_delete_entry.on_clicked(self.delete_selected_entry)
        self.btn_hide_show_pred.on_clicked(self.toggle_predictions)  # ← NEW
        self.btn_accept_all_pred.on_clicked(self.accept_all_preds)  # ← NEW

        # Color the accept button green for clarity
        self.btn_accept_pred.ax.set_facecolor('#c8f0c8')

        # Initially hide angle controls
        self.btn_angle_up.ax.set_visible(False)
        self.btn_angle_down.ax.set_visible(False)
        self.angle_text.set_visible(False)
        
        # Initially hide cancel button
        self.btn_cancel.ax.set_visible(False)
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        # Load first image
        self.load_image(0)
        
        plt.show()
    
    def update_gt_counter(self):
        self.gt_count_text.set_text(
            f"GT: {len(self.bboxes)} | Pred: {len(self.pred_bboxes)}"
        )
        self.fig.canvas.draw_idle()

    def update_prediction_visibility(self, val):
        threshold = self.confidence_slider.val

        for poly, bbox_data, _ in self.pred_bboxes:
            score = bbox_data.get('confidence', 0)
            visible = score >= threshold
            poly.set_visible(visible)
            if 'text' in bbox_data:
                bbox_data['text'].set_visible(visible)

        self.fig.canvas.draw_idle()

    def polygon_iou(self, pts1, pts2):
        from shapely.geometry import Polygon as ShapelyPolygon

        poly1 = ShapelyPolygon(pts1)
        poly2 = ShapelyPolygon(pts2)

        if not poly1.is_valid or not poly2.is_valid:
            return 0.0

        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area

        if union == 0:
            return 0.0

        return intersection / union

    def accept_all_preds(self, event):

        confidence_threshold = self.confidence_slider.val
        accepted = 0
        to_remove_pred = []
        iou_threshold = 0.3   # adjust if needed

        for i, (pred_poly, pred_data, _) in enumerate(self.pred_bboxes):

            score = pred_data.get('confidence', 0)
            if score < confidence_threshold:
                continue

            pred_pts = pred_data['pts']

            # -----------------------------------------
            # 1️⃣ Remove overlapping GT boxes
            # -----------------------------------------
            gt_to_remove = []

            for j, (gt_poly, gt_data, _) in enumerate(self.bboxes):
                gt_pts = gt_data['pts']
                iou = self.polygon_iou(pred_pts, gt_pts)

                if iou > iou_threshold:
                    gt_to_remove.append(j)

            for j in reversed(gt_to_remove):
                self.bboxes[j][0].remove()
                del self.bboxes[j]

            # -----------------------------------------
            # 2️⃣ Promote prediction to GT
            # -----------------------------------------

            to_remove_pred.append(i)

            new_poly = Polygon(
                pred_pts,
                closed=True,
                fill=False,
                edgecolor='red',
                linewidth=2,
                picker=True
            )
            self.ax.add_patch(new_poly)

            coords = pred_data['coords']
            x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = coords

            label_line = (
                f"{pred_data['class']} "
                f"{x_1:.6f} {y_1:.6f} {x_2:.6f} {y_2:.6f} "
                f"{x_3:.6f} {y_3:.6f} {x_4:.6f} {y_4:.6f}"
            )

            gt_bbox_data = {
                'line': label_line,
                'class': pred_data['class'],
                'coords': coords,
                'pts': pred_pts,
                'line_idx': -1
            }

            self.bboxes.append([new_poly, gt_bbox_data, False])
            accepted += 1

        print(f"Accepted {accepted} predictions and removed overlapping GT.")
        self.update_gt_counter()
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right' or event.key == 'd':
            self.next_image(None)
        elif event.key == 'left' or event.key == 'a':
            self.prev_image(None)
        elif event.key == 'delete' or event.key == 'x':
            self.delete_selected(None)
        elif event.key == 'ctrl+s':
            self.save_labels(None)
        elif event.key == 'escape':
            self.cancel_drawing(None)
        elif event.key == 'enter':
            self.accept_selected_preds(None)
        elif event.key == 'r':
            self.reset_image(None)
        elif event.key == 'o':
            self.deselect_all(None)
        elif event.key == 't':
            self.toggle_predictions(None)

    def deselect_all(self, event):
        for i, (poly, bbox_data, selected) in enumerate(self.bboxes):
            if selected:
                self.bboxes[i][2] = False
                poly.set_edgecolor('red')
        for i, (poly, bbox_data, selected) in enumerate(self.pred_bboxes):
            if selected:
                self.pred_bboxes[i][2] = False
                poly.set_edgecolor('white')
        self.fig.canvas.draw()

    def toggle_predictions(self, event):
        """Toggle visibility of predicted bounding boxes."""
        if not self.pred_bboxes:
            print("No prediction boxes to toggle.")
            return
        
        # Check current visibility state based on the first prediction box
        current_visibility = self.pred_bboxes[0][0].get_visible()
        new_visibility = not current_visibility
        for poly, bbox_data, _ in self.pred_bboxes:
            poly.set_visible(new_visibility)
            # Also toggle the visibility of the confidence text if it exists
            if 'text' in bbox_data:
                bbox_data['text'].set_visible(new_visibility)

        state = "shown" if new_visibility else "hidden"
        print(f"Prediction boxes are now {state}.")
        self.fig.canvas.draw()
        
    # ------------------------------------------------------------------
    # NEW: Accept selected prediction boxes as ground truth
    # ------------------------------------------------------------------
    def accept_selected_preds(self, event):
        """Promote selected prediction boxes into ground truth bboxes."""
        accepted = 0
        to_remove = []

        for i, (poly, bbox_data, selected) in enumerate(self.pred_bboxes):
            if not selected:
                continue

            # Remove the white prediction patch
            poly.remove()
            to_remove.append(i)

            # Re-draw as a red ground-truth patch
            pts = bbox_data['pts']
            new_poly = Polygon(pts, closed=True, fill=False,
                               edgecolor='red', linewidth=2, picker=True)
            self.ax.add_patch(new_poly)

            # Build a GT-style bbox_data entry.
            # Mark line_idx as -1 so save_labels treats it as a newly added box.
            coords = bbox_data['coords']
            x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = coords
            label_line = (f"{bbox_data['class']} "
                          f"{x_1:.6f} {y_1:.6f} {x_2:.6f} {y_2:.6f} "
                          f"{x_3:.6f} {y_3:.6f} {x_4:.6f} {y_4:.6f}")

            gt_bbox_data = {
                'line': label_line,
                'class': bbox_data['class'],
                'coords': coords,
                'pts': pts,
                'line_idx': -1  # Treated as a new box by save_labels
            }
            self.bboxes.append([new_poly, gt_bbox_data, False])
            accepted += 1

        # Remove accepted entries from pred list (reverse order)
        for i in reversed(to_remove):
            del self.pred_bboxes[i]

        print(f"Accepted {accepted} prediction box(es) as ground truth.")
        self.fig.canvas.draw()
        self.update_gt_counter()

    # ------------------------------------------------------------------

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
        self.new_class += 1
        self.class_text.set_text(f'Class ID: {self.new_class}')
        self.fig.canvas.draw()
    
    def decrement_class(self, event):
        self.new_class = max(0, self.new_class - 1)
        self.class_text.set_text(f'Class ID: {self.new_class}')
        self.fig.canvas.draw()
    
    def increment_angle(self, event):
        self.rotation_angle = (self.rotation_angle + 15) % 360
        self.angle_text.set_text(f'Angle: {self.rotation_angle}°')
        self.fig.canvas.draw()
    
    def decrement_angle(self, event):
        self.rotation_angle = (self.rotation_angle - 15) % 360
        self.angle_text.set_text(f'Angle: {self.rotation_angle}°')
        self.fig.canvas.draw()
    
    def update_title(self):
        tile_name = Path(self.tif_files[self.current_idx]).stem
        if self.mode == 'select':
            title = f"{tile_name} - SELECT MODE: Click GT (red) or Pred (white) to select. 'Accept Pred' promotes selected preds to GT."
        elif self.mode == 'draw_rect':
            title = f"{tile_name} - DRAW RECTANGLE: Click and drag to draw axis-aligned rectangle"
        else:
            title = f"{tile_name} - DRAW ROTATED BOX: Click and drag, adjust angle with +/- buttons"
        self.ax.set_title(title, fontsize=11, fontweight='bold')
    
    def rotate_rectangle(self, center, width, height, angle_deg):
        angle = np.radians(angle_deg)
        corners = np.array([
            [-width/2, -height/2],
            [width/2,  -height/2],
            [width/2,   height/2],
            [-width/2,  height/2]
        ])
        R = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        rotated = corners @ R.T + center
        return rotated
    
    def load_image(self, idx):
        """Load and display image with bounding boxes"""
        self.current_idx = idx
        image_path = self.tif_files[idx]
        tile_name = Path(image_path).stem
        label_path = f"{self.label_dir}/{tile_name}.txt"
        nir_png_path = f"{self.nir_output_dir}/{tile_name}_NIR.png"
        pred_found = False
        # Per-tile prediction file
        for pred_file in Path(predictions_dir).glob(f"{tile_name}_*.txt"):
            pred_label_path = str(pred_file)
            pred_found = True

        self.legend_text.set_text(f"Image {idx+1}/{len(self.tif_files)}")
        self.fig.canvas.draw()
        
        print(f"Loading {tile_name}... ({idx+1}/{len(self.tif_files)})")
        
        # Save NIR channel as PNG
        try:
            with rasterio.open(image_path) as src:
                img = src.read([7])
                img = np.transpose(img, (1, 2, 0))
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if not Path(nir_png_path).exists():
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
        self.pred_bboxes = []
        self.deleted_bboxes = []
        self.cancel_drawing(None)
        

        # Load and draw ground truth bounding boxes (red)
        if Path(label_path).exists():
            with open(label_path, "r") as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    
                    cls = int(parts[0])
                    x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = map(float, parts[1:9])
                    
                    pts = np.array([
                        [x_1 * self.w, y_1 * self.h],
                        [x_2 * self.w, y_2 * self.h],
                        [x_3 * self.w, y_3 * self.h],
                        [x_4 * self.w, y_4 * self.h]
                    ])
                    
                    poly = Polygon(pts, closed=True, fill=False,
                                   edgecolor='red', linewidth=2, picker=True)
                    self.ax.add_patch(poly)
                    
                    bbox_data = {
                        'line': line.strip(),
                        'class': cls,
                        'coords': (x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4),
                        'pts': pts,
                        'line_idx': line_idx
                    }
                    self.bboxes.append([poly, bbox_data, False])
        else:
            print(f"Note: GT label file not found: {label_path}")
            
        if pred_found:
            if  Path(pred_label_path).exists():
                print(f"Note: Prediction file found: {pred_label_path}")
                # Load and draw predicted bounding boxes (white) — per-tile file
                with open(pred_label_path, "r") as f:
                    for line_idx, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) < 10:
                            continue
                        
                        cls = int(parts[0])
                        x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = map(float, parts[1:9])
                        prediction_score = float(parts[9])  # Assuming confidence is the tenth value
                        
                        pts = np.array([
                            [x_1 * self.w, y_1 * self.h],
                            [x_2 * self.w, y_2 * self.h],
                            [x_3 * self.w, y_3 * self.h],
                            [x_4 * self.w, y_4 * self.h]
                        ])
                        
                        poly = Polygon(pts, closed=True, fill=False,
                                    edgecolor='white', linewidth=2, picker=True)
                        self.ax.add_patch(poly)
                        
                        bbox_data = {
                            'line': line.strip(),
                            'class': cls,
                            'coords': (x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4),
                            'pts': pts,
                            'line_idx': line_idx,
                            'confidence': prediction_score,
                            'text': self.ax.text(pts[2][0], pts[2][1] - self.offset, f"{cls} ({prediction_score:.2f})", color='white', fontsize=8, backgroundcolor='black')
                        }
                        self.pred_bboxes.append([poly, bbox_data, False])

                        
            else:
                print(f"Note: Prediction file not found: {pred_label_path}")
        
        self.update_gt_counter()
        self.fig.canvas.draw()
    
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        if self.mode == 'select':
            # Check GT bboxes (red)
            for i, (poly, bbox_data, selected) in enumerate(self.bboxes):
                if poly.contains_point((event.x, event.y)):
                    self.bboxes[i][2] = not selected
                    poly.set_edgecolor('blue' if self.bboxes[i][2] else 'red')
                    poly.set_linewidth(3 if self.bboxes[i][2] else 2)
                    self.fig.canvas.draw()
                    break

            # Check prediction bboxes (white)
            for i, (poly, bbox_data, selected) in enumerate(self.pred_bboxes):
                if poly.contains_point((event.x, event.y)):
                    self.pred_bboxes[i][2] = not selected
                    poly.set_edgecolor('cyan' if self.pred_bboxes[i][2] else 'white')
                    poly.set_linewidth(3 if self.pred_bboxes[i][2] else 2)
                    self.fig.canvas.draw()
                    break
        
        elif self.mode in ['draw_rect', 'draw_rotated']:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                if len(self.drawing_points) == 0:
                    self.drawing_points.append([x, y])
                    marker, = self.ax.plot(x, y, 'ro', markersize=8)
                    self.drawing_markers.append(marker)
                    print(f"Starting point: ({x:.1f}, {y:.1f})")
                    self.fig.canvas.draw()
                elif len(self.drawing_points) == 1:
                    self.drawing_points.append([x, y])
                    self.finish_rectangle()
    
    def on_motion(self, event):
        if self.mode not in ['draw_rect', 'draw_rotated'] or len(self.drawing_points) != 1:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        
        if self.temp_shape:
            self.temp_shape.remove()
        
        x1, y1 = self.drawing_points[0]
        x2, y2 = event.xdata, event.ydata
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        width  = abs(x2 - x1)
        height = abs(y2 - y1)
        
        if self.mode == 'draw_rect':
            pts = np.array([
                [min(x1, x2), min(y1, y2)],
                [max(x1, x2), min(y1, y2)],
                [max(x1, x2), max(y1, y2)],
                [min(x1, x2), max(y1, y2)]
            ])
        else:
            pts = self.rotate_rectangle(center, width, height, self.rotation_angle)
        
        self.temp_shape = Polygon(pts, closed=True, fill=False,
                                  edgecolor='blue', linewidth=2, linestyle='--')
        self.ax.add_patch(self.temp_shape)
        self.fig.canvas.draw()
    
    def finish_rectangle(self):
        if len(self.drawing_points) != 2:
            return
        
        x1, y1 = self.drawing_points[0]
        x2, y2 = self.drawing_points[1]
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        width  = abs(x2 - x1)
        height = abs(y2 - y1)
        
        if width < 5 or height < 5:
            print("Rectangle too small, cancelled")
            self.cancel_drawing(None)
            return
        
        if self.mode == 'draw_rect':
            pts = np.array([
                [min(x1, x2), min(y1, y2)],
                [max(x1, x2), min(y1, y2)],
                [max(x1, x2), max(y1, y2)],
                [min(x1, x2), max(y1, y2)]
            ])
        else:
            pts = self.rotate_rectangle(center, width, height, self.rotation_angle)
        
        normalized_coords = []
        for pt in pts:
            normalized_coords.extend([pt[0] / self.w, pt[1] / self.h])
        
        poly = Polygon(pts, closed=True, fill=False,
                       edgecolor='red', linewidth=2, picker=True)
        self.ax.add_patch(poly)
        
        x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = normalized_coords
        label_line = (f"{self.new_class} {x_1:.6f} {y_1:.6f} {x_2:.6f} {y_2:.6f} "
                      f"{x_3:.6f} {y_3:.6f} {x_4:.6f} {y_4:.6f}")
        
        bbox_data = {
            'line': label_line,
            'class': self.new_class,
            'coords': (x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4),
            'pts': pts,
            'line_idx': -1
        }
        self.bboxes.append([poly, bbox_data, False])
        
        angle_str = f" (angle: {self.rotation_angle}°)" if self.mode == 'draw_rotated' else ""
        print(f"Added new bounding box with class {self.new_class}{angle_str}")
        
        self.cancel_drawing(None)
        self.update_gt_counter()
        self.fig.canvas.draw()
    
    def cancel_drawing(self, event):
        for marker in self.drawing_markers:
            marker.remove()
        if self.temp_shape:
            self.temp_shape.remove()
            self.temp_shape = None
        
        self.drawing_points = []
        self.drawing_markers = []
        
        if event is not None:
            self.fig.canvas.draw()
    
    def delete_selected(self, event):
        """Delete selected ground truth bounding boxes"""
        to_delete = []
        for i, (poly, bbox_data, selected) in enumerate(self.bboxes):
            if selected:
                to_delete.append(i)
                poly.remove()
                if bbox_data['line_idx'] != -1:
                    self.deleted_bboxes.append(bbox_data['line_idx'])
        
        for i in reversed(to_delete):
            del self.bboxes[i]
        
        print(f"Deleted {len(to_delete)} GT bounding box(es)")
        self.update_gt_counter()
        self.fig.canvas.draw()
    
    def save_labels(self, event):
        image_path = self.tif_files[self.current_idx]
        tile_name = Path(image_path).stem
        label_path = f"{self.label_dir}/{tile_name}.txt"

        # ----------------------------------
        # 1️⃣ Build final label list ONLY from memory
        # ----------------------------------
        all_labels = []

        for poly, bbox_data, _ in self.bboxes:
            all_labels.append(bbox_data['line'])

        # ----------------------------------
        # 2️⃣ Write to file (overwrite cleanly)
        # ----------------------------------
        with open(label_path, "w") as f:
            for label in all_labels:
                f.write(label + "\n")

        # ----------------------------------
        # 3️⃣ Reset state so duplicates cannot happen
        # ----------------------------------
        for idx, (poly, bbox_data, selected) in enumerate(self.bboxes):
            bbox_data['line_idx'] = idx  # now it's treated as original
            self.bboxes[idx][1] = bbox_data

        self.deleted_bboxes = []

        print(f"Saved {len(all_labels)} labels to {label_path}")
        self.update_gt_counter()
    
    def reset_image(self, event):
        self.load_image(self.current_idx)
        self.update_gt_counter()
    
    def prev_image(self, event):
        if self.current_idx > 0:
            self.load_image(self.current_idx - 1)
    
    def next_image(self, event):
        if self.current_idx < len(self.tif_files) - 1:
            self.load_image(self.current_idx + 1)

# Run the interactive viewer
viewer = InteractiveBBoxViewer(tif_files, image_dir, label_dir, nir_output_dir)