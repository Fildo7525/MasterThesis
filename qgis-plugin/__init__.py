#-----------------------------------------------------------
# Copyright (C) 2026 Filip Lobpreis
#-----------------------------------------------------------
# Licensed under the terms of GNU GPL 2
#-----------------------------------------------------------

import sys
from pathlib import Path
from dataclasses import dataclass
from .nrn_generator import create_nrn_image_from_orthomosaic

sys.path.append(str(Path(__file__).resolve().parent / ".venv/lib/python3.12/site-packages"))

from PyQt5.QtWidgets import (
    QAction,
    QMessageBox,
    QDialog,
    QVBoxLayout,
    QLabel,
    QDialogButtonBox,
    QCheckBox,
    QRadioButton,
)

from qgis.gui import QgsMapLayerComboBox, QgsFileWidget
from qgis.core import (
    Qgis,
    QgsMapLayerProxyModel,
    QgsMessageLog,
    QgsTask,
    QgsApplication,
    QgsVectorLayer,
    QgsProject
)

from .ga_shortest_path import compute_shortest_route
from ultralytics import YOLO
from .yolo_qgis_converter import YOLOShapefileConverter

MESSAGE_CATEGORY = "ProcessOrthomosaicTask"


def classFactory(iface):
    return MinimalPlugin(iface)


@dataclass
class Approach:
    orthomosaic_dir: Path
    output_dir: Path
    generate_shortes_path: bool
    image_size: int


class ProcessOrthomosaicTask(QgsTask):
    def __init__(self, model, args, description="Processing Orthomosaic"):
        super().__init__(description, QgsTask.CanCancel)
        self.model = model
        self.args: Approach = args
        self.exception = None
        self.prediction_dir = Path.cwd()
        self.prediction_converter = YOLOShapefileConverter()
        self.tif_tiles_dir = Path.cwd() / "tiles"
        self.nrn_tiles_dir = Path.cwd() / "nrn"

    def increase_task_count(self, idx):
        self.setProgress(idx)
        QgsMessageLog.logMessage(f"Done tile: {idx}", MESSAGE_CATEGORY, Qgis.Info)


    def run(self):
        """Runs in a background thread — no Qt UI calls here."""
        self.prediction_dir = self.args.output_dir / "prediction_file"
        self.prediction_dir.mkdir(parents=True, exist_ok=True)

        try:
            tif_dir, nrn_dir = create_nrn_image_from_orthomosaic(self.args.orthomosaic_dir, self.args.output_dir, global_normalize=True)
            # tif_dir, nrn_dir = Path.home() / "SDU/MasterThesis/Orthomosaics/qgis_yolo_out/image_tiles", Path.home() / "SDU/MasterThesis/Orthomosaics/qgis_yolo_out/temp_nrn_pngs/"

            self.tif_tiles_dir = tif_dir
            self.nrn_tiles_dir = nrn_dir

            size = len(list(self.nrn_tiles_dir.glob("*.png")))
            idx = 0
            for file in self.nrn_tiles_dir.glob("*.png"):
                QgsMessageLog.logMessage(f"png file: {file}")

                img_pth = self.nrn_tiles_dir / file
                results = self.model.predict(
                    source=img_pth,
                    imgsz=self.args.image_size,
                    conf=0.8,  # match COCO eval
                    iou=0.5,
                    save_txt=True,
                    save_conf=True,
                    exist_ok=True
                )


                QgsMessageLog.logMessage("Predicted")

                prediction_file = self.prediction_dir / (img_pth.stem + ".txt")
                for result in results:
                    img_w, img_h = result.orig_shape[1], result.orig_shape[0]

                    if result.boxes is None or len(result.boxes) == 0:
                        QgsMessageLog.logMessage(f"No box found")
                        continue

                    for box in result.boxes:
                        QgsMessageLog.logMessage(f"results box: {box}")
                        cls  = int(box.cls[0])
                        conf = float(box.conf[0])

                        # xywh returns [cx, cy, w, h] in pixel coords
                        cx, cy, w, h = box.xywh[0].tolist()

                        # Normalise to [0, 1]
                        cx_n = cx / img_w
                        cy_n = cy / img_h
                        w_n  = w  / img_w
                        h_n  = h  / img_h

                        with open(prediction_file, "a") as f:
                            f.write(
                                f"{cls} {conf:.4f} "
                                f"{cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}\n"
                            )
                idx = idx + 1
                self.increase_task_count(float(idx) / size * 100)

            output_shp = self.args.output_dir / "labels_predictions.shp"
            self.prediction_converter.labels_to_shapefile(
                labels_dir = self.prediction_dir,
                reference_tif_dir = self.tif_tiles_dir,
                output_shapefile = output_shp
            )

            predictions_layer = QgsVectorLayer(str(output_shp), output_shp.stem)
            QgsProject.instance().addMapLayer(predictions_layer)

            if self.args.generate_shortes_path:
                shortest_path_path = Path(str(output_shp).replace(".shp", "_ga_path.shp"))
                compute_shortest_route(output_shp, str(shortest_path_path))

                path_layer = QgsVectorLayer(str(shortest_path_path), shortest_path_path.stem)
                QgsProject.instance().addMapLayer(path_layer)

            return True

        except Exception as e:
            self.exception = e
            return False

    def finished(self, result):
        """Called on the main thread when run() completes."""
        if self.isCanceled():
            QgsMessageLog.logMessage(
                f"Task \"{self.description()}\" was cancelled.",
                MESSAGE_CATEGORY, Qgis.Info)
            return

        if result:
            QMessageBox.information(None, "Done", "Orthomosaic processing complete!")
        else:
            QMessageBox.critical(
                None, "Error",
                f"Processing failed:\n{self.exception}"
            )

    def cancel(self):
        QgsMessageLog.logMessage(
            f"RProcessOrhomosaic \"{self.description()}\" was canceled",
            MESSAGE_CATEGORY, Qgis.Info)
        super().cancel()
        self.model.terminate()


class InputDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Potato Finder Input")
        self.setMinimumSize(600, 200)

        layout = QVBoxLayout()

        # Orthomosaic (mandatory raster)
        layout.addWidget(QLabel("Orthomosaic (required):", self))

        self.raster_box = QgsMapLayerComboBox()
        self.raster_box.setFilters(QgsMapLayerProxyModel.RasterLayer)
        layout.addWidget(self.raster_box)

        # Output directory
        layout.addWidget(QLabel("Output directory (required):"))

        self.folder_widget = QgsFileWidget(self)
        self.folder_widget.setStorageMode(QgsFileWidget.GetDirectory)
        layout.addWidget(self.folder_widget)

        # Optional shapefile
        layout.addWidget(QLabel("Shapefile:"))

        self.compute_shortest_path_checkbox = QCheckBox("Find shortest path", self)
        self.compute_shortest_path_checkbox.setChecked(False)
        layout.addWidget(self.compute_shortest_path_checkbox)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

        self.setLayout(layout)


    def get_inputs(self):
        raster = self.raster_box.currentLayer()
        output_dir = self.folder_widget.filePath()

        if not raster.isValid() or not output_dir:
            QMessageBox.information(self, "Invalid input", "Raster layer and output directory are required fields.")
            return None

        compute_shortest_path = self.compute_shortest_path_checkbox.isChecked()

        return raster, Path(output_dir), compute_shortest_path


    def on_radio_change(self, button: QRadioButton):
        button.setChecked(True)


class MinimalPlugin:

    def __init__(self, iface):
        self.iface = iface
        self.model = YOLO(str(Path(__file__).resolve().parent / "yolo26s_global_best.pt"))
        self.image_size: int = 1024

    def initGui(self):
        self.action = QAction('Potato Finder', self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        self.iface.removeToolBarIcon(self.action)
        del self.action

    def run(self):

        dlg = InputDialog()

        if dlg.exec_():

            inputs = dlg.get_inputs()

            if inputs is None:
                QMessageBox.warning(
                    None,
                    "Error",
                    "You must select an orthomosaic raster."
                )
                return

            raster, output_dir, shortest_path = inputs

            self.output_dir = Path(output_dir)
            if not self.output_dir.exists():
                self.output_dir.mkdir(exist_ok=True, parents=True)

            args = Approach(
                orthomosaic_dir = Path(str(raster.source())),
                output_dir = Path(output_dir),
                generate_shortes_path = shortest_path,
                image_size = self.image_size
            )

            task = ProcessOrthomosaicTask(self.model, args)
            self._task = task
            QgsApplication.taskManager().addTask(task)

