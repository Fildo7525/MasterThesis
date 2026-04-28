#-----------------------------------------------------------
# Copyright (C) 2026 Filip Lobpreis
#-----------------------------------------------------------
# Licensed under the terms of GNU GPL 2
#-----------------------------------------------------------

from enum import StrEnum
from PyQt5.QtWidgets import QButtonGroup, QRadioButton, QPushButton
from PyQt5.QtWidgets import QCheckBox
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / ".venv/lib/python3.12/site-packages"))
import joblib
# sys.path.append(str(Path("/usr/lib/python3/dist-packages")))
# sys.path.append(str(Path(__file__).resolve().parents[1] / "OpenCV"))

from .ngrvi_approach import ApproachArgs, NgrviApproach
from .svm_pretrain import SVMDetector

from PyQt5.QtWidgets import QAction, QMessageBox, QDialog, QVBoxLayout, QLabel, QDialogButtonBox
from qgis.gui import QgsMapLayerComboBox
from qgis.core import (
    QgsMapLayerProxyModel,
    QgsMessageLog,
    QgsTask,
    QgsApplication,
    QgsVectorLayer,
    QgsProject
)

class Approaches(StrEnum):
    OPENCV = "Classical computer vision"
    AI = "Object detection via AI model"
    MIX = "Merged approach"


def classFactory(iface):
    return MinimalPlugin(iface)


class ProcessOrthomosaicTask(QgsTask):
    def __init__(self, model, args, description="Processing Orthomosaic"):
        super().__init__(description, QgsTask.CanCancel)
        self.model = model
        self.args = args
        self.exception = None

    def run(self):
        """Runs in a background thread — no Qt UI calls here."""
        try:
            predicted_shapefile = self.model.process_orthomosaic(self.args)
            vector_layer = QgsVectorLayer(str(predicted_shapefile), predicted_shapefile.stem)
            if not vector_layer.isValid():
                QMessageBox.critical(
                    None, "Error",
                    f"Layer: {predicted_shapefile} does not exist"
                )
            else:
                QgsProject.instance().addMapLayer(vector_layer)

            return True
        except Exception as e:
            self.exception = e
            return False

    def finished(self, result):
        """Called on the main thread when run() completes."""
        if result:
            QMessageBox.information(None, "Done", "Orthomosaic processing complete!")
        else:
            QMessageBox.critical(
                None, "Error",
                f"Processing failed:\n{self.exception}"
            )


class InputDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("VPD Input")
        self.setMinimumSize(600, 200)

        layout = QVBoxLayout()

        # Orthomosaic (mandatory raster)
        layout.addWidget(QLabel("Orthomosaic (required):", self))

        self.raster_box = QgsMapLayerComboBox()
        self.raster_box.setFilters(QgsMapLayerProxyModel.RasterLayer)
        layout.addWidget(self.raster_box)

        # Optional shapefile
        layout.addWidget(QLabel("Shapefile:"))

        self.use_gt_checkbox = QCheckBox("Use Ground Truth", self)
        layout.addWidget(self.use_gt_checkbox)

        self.vector_box = QgsMapLayerComboBox(self)
        self.vector_box.setFilters(QgsMapLayerProxyModel.VectorLayer)
        layout.addWidget(self.vector_box)

         # Connect checkbox
        self.use_gt_checkbox.toggled.connect(self.vector_box.setEnabled)

        # Initial state of vector dropdown menu
        self.use_gt_checkbox.setChecked(False)
        self.vector_box.setDisabled(True)

        # Choose approach with which to find the potatoes.
        self.radio_button_group = QButtonGroup(self)

        self.cv_approach = QRadioButton(Approaches.OPENCV)
        self.ai_approach = QRadioButton(Approaches.AI)
        self.merge_approach = QRadioButton(Approaches.MIX)

        self.radio_button_group.addButton(self.cv_approach)
        self.radio_button_group.addButton(self.ai_approach)
        self.radio_button_group.addButton(self.merge_approach)
        self.merge_approach.setChecked(True)

        self.radio_button_group.buttonPressed.connect(self.on_radio_change)

        self.advanced_setting = QLabel("What approach to choose", self)
        layout.addWidget(self.advanced_setting)

        layout.addWidget(self.cv_approach)
        layout.addWidget(self.ai_approach)
        layout.addWidget(self.merge_approach)

        self.approach_label = QLabel("Selected:")
        layout.addWidget(self.approach_label)

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

        if self.use_gt_checkbox.isChecked():
            vector = self.vector_box.currentLayer()
        else:
            vector = None

        approach = self.radio_button_group.checkedButton().text()
        return raster, vector, approach


    def on_radio_change(self, button: QRadioButton):
        button.setChecked(True)
        self.approach_label.setText(f"Selected: {button.text()}")


class MinimalPlugin:

    def __init__(self, iface):
        self.iface = iface

    def initGui(self):
        self.action = QAction('VPD', self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        self.iface.removeToolBarIcon(self.action)
        del self.action

    def run(self):

        dlg = InputDialog()

        if dlg.exec_():

            raster, vector, approach = dlg.get_inputs()

            if raster is None:
                QMessageBox.warning(
                    None,
                    "Error",
                    "You must select an orthomosaic raster."
                )
                return

            msg = f"Raster: {raster.source()}\n"
            if approach == Approaches.OPENCV:
                model_pth = Path.home() / "SDU/MasterThesis/OpenCV/svm_output_nrn_rgb/pretrain_output_model.joblib"
                QMessageBox.information(
                    None,
                    "model",
                    f"Model path: {model_pth}\nModel exists: {model_pth.exists()}\nJoblib version: {joblib.__version__}")
                model = NgrviApproach(model_pth)

                args = ApproachArgs(
                    ground_truth_shp = Path(str(vector.source())) if vector else None,
                    orthomosaic_path = Path(str(raster.source()))
                )

                QgsMessageLog.logMessage(f"args.ground_truth_shp: {args.ground_truth_shp}\nargs.orthomosaic_path: {args.orthomosaic_path}")

                task = ProcessOrthomosaicTask(model, args)
                self._task = task
                QgsApplication.taskManager().addTask(task)


            QMessageBox.information(None, "Inputs received", msg)
