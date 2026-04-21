#-----------------------------------------------------------
# Copyright (C) 2026 Filip Lobpreis
#-----------------------------------------------------------
# Licensed under the terms of GNU GPL 2
#-----------------------------------------------------------

from enum import StrEnum
from PyQt5.QtWidgets import QButtonGroup, QRadioButton
from PyQt5.QtWidgets import QCheckBox
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / ".venv/lib/python3.12/site-packages"))
import joblib
# sys.path.append(str(Path("/usr/lib/python3/dist-packages")))
# sys.path.append(str(Path(__file__).resolve().parents[1] / "OpenCV"))

from .ngrvi_approach import ApproachArgs, NgrviApproach

from PyQt5.QtWidgets import QAction, QMessageBox, QDialog, QVBoxLayout, QLabel, QDialogButtonBox
from qgis.gui import QgsMapLayerComboBox
from qgis.core import QgsMapLayerProxyModel, QgsMessageLog

class Approaches(StrEnum):
    OPENCV = "Classical computer vision"
    AI = "Object detection via AI model"
    MIX = "Merged approach"


def classFactory(iface):
    return MinimalPlugin(iface)


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
                model_pth = Path.home() / "SDU/MasterThesis/OpenCV/svm_output_nrn_ex/pretrain_output_model.joblib"
                QMessageBox.information(
                    None,
                    "model",
                    f"Model path: {model_pth}\nModel exists: {model_pth.exists()}\nJoblib version: {joblib.__version__}")
                model = NgrviApproach(model_pth)

                args = ApproachArgs(
                    ground_truth_shp = Path(str(vector)) if vector else None,
                    orthomosaic_path = Path(str(raster))
                )

                model.process_orthomosaic(args)


            QMessageBox.information(None, "Inputs received", msg)
