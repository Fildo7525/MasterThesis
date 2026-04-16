#-----------------------------------------------------------
# Copyright (C) 2026 Filip Lobpreis
#-----------------------------------------------------------
# Licensed under the terms of GNU GPL 2
#-----------------------------------------------------------

from PyQt5.QtWidgets import QCheckBox
import sys
from pathlib import Path
sys.path.append(str(Path("/usr/lib/python3/dist-packages")))

from PyQt5.QtWidgets import QAction, QMessageBox, QDialog, QVBoxLayout, QLabel, QPushButton, QDialogButtonBox
from PyQt5.QtCore import QSize
from qgis.gui import QgsMapLayerComboBox
from qgis.core import QgsMapLayerProxyModel


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

        return raster, vector


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

            raster, vector = dlg.get_inputs()

            if raster is None:
                QMessageBox.warning(
                    None,
                    "Error",
                    "You must select an orthomosaic raster."
                )
                return

            msg = f"Raster: {raster.name()}\n"

            if vector:
                msg += f"Shapefile: {vector.name()}"
            else:
                msg += "No shapefile selected"

            QMessageBox.information(None, "Inputs received", msg)
