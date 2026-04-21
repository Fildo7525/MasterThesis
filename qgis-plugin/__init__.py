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

sys.path.append(str(Path(__file__).resolve().parents[1] / ".venv/lib/python3.12/site-packages"))
import joblib
# sys.path.append(str(Path("/usr/lib/python3/dist-packages")))
# sys.path.append(str(Path(__file__).resolve().parents[1] / "OpenCV"))

from .ngrvi_approach import ApproachArgs, NgrviApproach
from .svm_pretrain import SVMDetector

from PyQt5.QtWidgets import QAction, QMessageBox, QDialog, QVBoxLayout, QLabel, QDialogButtonBox
from qgis.gui import QgsMapLayerComboBox
from qgis.core import QgsMapLayerProxyModel, QgsMessageLog, QgsTask, QgsApplication

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
            self.model.process_orthomosaic(self.args)
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

        train_button = QPushButton("Train SVM", self)
        train_button.clicked.connect(self.train_invoked)
        layout.addWidget(train_button)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)

        self.setLayout(layout)


    def train_invoked(self):
        from .svm_pretrain import Pretrainer, PretrainConfig, get_feature_names
        from .create_indexes import Bands, Indices

        NU           = 0.001
        KERNEL       = "rbf"
        PCA_VARIANCE = 0.95    # fraction of variance to retain after PCA

        OUTPUT_PATH = Path.home() / "SDU/MasterThesis/OpenCV/svm_output_nrn_rgb"
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        BANDS_TO_USE   = [Bands.EXTEND_RED, Bands.NIR]
        INDICES_TO_USE = [Indices.NGRDI]

        home = Path.home()

        configs = [
            PretrainConfig(
                ortho_path     = home / "SDU/MasterThesis/Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small.tif",
                shapefile_path = home / "SDU/MasterThesis/Orthomosaics/shapefiles/small/small_obb_test.shp",
            ),
            PretrainConfig(
                ortho_path     = home / "SDU/MasterThesis/Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_mid.tif",
                shapefile_path = home / "SDU/MasterThesis/Orthomosaics/shapefiles/mid/mid_obb_test.shp",
            ),
            PretrainConfig(
                ortho_path     = home / "SDU/MasterThesis/Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_bigger_v2.tif",
                shapefile_path = home / "SDU/MasterThesis/Orthomosaics/shapefiles/large/large_obb_test.shp",
            ),
        ]

        trainer = Pretrainer(
            nu                 = NU,
            kernel             = KERNEL,
            pca_variance       = PCA_VARIANCE,
            band_indices       = BANDS_TO_USE,
            vegetation_indices = INDICES_TO_USE,
            rectangle          = False,
        )

        # Phase 1: accumulate feature vectors from all orthomosaics
        for cfg in configs:
            trainer.train(
                ortho_path     = cfg.ortho_path,
                shapefile_path = cfg.shapefile_path,
                limit          = 0.8,
            )

        # Phase 2: fit once on the full combined matrix
        trainer.fit()

        # Phase 3: save
        trainer.dump(OUTPUT_PATH / "pretrain_output_model.joblib")

        # Phase 4: diagnostics
        feature_names = get_feature_names()

        from .run_diagnostics import plot_feature_matrix, plot_pca_importance, plot_pca_scatter
        plot_feature_matrix(trainer, OUTPUT_PATH / "feature_matrix.png",
                            feature_names=feature_names, max_features=10)
        plot_pca_importance(trainer, OUTPUT_PATH / "pca_importance.png",
                            feature_names=feature_names, pca_variance=PCA_VARIANCE)
        plot_pca_scatter(trainer, OUTPUT_PATH / "pca_scatter.png")


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
