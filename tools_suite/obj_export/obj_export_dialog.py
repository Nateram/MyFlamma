# -*- coding: utf-8 -*-
"""
Dialog for OBJ Export tool - Input file + export options.
"""

import os
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QCheckBox, QDoubleSpinBox, QSpinBox, QLineEdit, QFileDialog, QMessageBox
)
from qgis.core import QgsProject, QgsPointCloudLayer


class ObjExportDialog(QDialog):
    def __init__(self, parent=None, translator=None, require_input=False, iface=None):
        super().__init__(parent)
        self.tr_func = translator if translator else lambda x: x
        self.require_input = require_input
        self.iface = iface
        self.setWindowTitle(self.tr_func("Export to OBJ"))
        self.setMinimumWidth(500)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()

        # --- Descripción ---
        desc_text = (
            "<b>Export to OBJ</b><br>"
            "Generate a 3D OBJ file with trees, shrubs, grass, and buildings.<br>"
            "The file will be saved in the project directory as <i>*_3d_scene.obj</i>"
        )

        desc = QLabel(desc_text)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # --- Input file (if needed) ---
        if self.require_input:
            input_group = QGroupBox(self.tr_func("Input LAS/LAZ File"))
            input_layout = QVBoxLayout()
            
            # File selector
            file_row = QHBoxLayout()
            file_row.addWidget(QLabel(self.tr_func("Selected file:")))
            
            self.input_edit = QLineEdit()
            self.input_edit.setPlaceholderText(self.tr_func("Browse for LAS/LAZ file..."))
            
            # Pre-fill with last imported layer if available
            last_layer_path = self._get_last_point_cloud_layer()
            if last_layer_path:
                self.input_edit.setText(last_layer_path)
            
            file_row.addWidget(self.input_edit)
            
            btn_browse = QPushButton("...")
            btn_browse.setFixedWidth(40)
            btn_browse.clicked.connect(self._browse_input)
            file_row.addWidget(btn_browse)
            
            input_layout.addLayout(file_row)
            input_group.setLayout(input_layout)
            layout.addWidget(input_group)

        # --- Export Options ---
        opts_group = QGroupBox(self.tr_func("Elements to Include"))
        opts_layout = QVBoxLayout()

        self.check_ground = QCheckBox(self.tr_func("Include ground plane"))
        self.check_ground.setChecked(True)
        opts_layout.addWidget(self.check_ground)

        self.check_trees = QCheckBox(self.tr_func("Include trees (trunk + crown)"))
        self.check_trees.setChecked(True)
        opts_layout.addWidget(self.check_trees)

        self.check_shrubs = QCheckBox(self.tr_func("Include shrubs (cubes)"))
        self.check_shrubs.setChecked(True)
        opts_layout.addWidget(self.check_shrubs)

        self.check_grass = QCheckBox(self.tr_func("Include grass (flat squares)"))
        self.check_grass.setChecked(True)
        opts_layout.addWidget(self.check_grass)

        self.check_buildings = QCheckBox(self.tr_func("Include buildings (boxes)"))
        self.check_buildings.setChecked(True)
        opts_layout.addWidget(self.check_buildings)

        self.check_roads = QCheckBox(self.tr_func("Include roads (flat planes)"))
        self.check_roads.setChecked(True)
        opts_layout.addWidget(self.check_roads)

        opts_group.setLayout(opts_layout)
        layout.addWidget(opts_group)

        # --- Export Configuration ---
        config_group = QGroupBox(self.tr_func("Export Configuration"))
        config_layout = QVBoxLayout()

        # Scale factor
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel(self.tr_func("Scale factor:")))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.01, 100.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setDecimals(2)
        self.scale_spin.setToolTip(self.tr_func("Scale multiplier for the model (1.0 = real world meters)"))
        scale_row.addWidget(self.scale_spin)
        config_layout.addLayout(scale_row)

        # Ground padding
        padding_row = QHBoxLayout()
        padding_row.addWidget(QLabel(self.tr_func("Ground padding (m):")))
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 200)
        self.padding_spin.setValue(10)
        self.padding_spin.setToolTip(self.tr_func("Extra margin around all elements for the ground plane"))
        padding_row.addWidget(self.padding_spin)
        config_layout.addLayout(padding_row)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        btn_export = QPushButton(self.tr_func("Export"))
        btn_export.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        btn_export.clicked.connect(self._validate_and_accept)
        btn_layout.addWidget(btn_export)

        btn_cancel = QPushButton(self.tr_func("Cancel"))
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def _get_point_cloud_layers(self):
        """Get list of (layer_name, layer_path) for point cloud layers in project."""
        layers = []
        project = QgsProject.instance()
        for layer in project.mapLayers().values():
            if isinstance(layer, QgsPointCloudLayer) and layer.isValid():
                source = layer.source()
                layers.append((layer.name(), source))
        return layers

    def _get_last_point_cloud_layer(self):
        """Get the path of the last imported point cloud layer, or None."""
        layers = self._get_point_cloud_layers()
        if layers:
            return layers[-1][1]  # Return the path of the last layer
        return None

    def _on_layer_changed(self, index):
        """When user changes the layer selection, update the file path."""
        if hasattr(self, 'layer_combo') and index >= 0:
            selected_path = self.layer_combo.itemData(index)
            self.input_edit.setText(selected_path)

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr_func("Select LAS/LAZ File"), "",
            "LAS/LAZ Files (*.las *.laz);;All Files (*)"
        )
        if path:
            self.input_edit.setText(path)

    def _validate_and_accept(self):
        input_file = self.get_input_file()
        if self.require_input and not input_file:
            QMessageBox.warning(self, self.tr_func("No Input"),
                                self.tr_func("Please select a LAS/LAZ file or use an imported layer."))
            return
        self.accept()

    def get_input_file(self):
        """Return input file path if required, else None."""
        if self.require_input:
            return self.input_edit.text().strip()
        return None

    def get_params(self):
        """Return all dialog parameters as a dictionary."""
        return {
            'include_ground': self.check_ground.isChecked(),
            'include_trees': self.check_trees.isChecked(),
            'include_shrubs': self.check_shrubs.isChecked(),
            'include_grass': self.check_grass.isChecked(),
            'include_buildings': self.check_buildings.isChecked(),
            'include_roads': self.check_roads.isChecked(),
            'scale': self.scale_spin.value(),
            'ground_padding': self.padding_spin.value(),
        }
