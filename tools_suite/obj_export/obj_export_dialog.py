# -*- coding: utf-8 -*-
"""
Dialog for OBJ Export tool - Input file + export options.
"""

import os
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QCheckBox, QDoubleSpinBox, QSpinBox, QLineEdit, QFileDialog, QMessageBox,
    QDialogButtonBox, QWidget, QSpacerItem, QSizePolicy, QTextBrowser
)
from qgis.PyQt.QtCore import Qt
from qgis.core import QgsProject, QgsPointCloudLayer


class ObjExportDialog(QDialog):
    def __init__(self, parent=None, translator=None, require_input=False, iface=None):
        super().__init__(parent)
        self.tr_func = translator if translator else lambda x: x
        self.require_input = True  # Always require input file selection
        self.iface = iface
        self.setWindowTitle(self.tr_func("Export to OBJ"))
        self.resize(900, 600)
        self.setMinimumWidth(850)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)

        # --- Left Panel ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.setAlignment(Qt.AlignTop)

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
            
            btn_browse = QPushButton(self.tr_func("..."))
            btn_browse.setFixedWidth(40)
            btn_browse.clicked.connect(self._browse_input)
            file_row.addWidget(btn_browse)
            
            input_layout.addLayout(file_row)
            input_group.setLayout(input_layout)
            left_layout.addWidget(input_group)

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

        self.check_terrain = QCheckBox(self.tr_func("Include terrain elevation"))
        self.check_terrain.setChecked(True)
        self.check_terrain.setToolTip(self.tr_func("Generate 3D terrain mesh from ground elevation data (mountains, hills, slopes)"))
        opts_layout.addWidget(self.check_terrain)

        opts_group.setLayout(opts_layout)
        left_layout.addWidget(opts_group)

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
        left_layout.addWidget(config_group)

        left_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # --- OK / Cancel buttons ---
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        left_layout.addWidget(buttons)

        # --- Right Panel (Description) ---
        desc_box = QTextBrowser()
        desc_box.setOpenExternalLinks(False)
        desc_box.setFixedWidth(320)
        desc_box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        desc_box.setStyleSheet("""
            QTextBrowser {
                background-color: #fafafa;
                border: 1px solid #dcdcdc;
                padding: 10px;
                border-radius: 6px;
                font-family: "Segoe UI", sans-serif;
                font-size: 10pt;
            }
        """)

        title = self.tr_func("OBJ 3D Export")
        intro = self.tr_func(
            "This tool generates a 3D OBJ file from classified LiDAR data, including trees, shrubs, grass, and buildings with real terrain elevation."
        )
        workflow = self.tr_func("Workflow:")
        step1 = self.tr_func("Reads classified data from memory or runs a quick classification on the input LAS/LAZ file.")
        step2 = self.tr_func("Generates 3D geometry for each element type: trunks and crowns for trees, cubes for shrubs, inclined quads for grass, extruded boxes for buildings, and a terrain mesh with real elevation data.")
        step3 = self.tr_func("Creates an OBJ file with materials (MTL) ready for import into Unity, Blender, or other 3D tools.")
        note = self.tr_func(
            "The output file will be saved as *_3d_scene.obj alongside the input file. "
            "A matching .mtl material file is generated automatically. "
            "Coordinate system uses Y-up convention (compatible with Unity)."
        )

        desc_html = f"""
            <div style="position: relative;">
                <h3 style="margin-bottom:4px;">{title}</h3>
                <p style="font-size:9.5pt; color:#444;">{intro}</p>
                <hr style="border:none; border-top:1px solid #ccc; margin:6px 0;">
                <h4 style="margin-bottom:2px;">{workflow}</h4>
                <ul>
                    <li>{step1}</li>
                    <li>{step2}</li>
                    <li>{step3}</li>
                </ul>
                <p style="margin-top:4px; font-size:9pt; color:#666;">{note}</p>
            </div>
        """
        desc_box.setHtml(desc_html)

        main_layout.addWidget(left_panel, stretch=3)
        main_layout.addWidget(desc_box, stretch=2)

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
        """Return input file path."""
        return self.input_edit.text().strip()

    def get_params(self):
        """Return all dialog parameters as a dictionary."""
        return {
            'include_ground': self.check_ground.isChecked(),
            'include_trees': self.check_trees.isChecked(),
            'include_shrubs': self.check_shrubs.isChecked(),
            'include_grass': self.check_grass.isChecked(),
            'include_buildings': self.check_buildings.isChecked(),
            'include_terrain': self.check_terrain.isChecked(),
            'scale': self.scale_spin.value(),
            'ground_padding': self.padding_spin.value(),
        }
