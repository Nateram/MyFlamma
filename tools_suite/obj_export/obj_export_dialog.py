# -*- coding: utf-8 -*-
"""
Dialog for OBJ Export tool - Input file + export options.
"""

import os
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QCheckBox, QDoubleSpinBox, QSpinBox, QLineEdit,
    QComboBox, QFileDialog, QMessageBox, QDialogButtonBox, QWidget, QSpacerItem, QSizePolicy, QTextBrowser
)
from qgis.PyQt.QtCore import Qt
from qgis.core import QgsProject, QgsPointCloudLayer


class ObjExportDialog(QDialog):
    def __init__(self, parent=None, translator=None, iface=None):
        super().__init__(parent)
        self.tr_func = translator if translator else lambda x: x
        self.iface = iface
        self.selected_output = None
        self.selected_input = None
        self.user_edited_output = False
        self._updating_output = False
        self.setWindowTitle(self.tr_func("Export to OBJ"))
        self.resize(900, 570)
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

        # --- Input file ---
        left_layout.addWidget(QLabel(self.tr_func("LiDAR layer or file:")))
        file_row = QHBoxLayout()

        self.input_combo = QComboBox()
        self.input_combo.setToolTip(self.tr_func("Select a point cloud layer from the project"))
        file_row.addWidget(self.input_combo)

        btn_browse = QPushButton(self.tr_func("..."))
        btn_browse.setFixedWidth(28)
        btn_browse.setToolTip(self.tr_func("Select input file (.las / .laz)"))
        btn_browse.clicked.connect(self._browse_input)
        file_row.addWidget(btn_browse)

        left_layout.addLayout(file_row)

        # --- Output file ---
        left_layout.addWidget(QLabel(self.tr_func("Output OBJ File")))
        output_layout = QHBoxLayout()

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText(self.tr_func("Select output OBJ file..."))
        self.output_edit.textChanged.connect(self._on_output_text_changed)
        output_layout.addWidget(self.output_edit)

        btn_output = QPushButton(self.tr_func("..."))
        btn_output.setFixedWidth(28)
        btn_output.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn_output.clicked.connect(self._browse_output)
        output_layout.addWidget(btn_output)

        left_layout.addLayout(output_layout)

        self.populate_input_layers()
        self.input_combo.currentIndexChanged.connect(self._on_layer_changed)

        if self.get_input_file():
            self._update_default_output()

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
            "The output file will be saved as *_3d_scene.obj alongside the input file. A matching .mtl material file is generated automatically. Coordinate system uses Y-up convention (compatible with Unity)."
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
        """Get valid point cloud layers currently loaded in the project."""
        layers = []
        project = QgsProject.instance()
        for layer in project.mapLayers().values():
            if isinstance(layer, QgsPointCloudLayer) and layer.isValid():
                layers.append(layer)
        return layers

    def populate_input_layers(self):
        """Populate the input selector with point cloud layers in the project."""
        self.input_combo.clear()
        for layer in self._get_point_cloud_layers():
            crs = f" [{layer.crs().authid()}]" if layer.crs().isValid() else ""
            self.input_combo.addItem(
                f"{layer.name()}{crs}",
                layer.source()
            )

        if self.input_combo.count():
            self.input_combo.setCurrentIndex(0)
            self._on_layer_changed(0)

    def _on_layer_changed(self, index):
        """Update the selected input path when a project layer is chosen."""
        if index < 0:
            self.selected_input = None
            return

        selected_path = self.input_combo.itemData(index)
        self.selected_input = selected_path
        if not self.user_edited_output:
            self._update_default_output()

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr_func("Select LiDAR File"), "",
            self.tr_func("LiDAR Files (*.las *.laz)")
        )
        if path:
            existing_index = self.input_combo.findData(path)
            if existing_index == -1:
                self.input_combo.addItem(os.path.basename(path), path)
                existing_index = self.input_combo.count() - 1
            self.input_combo.setCurrentIndex(existing_index)
            self.selected_input = path
            self.user_edited_output = False
            self._update_default_output()

    def _browse_output(self):
        default_dir = os.path.dirname(self.get_input_file()) or ""
        default_name = self._default_output_name()
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr_func("Save OBJ File"),
            os.path.join(default_dir, default_name),
            self.tr_func("OBJ Files (*.obj)")
        )
        if path:
            if not path.lower().endswith('.obj'):
                path += '.obj'
            self._updating_output = True
            self.output_edit.setText(path)
            self._updating_output = False
            self.selected_output = path
            self.user_edited_output = True

    def _default_output_name(self):
        input_file = self.get_input_file()
        if not input_file:
            return "scene_3d.obj"
        return os.path.splitext(os.path.basename(input_file))[0] + "_3d_scene.obj"

    def _update_default_output(self):
        if self.user_edited_output:
            return
        default_output = self._default_output_name()
        input_dir = os.path.dirname(self.get_input_file()) or ""
        full_path = os.path.join(input_dir, default_output) if input_dir else default_output
        self._updating_output = True
        self.output_edit.setText(full_path)
        self._updating_output = False
        self.selected_output = full_path

    def _on_output_text_changed(self, text):
        if self._updating_output:
            return
        self.selected_output = text.strip()
        if text.strip():
            self.user_edited_output = True

    def _validate_and_accept(self):
        input_file = self.get_input_file()
        if not input_file:
            QMessageBox.warning(self, self.tr_func("No Input"),
                                self.tr_func("Please select a LAS/LAZ file or use an imported layer."))
            return
        output_file = self.get_output_file()
        if not output_file:
            QMessageBox.warning(self, self.tr_func("No Output"),
                                self.tr_func("Please choose an output OBJ file."))
            return
        self.accept()

    def get_input_file(self):
        """Return input file path."""
        return (self.selected_input or "").strip()

    def get_output_file(self):
        """Return output OBJ file path, ensuring a .obj extension."""
        path = self.output_edit.text().strip()
        if not path:
            return ""
        if not path.lower().endswith('.obj'):
            path += '.obj'
        return path

    def get_params(self):
        """Return all dialog parameters as a dictionary."""
        return {
            'output_obj': self.get_output_file(),
            'include_ground': self.check_ground.isChecked(),
            'include_trees': self.check_trees.isChecked(),
            'include_shrubs': self.check_shrubs.isChecked(),
            'include_grass': self.check_grass.isChecked(),
            'include_buildings': self.check_buildings.isChecked(),
            'include_terrain': self.check_terrain.isChecked(),
            'scale': self.scale_spin.value(),
            'ground_padding': self.padding_spin.value(),
        }
