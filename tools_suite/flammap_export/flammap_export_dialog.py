from qgis.PyQt.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,QDialogButtonBox, QFileDialog, QLineEdit, QSizePolicy, QTextBrowser, QWidget, QSpacerItem, QGroupBox, QFormLayout, QDoubleSpinBox, QCheckBox, QListWidget, QListWidgetItem, QAction, QMessageBox
from qgis.PyQt.QtCore import Qt, QSettings
from qgis.core import QgsProject, QgsPointCloudLayer
from qgis.gui import QgsMapTool
import os
import functools

class FlamMapExportDialog(QDialog):
    """Dialog window for FlamMap export settings."""

    def __init__(self, parent=None, translator=lambda s: s, iface=None, ignition_points_list=None):
        super().__init__(parent)
        self.tr = translator
        self.iface = iface
        self.ignition_points = ignition_points_list if ignition_points_list is not None else []
        self.settings = QSettings()
        self.selected_input = None
        self.selected_output = None
        self.is_layer = False
        self.user_edited_output = False

        # --- Window ---
        self.setWindowTitle(self.tr("Export to FlamMap GeoTIFF"))
        self.resize(850, 560)
        self.setMinimumWidth(800)
        self.setModal(False)

        # --- Layout ---
        main_layout = QHBoxLayout(self)

        # --- Left Panel (Inputs, Parameters, and Buttons) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.setAlignment(Qt.AlignTop)

        # --- Input selection ---
        left_layout.addWidget(QLabel(self.tr("LiDAR layer or file:")))
        input_layout = QHBoxLayout()

        self.input_combo = QComboBox()
        self.input_combo.setEditable(False)
        input_layout.addWidget(self.input_combo)

        self.input_button = QPushButton("...")
        self.input_button.setToolTip(self.tr("Select input file (.las / .laz)"))
        self.input_button.setFixedWidth(28)
        self.input_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        input_layout.addWidget(self.input_button)
        left_layout.addLayout(input_layout)

        # --- Output selection ---
        left_layout.addWidget(QLabel(self.tr("Output file:")))
        output_layout = QHBoxLayout()

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText(self.tr("Select output file path..."))
        output_layout.addWidget(self.output_edit)

        self.output_button = QPushButton("...")
        self.output_button.setToolTip(self.tr("Select output file (.tif)"))
        self.output_button.setFixedWidth(28)
        self.output_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        output_layout.addWidget(self.output_button)
        left_layout.addLayout(output_layout)

        # --- Parameters group ---
        param_group = QGroupBox(self.tr("Export Parameters"))
        param_layout = QFormLayout(param_group)
        param_layout.setLabelAlignment(Qt.AlignRight)

        # Cell resolution
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setRange(0.1, 100.0)
        self.resolution_spin.setSingleStep(0.1)
        self.resolution_spin.setDecimals(1)
        self.resolution_spin.setValue(5.0)
        self.resolution_spin.setSuffix(" m")
        param_layout.addRow(self.tr("Cell resolution (m):"), self.resolution_spin)

        # Generate scenarios checkbox
        self.scenarios_checkbox = QCheckBox(self.tr("Generate Simulation Scenarios (Ignition & Weather)"))
        self.scenarios_checkbox.setChecked(False)
        self.scenarios_checkbox.setToolTip(self.tr("Creates GeoTIFF files + ignition points (.shp) and weather files (.fms) in a 'scenarios' subfolder"))
        param_layout.addRow(self.scenarios_checkbox)

        # Generate only scenarios checkbox
        self.only_scenarios_checkbox = QCheckBox(self.tr("Generate Only Scenarios and Ignition Points"))
        self.only_scenarios_checkbox.setChecked(False)
        self.only_scenarios_checkbox.setToolTip(self.tr("Creates only ignition points (.shp) and weather files (.fms) in a 'scenarios' subfolder, without GeoTIFF files"))
        param_layout.addRow(self.only_scenarios_checkbox)

        left_layout.addWidget(param_group)

        # --- Ignition points group ---
        ignition_group = QGroupBox(self.tr("Ignition Points"))
        ignition_layout = QVBoxLayout(ignition_group)

        self.ignition_widget = QWidget()
        self.ignition_layout_inner = QVBoxLayout(self.ignition_widget)
        self.ignition_layout_inner.setContentsMargins(0, 0, 0, 0)
        ignition_layout.addWidget(self.ignition_widget)

        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel(self.tr("Coordinates (X,Y):")))
        self.coord_edit = QLineEdit()
        self.coord_edit.setPlaceholderText("e.g. 123456.78, 9876543.21")
        coord_layout.addWidget(self.coord_edit)
        add_btn = QPushButton(self.tr("Add Point"))
        add_btn.clicked.connect(self.add_point_from_text)
        coord_layout.addWidget(add_btn)
        ignition_layout.addLayout(coord_layout)

        left_layout.addWidget(ignition_group)

        left_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # --- OK / Cancel buttons ---
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
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

        title = self.tr("FlamMap GeoTIFF Export")
        intro = self.tr(
            "This tool exports LiDAR point cloud data to a multi-band GeoTIFF file compatible with FlamMap fire simulation software."
        )
        workflow = self.tr("Workflow:")
        step1 = self.tr("Reads the LiDAR file and extracts coordinates and classifications.")
        step2 = self.tr("Rasterizes the data into a grid at the specified resolution.")
        step3 = self.tr("Calculates 8 bands: Elevation, Slope, Aspect, Fuel Model, Canopy Cover, Stand Height, Canopy Base Height, and Canopy Bulk Density.")
        note = self.tr(
            "The output will be 8 individual GeoTIFF files (one per variable). If 'Generate Scenarios' is checked, ignition points (.shp) and fuel moisture files (.fms) will be created in a /scenarios subfolder. If 'Generate Only Scenarios' is checked, only scenarios will be created without GeoTIFF files. You can manually add ignition points by entering coordinates."
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

        # --- Connections ---
        self.populate_input_layers()
        self.input_combo.currentIndexChanged.connect(self.on_input_changed)
        self.input_button.clicked.connect(self.select_input_file)
        self.output_button.clicked.connect(self.select_output_file)
        self.coord_edit.returnPressed.connect(self.add_point_from_text)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # Update display
        self.update_ignition_list()

    def accept(self):
        super().accept()

    def reject(self):
        super().reject()

    # --- Layer & File Management ---
    def populate_input_layers(self):
        """Populate combo with loaded LiDAR layers."""
        self.input_combo.clear()
        layers = [
            layer for layer in QgsProject.instance().mapLayers().values()
            if isinstance(layer, QgsPointCloudLayer)
        ]

        if not layers:
            self.selected_input = None
            self.is_layer = False
            return

        for layer in layers:
            crs = f" [{layer.crs().authid()}]" if layer.crs().isValid() else ""
            self.input_combo.addItem(f"{layer.name()}{crs}", layer)

        self.input_combo.setCurrentIndex(0)
        self.on_input_changed(0)

    def on_input_changed(self, index):
        """Handle selection of input layer."""
        layer = self.input_combo.itemData(index)
        if isinstance(layer, QgsPointCloudLayer):
            self.selected_input = layer.source()
            self.is_layer = True
        elif isinstance(layer, str):
            self.selected_input = layer
            self.is_layer = False
        else:
            return

        if not self.user_edited_output:
            self.update_default_output()

    def update_default_output(self):
        """Generate a default output file path."""
        if not self.selected_input:
            return
        base_name = os.path.splitext(os.path.basename(self.selected_input))[0]
        default_output = os.path.join(
            os.path.dirname(self.selected_input),
            base_name + "_flammap.tif"
        )
        self.output_edit.setText(default_output)
        self.selected_output = default_output

    def select_input_file(self):
        """Prompt user to select an input LiDAR file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select LiDAR File"),
            "",
            self.tr("LiDAR Files (*.las *.laz)")
        )
        if filename:
            existing_index = self.input_combo.findData(filename)
            if existing_index == -1:
                self.input_combo.addItem(filename, filename)
                self.input_combo.setCurrentIndex(self.input_combo.count() - 1)
            else:
                self.input_combo.setCurrentIndex(existing_index)
            self.selected_input = filename
            self.is_layer = False
            self.user_edited_output = False
            self.update_default_output()

    def select_output_file(self):
        """Prompt user to select output GeoTIFF file path."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save FlamMap GeoTIFF File"),
            self.output_edit.text() or "",
            self.tr("GeoTIFF Files (*.tif)")
        )
        if filename:
            self.output_edit.setText(filename)
            self.selected_output = filename
            self.user_edited_output = True

    def get_resolution(self):
        """Return cell resolution value."""
        return self.resolution_spin.value()

    def get_generate_scenarios(self):
        """Return whether to generate simulation scenarios."""
        return self.scenarios_checkbox.isChecked()

    def get_generate_only_scenarios(self):
        """Return whether to generate only scenarios and ignition points."""
        return self.only_scenarios_checkbox.isChecked()

    def add_point_from_text(self):
        text = self.coord_edit.text().strip()
        try:
            x_str, y_str = text.split(',')
            x = float(x_str.strip())
            y = float(y_str.strip())
            self.add_ignition_point(x, y)
            self.coord_edit.clear()
        except ValueError:
            QMessageBox.warning(self, self.tr("Invalid Coordinates"), self.tr("Please enter coordinates in the format X,Y"))

    def add_ignition_point(self, x, y):
        self.ignition_points.append((x, y))
        self.settings.setValue("MyLiDAR/ignition_points", self.ignition_points)
        self.update_ignition_list()

    def update_ignition_list(self):
        # Clear existing
        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                elif item.layout():
                    clear_layout(item.layout())
        clear_layout(self.ignition_layout_inner)

        # Add new
        for i, (x, y) in enumerate(self.ignition_points):
            h_layout = QHBoxLayout()
            label = QLabel(f"Point {i+1}: ({x:.2f}, {y:.2f})")
            remove_btn = QPushButton("X")
            remove_btn.setFixedWidth(30)
            remove_btn.setStyleSheet("background-color: red; color: white;")
            remove_btn.clicked.connect(functools.partial(self.remove_point, i))
            h_layout.addWidget(label)
            h_layout.addWidget(remove_btn)
            h_layout.addStretch()
            self.ignition_layout_inner.addLayout(h_layout)

    def remove_point(self, idx):
        if 0 <= idx < len(self.ignition_points):
            del self.ignition_points[idx]
            self.settings.setValue("MyLiDAR/ignition_points", self.ignition_points)
            self.update_ignition_list()

    def get_ignition_points(self):
        return self.ignition_points

    def get_input_output(self):
        """Return (input_path, output_path)."""
        return self.selected_input, self.output_edit.text().strip()
