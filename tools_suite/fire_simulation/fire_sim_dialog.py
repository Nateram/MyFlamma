from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                 QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox, 
                                 QGroupBox, QFormLayout, QDialogButtonBox, QMessageBox,
                                 QTabWidget, QWidget, QComboBox, QCheckBox, QFileDialog)
from qgis.core import QgsSettings

class FireSimulationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de Simulación (FlamMap)")
        self.resize(600, 500)
        self.settings = QgsSettings()
        
        main_layout = QVBoxLayout(self)

        # --- SELECCIÓN DE ARCHIVOS (Siempre visible) ---
        files_group = QGroupBox("Archivos Base")
        files_layout = QFormLayout(files_group)
        
        # Lidar
        lidar_box = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_btn = QPushButton("...")
        self.input_btn.clicked.connect(self.select_lidar)
        lidar_box.addWidget(self.input_edit)
        lidar_box.addWidget(self.input_btn)
        files_layout.addRow("LiDAR Input:", lidar_box)

        # Selector de capas cargadas
        from qgis.core import QgsProject, QgsMapLayer, QgsPointCloudLayer
        self.lidar_combo = QComboBox()
        self.lidar_combo.addItem("(Seleccionar capa cargada)")
        self.lidar_layers = []
        for layer in QgsProject.instance().mapLayers().values():
            if layer.type() == QgsMapLayer.PointCloudLayer:
                self.lidar_combo.addItem(layer.name())
                self.lidar_layers.append(layer)
        self.lidar_combo.currentIndexChanged.connect(self.set_lidar_from_layer)
        files_layout.addRow("Capa LiDAR cargada:", self.lidar_combo)
        
        # Exe
        exe_box = QHBoxLayout()
        self.exe_edit = QLineEdit()
        last_exe = self.settings.value("MyLiDAR/flammap_exe", "", type=str)
        self.exe_edit.setText(last_exe)
        self.exe_btn = QPushButton("Buscar EXE")
        self.exe_btn.clicked.connect(self.select_exe)
        exe_box.addWidget(self.exe_edit)
        exe_box.addWidget(self.exe_btn)
        files_layout.addRow("Motor FlamMap:", exe_box)
        
        main_layout.addWidget(files_group)

        # --- PESTAÑAS DE CONFIGURACIÓN ---
        self.tabs = QTabWidget()
        
        # TAB 1: Clima y Tiempo (Lo básico)
        self.tab_weather = QWidget()
        weather_layout = QFormLayout(self.tab_weather)
        
        self.wind_speed = QSpinBox()
        self.wind_speed.setRange(0, 200)
        self.wind_speed.setValue(30)
        self.wind_speed.setSuffix(" km/h")
        weather_layout.addRow("Velocidad Viento:", self.wind_speed)

        self.wind_dir = QSpinBox()
        self.wind_dir.setRange(0, 360)
        self.wind_dir.setValue(225)
        self.wind_dir.setSuffix(" ° (Azimut)")
        weather_layout.addRow("Dirección Viento:", self.wind_dir)

        self.duration = QSpinBox()
        self.duration.setRange(10, 2880)
        self.duration.setValue(60)
        self.duration.setSuffix(" min")
        weather_layout.addRow("Duración Simulación:", self.duration)
        
        self.foliar_moisture = QSpinBox()
        self.foliar_moisture.setRange(1, 150)
        self.foliar_moisture.setValue(100)
        self.foliar_moisture.setSuffix(" %")
        weather_layout.addRow("Humedad Foliar:", self.foliar_moisture)
        
        self.tabs.addTab(self.tab_weather, "Clima")

        # TAB 2: Física del Fuego (Avanzado)
        self.tab_physics = QWidget()
        physics_layout = QFormLayout(self.tab_physics)

        self.resolution = QDoubleSpinBox()
        self.resolution.setRange(0.5, 100.0)
        self.resolution.setValue(5.0)
        self.resolution.setSuffix(" m")
        physics_layout.addRow("Resolución de Cálculo:", self.resolution)

        self.crown_method = QComboBox()
        self.crown_method.addItem("Finney (1998)", 0)
        self.crown_method.addItem("Scott & Reinhardt (2001)", 1)
        self.crown_method.setCurrentIndex(1) # Recomendado Scott
        physics_layout.addRow("Método de Fuego de Copas:", self.crown_method)

        self.spot_prob = QDoubleSpinBox()
        self.spot_prob.setRange(0.0, 1.0)
        self.spot_prob.setSingleStep(0.01)
        self.spot_prob.setValue(0.05) # 5% por defecto
        physics_layout.addRow("Probabilidad de Focos (Spotting):", self.spot_prob)
        
        self.tabs.addTab(self.tab_physics, "Física")

        # TAB 3: Outputs (Qué mapas generar)
        self.tab_outputs = QWidget()
        outputs_layout = QVBoxLayout(self.tab_outputs)
        outputs_layout.addWidget(QLabel("Selecciona los mapas a generar:"))
        
        self.chk_arrival = QCheckBox("Tiempo de Llegada (Arrival Time)")
        self.chk_arrival.setChecked(True)
        self.chk_intensity = QCheckBox("Intensidad de Línea (Fireline Intensity)")
        self.chk_intensity.setChecked(True)
        self.chk_flame = QCheckBox("Longitud de Llama (Flame Length)")
        self.chk_flame.setChecked(True)
        self.chk_spread = QCheckBox("Velocidad de Propagación (Rate of Spread)")
        
        outputs_layout.addWidget(self.chk_arrival)
        outputs_layout.addWidget(self.chk_intensity)
        outputs_layout.addWidget(self.chk_flame)
        outputs_layout.addWidget(self.chk_spread)
        outputs_layout.addStretch()
        
        self.tabs.addTab(self.tab_outputs, "Resultados")

        main_layout.addWidget(self.tabs)

        # --- BOTONES ---
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)
        # Asegura que el layout principal se asigne correctamente
        self.setLayout(main_layout)

    def set_lidar_from_layer(self, idx):
        if idx > 0 and idx-1 < len(self.lidar_layers):
            layer = self.lidar_layers[idx-1]
            # Para QgsPointCloudLayer, source() da la ruta real
            self.input_edit.setText(layer.source())


    def select_lidar(self):
        f, _ = QFileDialog.getOpenFileName(self, "LiDAR", "", "LiDAR (*.las *.laz)")
        if f:
            self.input_edit.setText(f)
            # Si coincide con una capa cargada, selecciona en el combo
            for i, layer in enumerate(self.lidar_layers):
                if hasattr(layer, 'source') and layer.source() == f:
                    self.lidar_combo.setCurrentIndex(i+1)
                    break

    def select_exe(self):
        f, _ = QFileDialog.getOpenFileName(self, "FlamMap EXE", "C:/Program Files", "EXE (*.exe)")
        if f: 
            self.exe_edit.setText(f)
            self.settings.setValue("MyLiDAR/flammap_exe", f)

    def accept(self):
        if not self.input_edit.text() or not self.exe_edit.text():
            QMessageBox.warning(self, "Error", "Faltan archivos obligatorios.")
            return
        super().accept()

    def get_parameters(self):
        return {
            "lidar_path": self.input_edit.text(),
            "exe_path": self.exe_edit.text(),
            # Clima
            "wind_spd": self.wind_speed.value(),
            "wind_dir": self.wind_dir.value(),
            "duration": self.duration.value(),
            "foliar_moist": self.foliar_moisture.value(),
            # Física
            "resolution": self.resolution.value(),
            "crown_method": self.crown_method.currentData(), # 0 o 1
            "spot_prob": self.spot_prob.value(),
            # Outputs (Booleanos)
            "out_arrival": self.chk_arrival.isChecked(),
            "out_intensity": self.chk_intensity.isChecked(),
            "out_flame": self.chk_flame.isChecked(),
            "out_spread": self.chk_spread.isChecked()
        }