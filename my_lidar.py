# --- General imports ---
import os
import json

# --- QGIS and PyQt imports ---
from qgis.PyQt.QtCore import QCoreApplication, QLocale, QSettings
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox, QMenu
from qgis.core import Qgis, QgsApplication

# --- Method-specific imports ---
from .tools_suite.vegetation_classification.vegetation_classification_chm import classify_vegetation
from .tools_suite.flammap_export.flammap_export import FlamMapExportDialog, FlamMapExportTask
from .tools_suite.obj_export.obj_export import export_to_obj

# -----------------------------
# --- My LiDAR Plugin Class ---
# -----------------------------
class MyFlammaPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.running_tasks = []
        self.last_classification_data = None  # Para export OBJ

        self.translations = {}
        system_lang = QLocale.system().name()[:2] # Detect system language ("en", "es")
        self.current_lang = system_lang if system_lang else "en"
        self.load_language(self.current_lang)

        self.ignition_points = []
        self.settings = QSettings()
        self.ignition_points = self.settings.value("MyFlamma/ignition_points", [])

    def load_language(self, lang_code: str):
        lang_file = os.path.join(self.plugin_dir, "translations", f"{lang_code}.json")
        if os.path.exists(lang_file):
            with open(lang_file, "r", encoding="utf-8") as f:
                self.translations = json.load(f)
        else:
            self.translations = {}

    def tr(self, message: str) -> str:
        return self.translations.get(message, QCoreApplication.translate('MyFlamma', message))

    def initGui(self):
        main_win = self.iface.mainWindow()
        self.menu = QMenu("MyFlamma", main_win)

        menubar = main_win.menuBar()
        help_menu = None
        for act in menubar.actions():
            if act.text().replace("&", "").lower() == "help":
                help_menu = act
                break

        if help_menu:
            menubar.insertMenu(help_menu, self.menu)
        else:
            menubar.addMenu(self.menu)

        actions = [
            ("vegetation.png", self.tr("Classify Data"), self.vegetation_classification_chm),
            ("fire.png", self.tr("Export to FlamMap"), self.export_flammap),
            ("vegetation.png", self.tr("Export to OBJ"), self.export_obj),
        ]

        self.actions = []
        for icon_file, label, callback in actions:
            icon_path = os.path.join(self.plugin_dir, 'icons', icon_file)
            action = QAction(QIcon(icon_path), self.tr(label), main_win)
            action.triggered.connect(callback)
            self.menu.addAction(action)
            self.actions.append(action)

    # --- Vegetation Classification (CHM Method) ---
    def vegetation_classification_chm(self):
        classify_vegetation(self)

    # --- OBJ Export ---
    def export_obj(self):
        export_to_obj(self)

    # --- FlamMap Export ---
    def export_flammap(self):
        dialog = FlamMapExportDialog(self.iface.mainWindow(), translator=self.tr, iface=self.iface, ignition_points_list=self.ignition_points)
        dialog.accepted.connect(lambda: self.run_flammap_export(dialog))
        dialog.show()

    def run_flammap_export(self, dialog):
        input_filename, output_filename = dialog.get_input_output()
        if not input_filename:
            QMessageBox.warning(
                self.iface.mainWindow(),
                self.tr("No Input Selected"),
                self.tr("Please select a LiDAR layer or file.")
            )
            return

        if not output_filename:
            QMessageBox.warning(
                self.iface.mainWindow(),
                self.tr("No Output Selected"),
                self.tr("Please specify an output file path.")
            )
            return

        resolution = dialog.get_resolution()
        generate_scenarios = dialog.get_generate_scenarios()
        generate_only_scenarios = dialog.get_generate_only_scenarios()
        ignition_points = dialog.get_ignition_points()

        # Step 2: Create and run the background task
        task_description = f"{self.tr('Exporting to FlamMap GeoTIFF from')} {os.path.basename(input_filename)}"
        task = FlamMapExportTask(task_description, input_filename, output_filename, resolution, self, self.tr, generate_scenarios, ignition_points, generate_only_scenarios)
        self.running_tasks.append(task)
        QgsApplication.taskManager().addTask(task)

        self.iface.messageBar().pushMessage(
            self.tr("Task Started"),
            self.tr("Exporting to FlamMap GeoTIFF in the background"),
            level=Qgis.Info,
            duration=-1
        )

    def unload(self):
        if self.menu:
            self.iface.mainWindow().menuBar().removeAction(self.menu.menuAction())
        self.menu = None
        self.actions = []
