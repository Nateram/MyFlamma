import importlib
import subprocess
import sys
import os

def ensure_dependencies():
    packages = {
        "laspy": "laspy",
        "scipy": "scipy",
        "osgeo.gdal": "gdal",
        "osgeo.osr": "gdal",
        "matplotlib": "matplotlib",
        "reportlab": "reportlab",
        "sklearn": "scikit-learn",
        "shapely": "shapely"
    }

    python_exe = sys.executable
    if not os.path.basename(python_exe).lower().startswith("python"):
        python_exe = "python"

    for mod_name, pip_name in packages.items():
        try:
            importlib.import_module(mod_name)
        except ImportError:
            try:
                subprocess.check_call([python_exe, "-m", "pip", "install", pip_name])
            except Exception as e:
                from qgis.PyQt.QtWidgets import QMessageBox
                QMessageBox.critical(None, self.tr("Dependency installation failed"), self.tr("Please check your internet connection and try again."))
                raise

def classFactory(iface):
    ensure_dependencies()
    from .myFlamma import MyFlammaPlugin
    return MyFlammaPlugin(iface)
