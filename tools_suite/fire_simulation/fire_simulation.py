import os
import shutil
import time
from qgis.core import (QgsTask, QgsMessageLog, Qgis)
from qgis.PyQt.QtWidgets import QMessageBox
from osgeo import gdal
import laspy 

from ..flammap_export.flammap_export import FlamMapExportTask

class FireSimulationRunner(QgsTask):

    def __init__(self, params, parent_iface):
        super().__init__("Preparar Archivos FlamMap", QgsTask.CanCancel)
        self.params = params
        self.iface = parent_iface
        self.exception = None
        
        # --- CAMBIO: USAR CARPETA PÚBLICA (Evita bloqueos de permisos) ---
        self.safe_folder = "C:\\Users\\Public\\FlamMap_Temp"
        self.dummy_tif_path = os.path.join(self.safe_folder, "export_raw.tif")

    def run(self):
        try:
            QgsMessageLog.logMessage("=== MODO FINAL: RUTA ABSOLUTA DEL INPUT ===", "MyLiDAR", Qgis.Info)
            
            # 1. Limpieza (Borrar carpeta previa si existe)
            if os.path.exists(self.safe_folder):
                try: shutil.rmtree(self.safe_folder)
                except: pass
            
            # Crear carpeta limpia
            time.sleep(0.5)
            os.makedirs(self.safe_folder, exist_ok=True)
            self.setProgress(10)

            # --- Paso 1: Centro ---
            ign_point = []
            try:
                with laspy.open(self.params['lidar_path']) as las:
                    header = las.header
                    cx = (header.x_min + header.x_max) / 2
                    cy = (header.y_min + header.y_max) / 2
                    ign_point = [(cx, cy)]
            except: pass

            # --- Paso 2: Exportar ---
            exporter = FlamMapExportTask(
                description="Exportando Landscape",
                input_filename=self.params['lidar_path'],
                output_filename=self.dummy_tif_path,
                resolution=self.params['resolution'],
                parent=None,
                translator=lambda s: s,
                generate_scenarios=True,
                ignition_points=ign_point,
                generate_only_scenarios=False
            )
            exporter.output_folder = self.safe_folder 
            if not exporter.run(): raise Exception("Fallo al exportar LiDAR")
            self.setProgress(50)

            # --- Paso 3: GENERAR LCP ---
            # Buscar bandas TIF generadas
            bands_order = ["_elevation", "_slope", "_aspect", "_fuel_model", 
                           "_canopy_cover", "_stand_height", "_canopy_base_height", "_canopy_bulk_density"]
            tif_list = []
            
            # Buscar recursivamente en la carpeta segura
            for b in bands_order:
                found = None
                for root, dirs, files in os.walk(self.safe_folder):
                    for f in files:
                        if f.endswith(b + ".tif"):
                            found = os.path.join(root, f)
                            break
                    if found: break
                
                if found: tif_list.append(found)
                else: raise Exception(f"No encuentro la banda {b}")

            # Crear LCP
            vrt_opts = gdal.BuildVRTOptions(separate=True)
            vrt = gdal.BuildVRT("mem", tif_list, options=vrt_opts)
            
            final_lcp_name = "landscape.lcp"
            final_lcp_path = os.path.join(self.safe_folder, final_lcp_name)
            
            gdal.Translate(final_lcp_path, vrt, format="LCP")
            vrt = None
            self.setProgress(70)

            # --- Paso 4: FMS y SHP ---
            # Buscar FMS y copiar como 'fuel.fms'
            fms_path = None
            for root, dirs, files in os.walk(self.safe_folder):
                for f in files:
                    if f.endswith(".fms"): fms_path = os.path.join(root, f); break
            
            if not fms_path: # Fallback
                fms_path = os.path.join(self.safe_folder, "generated.fms")
                with open(fms_path, "w") as f:
                    f.write("ENGLISH\n")
                    for i in range(1, 200): f.write(f"{i} 3 4 5 60 100\n")
            
            shutil.copy2(fms_path, os.path.join(self.safe_folder, "fuel.fms"))

            # Buscar SHP y copiar como 'ign.shp'
            shp_path = None
            for root, dirs, files in os.walk(self.safe_folder):
                for f in files:
                    if f.endswith(".shp") and "ignition" in f.lower(): 
                        shp_path = os.path.join(root, f); break
            
            if not shp_path: raise Exception("Falta SHP de ignición")
            
            base_src = os.path.splitext(shp_path)[0]
            # Copiar extensiones vitales
            for ext in ['.shp', '.shx', '.dbf', '.prj']:
                src = base_src + ext
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(self.safe_folder, "ign" + ext))

            # --- Paso 5: INPUTS.TXT ---
            inputs_filename = "run_inputs.txt" # Nombre claro
            inputs_full_path = os.path.join(self.safe_folder, inputs_filename)
            
            with open(inputs_full_path, "w", encoding="ascii") as f:
                f.write("InputsFile\n")
                # Referencias locales simples
                f.write(f"LandscapeFile: landscape.lcp\n")
                f.write(f"FuelMoistureFile: fuel.fms\n")
                f.write(f"IgnitionFile: ign.shp\n")
                f.write(f"OutputFile: simulacion\n")
                
                f.write(f"WindSpeed: {int(self.params['wind_spd'])}\n")
                f.write(f"WindDirection: {int(self.params['wind_dir'])}\n")
                f.write(f"FoliarMoistureContent: {int(self.params['foliar_moist'])}\n")
                f.write(f"MaxSimTime: {int(self.params['duration'])}\n")
                f.write(f"Resolution: {self.params['resolution']}\n")
                
                method = "ScottReinhardt2001" if self.params['crown_method'] == 1 else "Finney1998"
                f.write(f"CrownFireMethod: {method}\n")
                f.write(f"SpotProbability: {self.params['spot_prob']}\n")
                
                f.write("Arrival Time Grid: arrival_time\n")
                f.write("Fireline Intensity Grid: intensity\n")

            self.setProgress(90)
            
            # --- Paso 6: BAT CON RUTA DE INPUT ABSOLUTA ---
            bat_path = os.path.join(self.safe_folder, "GO.bat")
            exe_clean = self.params['exe_path'] 

            with open(bat_path, "w", encoding="mbcs") as bat:
                bat.write("@echo off\n")
                bat.write("cls\n")
                bat.write("color 0A\n")
                
                # 1. Ir a la carpeta publica
                bat.write(f"cd /d \"{self.safe_folder}\"\n")
                
                bat.write("echo ==================================================\n")
                bat.write("echo    EJECUCION CON RUTA ABSOLUTA DE INPUT\n")
                bat.write("echo ==================================================\n\n")
                
                bat.write("echo Verificando inputs...\n")
                bat.write(f"if exist {inputs_filename} (echo [OK] Archivo inputs existe) else (echo [FAIL] No existe)\n")
                bat.write("echo.\n")
                
                bat.write("echo Ejecutando FlamMap...\n")
                bat.write("echo NOTA: Si esto falla, es problema de tu ejecutable FConstMTT.\n")
                bat.write("echo --------------------------------------------------\n")
                
                # --- AQUÍ ESTÁ LA MAGIA: LE PASAMOS LA RUTA COMPLETA AL ARCHIVO DE TEXTO ---
                # Esto obliga al ejecutable a encontrarlo sí o sí.
                bat.write(f'"{exe_clean}" "{inputs_full_path}"\n')
                # --------------------------------------------------------------------------
                
                bat.write("\n")
                bat.write("if %ERRORLEVEL% NEQ 0 (\n")
                bat.write("   color 4C\n")
                bat.write("   echo ERROR CRITICO.\n")
                bat.write(") else (\n")
                bat.write("   color 20\n")
                bat.write("   echo EXITO!!! Abre QGIS y carga los resultados.\n")
                bat.write(")\n")
                bat.write("cmd /k\n") 

            self.setProgress(100)
            return True

        except Exception as e:
            self.exception = e
            return False

    def finished(self, result):
        if result:
            QMessageBox.information(self.iface.mainWindow(), "Listo", 
                "Prueba en C:\\Users\\Public\\FlamMap_Temp\n\n"
                "Ejecuta el archivo GO.bat")
            try: os.startfile(self.safe_folder)
            except: pass
        else:
            QMessageBox.critical(self.iface.mainWindow(), "Error", str(self.exception))