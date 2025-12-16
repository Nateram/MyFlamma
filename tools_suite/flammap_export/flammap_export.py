# --- General imports ---
import os
import random
import laspy
from laspy import LazBackend
import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.interpolate import griddata

# --- QGIS and PyQt imports ---
from qgis.PyQt.QtWidgets import QMessageBox, QDialog
from qgis.core import QgsApplication, Qgis, QgsMessageLog, QgsTask

# --- Method-specific imports ---
from osgeo import gdal, osr, ogr
from scipy.spatial import cKDTree

# --- Dialog imports ---
from .flammap_export_dialog import FlamMapExportDialog

# ----------------------
# --- FlamMap Export ---
# --------------------------------------------------------------------------------------------
# Description:
# This function exports LiDAR point cloud data to a multi-band GeoTIFF compatible with FlamMap.
# Generates separate bands for Elevation, Slope, Aspect, Fuel Model, Canopy Cover, etc.
# Also generates automatic simulation scenarios (Ignition points and Weather files).
# --------------------------------------------------------------------------------------------

class FlamMapExportTask(QgsTask):
    """Export LiDAR to FlamMap GeoTIFF in a background thread"""

    def __init__(self, description, input_filename, output_filename, resolution, parent, translator, generate_scenarios=False, ignition_points=None, generate_only_scenarios=False):
        super().__init__(description, QgsTask.CanCancel)
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.resolution = resolution
        self.generate_scenarios = generate_scenarios
        self.ignition_points = ignition_points if ignition_points else []
        self.generate_only_scenarios = generate_only_scenarios
        self.should_generate_scenarios = self.generate_scenarios or self.generate_only_scenarios or (self.ignition_points and len(self.ignition_points) > 0)
        self.parent = parent
        self.exception = None
        self.tr = translator

    def run(self):
        try:
            # Step 1: Read input file
            las = laspy.read(self.input_filename, laz_backend=LazBackend.Lazrs)
            self.setProgress(10)

            # --- PREPARAR PROYECCIÓN (CRS) AL PRINCIPIO ---
            # Es vital tener esto definido antes para usarlo en TIFFs y SHPs
            srs = osr.SpatialReference()
            wkt_projection = None
            try:
                # Intentar leer del LAS
                crs = las.header.parse_crs()
                srs.ImportFromWkt(crs.to_wkt())
                wkt_projection = srs.ExportToWkt()
            except:
                try:
                    # Intentar fallback EPSG
                    srs.ImportFromEPSG(crs.to_epsg())
                    wkt_projection = srs.ExportToWkt()
                except:
                    # PLAN B: Si falla, forzamos ETRS89 UTM 30N (Estándar España PNOA)
                    srs.ImportFromEPSG(25830)
                    wkt_projection = srs.ExportToWkt()

            # Extract data
            x = las.x
            y = las.y
            z = las.z
            classification = las.classification

            # Define grid
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            x_bins = np.arange(x_min, x_max + self.resolution, self.resolution)
            y_bins = np.arange(y_min, y_max + self.resolution, self.resolution)
            
            # Create grid centers (useful for interpolation)
            x_centers = (x_bins[:-1] + x_bins[1:]) / 2
            y_centers = (y_bins[:-1] + y_bins[1:]) / 2
            X, Y = np.meshgrid(x_centers, y_centers)
            
            self.setProgress(20)

            # Función auxiliar para corregir orientación de matrices de binned_statistic_2d
            # scipy.stats.binned_statistic_2d devuelve (nx, ny) pero necesitamos (rows, cols) = (ny, nx)
            # Además, el sistema de imagen empieza arriba-izquierda, bins abajo-izquierda
            def correct_matrix_orientation(matrix):
                return np.flipud(matrix.T)

            # Step 2: Elevation (Band 1) - Min Z of ground points (class 2)
            ground_mask = classification == 2
            if np.any(ground_mask):
                elev_stat = binned_statistic_2d(x[ground_mask], y[ground_mask], z[ground_mask], statistic='min', bins=[x_bins, y_bins])
                elevation = correct_matrix_orientation(elev_stat.statistic)
            else:
                # Fallback to min of any points if no ground class
                elev_stat = binned_statistic_2d(x, y, z, statistic='min', bins=[x_bins, y_bins])
                elevation = correct_matrix_orientation(elev_stat.statistic)

            # Fill NaNs with interpolation
            valid_mask = ~np.isnan(elevation)
            if not np.all(valid_mask):
                # Points for interpolation - Y debe estar volteada para coincidir con elevation corregida
                Y_corrected = np.flipud(Y)
                valid_x = X[valid_mask]
                valid_y = Y_corrected[valid_mask]
                valid_z = elevation[valid_mask]
                
                if len(valid_z) > 0:
                     elevation = griddata((valid_x, valid_y), valid_z, (X, Y_corrected), method='nearest')
            
            # Get grid dimensions for later use
            rows, cols = elevation.shape
            
            self.setProgress(30)

            # Step 3: Slope (Band 2) - in degrees
            dy, dx = np.gradient(elevation, self.resolution)
            slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)
            slope = np.clip(slope, 0, 90)
            self.setProgress(40)

            # Step 4: Aspect (Band 3) - in degrees
            aspect = np.arctan2(-dx, dy) * (180 / np.pi)
            aspect = (aspect + 360) % 360
            self.setProgress(50)

            # Step 5: Precise Height Above Ground (HAG) using KDTree
            ground_mask = classification == 2
            veg_mask = (classification >= 3) & (classification <= 5)
            hag_results = []
            if np.any(ground_mask) and np.any(veg_mask):
                # Create KDTree with ground points
                ground_xy = np.vstack((x[ground_mask], y[ground_mask])).T
                ground_z = z[ground_mask]
                tree = cKDTree(ground_xy)
                
                # Vegetation points
                veg_xy = np.vstack((x[veg_mask], y[veg_mask])).T
                veg_z = z[veg_mask]
                
                # Batch processing for memory management
                batch_size = 50000
                n_veg = len(veg_xy)
                hag_results = np.empty(n_veg, dtype=np.float32)
                
                for i in range(0, n_veg, batch_size):
                    if self.isCanceled():
                        return False
                    j = min(i + batch_size, n_veg)
                    batch_xy = veg_xy[i:j]
                    batch_z = veg_z[i:j]
                    
                    # Find nearest ground neighbor
                    distances, indices = tree.query(batch_xy, k=1)
                    nearest_ground_z = ground_z[indices]
                    
                    # Calculate HAG
                    hag_batch = batch_z - nearest_ground_z
                    hag_batch = np.maximum(hag_batch, 0)  # Ensure non-negative
                    hag_results[i:j] = hag_batch
                    
                    # Update progress (50–70%)
                    progress = 50 + (20 * j / n_veg)
                    self.setProgress(progress)
            else:
                hag_results = np.array([], dtype=np.float32)
            
            self.setProgress(70)
            
            # Rasterize HAG results
            veg_height_grid = np.zeros_like(elevation)
            if len(hag_results) > 0:
                vh_stat = binned_statistic_2d(x[veg_mask], y[veg_mask], hag_results, statistic='max', bins=[x_bins, y_bins])
                veg_height_grid = correct_matrix_orientation(np.nan_to_num(vh_stat.statistic, nan=0))
            
            # Fuel Model (Band 4) - Asignación basada en Clases 3, 4, 5, 6, 9
            fuel_model = np.full_like(elevation, 9, dtype=np.uint8)  # Fondo: 9 (Litter)

            # --- CORRECCIÓN: AÑADIR MÁSCARA PARA CLASE 5 ---
            class_3_mask = classification == 3  # Veg Baja
            class_4_mask = classification == 4  # Veg Media
            class_5_mask = classification == 5  # Veg Alta
            class_6_mask = classification == 6  # Edificios
            class_9_mask = classification == 9  # Agua

            # 1. Vegetación Baja (Clase 3) -> Modelo 1 (Pasto Corto)
            if np.any(class_3_mask):
                class_3_count = binned_statistic_2d(x[class_3_mask], y[class_3_mask], None, statistic='count', bins=[x_bins, y_bins])
                class_3_grid = correct_matrix_orientation(class_3_count.statistic)
                fuel_model[class_3_grid > 0] = 1

            # 2. Vegetación Media (Clase 4) -> Modelo 4 (Matorral/Chaparral)
            if np.any(class_4_mask):
                class_4_count = binned_statistic_2d(x[class_4_mask], y[class_4_mask], None, statistic='count', bins=[x_bins, y_bins])
                class_4_grid = correct_matrix_orientation(class_4_count.statistic)
                fuel_model[class_4_grid > 0] = 4

            # 3. Vegetación Alta (Clase 5) -> Modelo 8 (Bosque/Timber Litter)
            if np.any(class_5_mask):
                class_5_count = binned_statistic_2d(x[class_5_mask], y[class_5_mask], None, statistic='count', bins=[x_bins, y_bins])
                class_5_grid = correct_matrix_orientation(class_5_count.statistic)
                fuel_model[class_5_grid > 0] = 8

            # 4. Edificios (Clase 6) -> Incombustible (99)
            if np.any(class_6_mask):
                class_6_count = binned_statistic_2d(x[class_6_mask], y[class_6_mask], None, statistic='count', bins=[x_bins, y_bins])
                class_6_grid = correct_matrix_orientation(class_6_count.statistic)
                fuel_model[class_6_grid > 0] = 99

            # 5. Agua (Clase 9) -> Incombustible (99)
            if np.any(class_9_mask):
                class_9_count = binned_statistic_2d(x[class_9_mask], y[class_9_mask], None, statistic='count', bins=[x_bins, y_bins])
                class_9_grid = correct_matrix_orientation(class_9_count.statistic)
                fuel_model[class_9_grid > 0] = 99

            # Corregir zonas sin datos de elevación
            fuel_model[np.isnan(elevation)] = 99
            
            self.setProgress(75)
            
            # Stand Height (Band 6) - Max height
            stand_height = veg_height_grid
            
            # Canopy Base Height (CBH, Band 7) - 20th percentile
            def p20(v): return np.percentile(v, 20) if len(v) > 0 else 0
            cbh = np.zeros_like(elevation)
            if len(hag_results) > 0:
                cbh_stat = binned_statistic_2d(x[veg_mask], y[veg_mask], hag_results, statistic=p20, bins=[x_bins, y_bins])
                cbh = correct_matrix_orientation(np.nan_to_num(cbh_stat.statistic, nan=0))
            
            self.setProgress(80)

            # Canopy Cover (Band 5)
            # Count veg points per cell
            if np.any(veg_mask):
                veg_count_stat = binned_statistic_2d(x[veg_mask], y[veg_mask], None, statistic='count', bins=[x_bins, y_bins])
                veg_count = correct_matrix_orientation(np.nan_to_num(veg_count_stat.statistic, nan=0))
            else:
                veg_count = np.zeros_like(elevation)

            # Count total points per cell
            total_count_stat = binned_statistic_2d(x, y, None, statistic='count', bins=[x_bins, y_bins])
            total_count = correct_matrix_orientation(np.nan_to_num(total_count_stat.statistic, nan=0))

            canopy_cover = np.zeros_like(elevation)
            with np.errstate(divide='ignore', invalid='ignore'):
                canopy_cover = (veg_count / total_count) * 100
                canopy_cover = np.nan_to_num(canopy_cover, nan=0)
            canopy_cover = np.clip(canopy_cover, 0, 100)
            
            self.setProgress(85)

            # Canopy Bulk Density (CBD, Band 8) - Simple estimator
            # CBD ~ (Cover/100) * BiomasFactor
            cbd = (canopy_cover / 100.0) * 0.15 # 0.15 is a generic factor for conifers
            cbd = np.nan_to_num(cbd, nan=0)
            
            self.setProgress(90)

            # Prepare output directory
            base_name = os.path.splitext(self.output_filename)[0]
            output_dir = os.path.dirname(self.output_filename)
            flammap_dir = os.path.join(output_dir, "flammap_output")
            os.makedirs(flammap_dir, exist_ok=True)

            if not os.access(flammap_dir, os.W_OK):
                raise Exception(f"No write permission in directory: {flammap_dir}")

            # Step 8: Create individual GeoTIFF files
            if not self.generate_only_scenarios:
                driver = gdal.GetDriverByName('GTiff')

                # Set geotransform
                geotransform = (x_min, self.resolution, 0, y_max, 0, -self.resolution)

                # List of (data, suffix, gdal_type)
                files_data = [
                    (elevation, "_elevation", gdal.GDT_Float32),
                    (slope, "_slope", gdal.GDT_Float32),
                    (aspect, "_aspect", gdal.GDT_Float32),
                    (fuel_model.astype(np.uint16), "_fuel_model", gdal.GDT_UInt16),  # Guardar como entero
                    (canopy_cover, "_canopy_cover", gdal.GDT_Float32),
                    (stand_height, "_stand_height", gdal.GDT_Float32),
                    (cbh, "_canopy_base_height", gdal.GDT_Float32),
                    (cbd, "_canopy_bulk_density", gdal.GDT_Float32)
                ]

                # --- BUCLE DE CREACIÓN DE TIFFS ---
                for data, suffix, gdal_type in files_data:
                    filename = os.path.normpath(os.path.join(flammap_dir, os.path.basename(base_name) + suffix + ".tif"))
                    
                    if os.path.exists(filename):
                        try:
                            os.remove(filename)
                        except OSError:
                            pass # File locked? Try saving anyway

                    try:
                        dataset = driver.Create(filename, cols, rows, 1, gdal_type)
                        if dataset is None:
                            continue
                        
                        dataset.SetGeoTransform(geotransform)
                        
                        if wkt_projection:
                            dataset.SetProjection(wkt_projection)

                        rb = dataset.GetRasterBand(1)
                        # Replace NaN with NoData
                        data = np.nan_to_num(data, nan=-9999)
                        rb.WriteArray(data)
                        rb.SetDescription(suffix[1:].replace('_', ' ').title())
                        rb.SetNoDataValue(-9999)

                        dataset.FlushCache()
                        dataset = None
                    except Exception as e:
                        raise Exception(f"Error creating {filename}: {str(e)}")

                # --- CREACIÓN DEL PRJ ÚNICO ---
                if wkt_projection:
                    prj_filename = os.path.join(flammap_dir, os.path.basename(base_name) + ".prj")
                    try:
                        with open(prj_filename, 'w') as prj_file:
                            prj_file.write(wkt_projection)
                    except:
                        pass

            # --- GENERACIÓN DE ESCENARIOS (Código Actualizado) ---
            if self.should_generate_scenarios:
                scenarios_dir = os.path.join(flammap_dir, "scenarios")
                os.makedirs(scenarios_dir, exist_ok=True)

                # 1. Definición de Escenarios de Clima
                fms_data = {
                    "dry": { "1h": 3, "10h": 4, "100h": 5, "Herb": 30, "Woody": 60 },
                    "moderate": { "1h": 6, "10h": 7, "100h": 8, "Herb": 60, "Woody": 90 },
                    "wet": { "1h": 10, "10h": 12, "100h": 15, "Herb": 90, "Woody": 120 }
                }
                
                # Lista de modelos presentes (Asegurar compatibilidad)
                models_present = [1, 4, 8, 9, 98, 99] 

                # 2. Generar Archivos
                for name, values in fms_data.items():
                    # --- A. Archivo .FMS (Tabla numérica estricta) ---
                    fms_filename = f"{name}.fms"
                    fms_path = os.path.join(scenarios_dir, fms_filename)
                    
                    with open(fms_path, 'w') as f:
                        # FlamMap exige 6 columnas numéricas sin cabeceras de texto
                        for model in models_present:
                            f.write(f"{model} {values['1h']} {values['10h']} {values['100h']} {values['Herb']} {values['Woody']}\n")

                    # --- B. Archivo de Inputs para Importar (.TXT) ---
                    # Este archivo se puede cargar en FlamMap > Runs > Import
                    inputs_filename = f"run_inputs_{name}.txt"
                    inputs_path = os.path.join(scenarios_dir, inputs_filename)
                    
                    # Definimos vientos típicos para cada escenario
                    wind_speed = 30 if name == "dry" else (15 if name == "moderate" else 5)
                    wind_dir = 225 # Suroeste constante (puedes variarlo si quieres)
                    
                    with open(inputs_path, 'w') as f:
                        # Formato Clave-Valor que suele aceptar el importador
                        f.write(f"FUEL_MOISTURE_FILE: {fms_filename}\n")
                        f.write(f"WIND_SPEED: {wind_speed}\n")
                        f.write(f"WIND_DIRECTION: {wind_dir}\n")
                        f.write(f"FOLIAR_MOISTURE_CONTENT: 100\n")
                        f.write(f"CROWN_FIRE_METHOD: ScottReinhardt2001\n") # Intento de forzar el método

                # 3. Puntos de Ignición (Shapefiles)
                if self.ignition_points:
                    # Usar puntos seleccionados manualmente
                    points_to_use = self.ignition_points
                elif np.any(veg_mask):
                    # Generar aleatorios si no hay manuales
                    veg_points_x = x[veg_mask]
                    veg_points_y = y[veg_mask]
                    count_veg = len(veg_points_x)
                    
                    if count_veg >= 3:
                        selected_indices = random.sample(range(count_veg), 3)
                        points_to_use = [(veg_points_x[idx], veg_points_y[idx]) for idx in selected_indices]
                    else:
                        points_to_use = []
                else:
                    points_to_use = []
                
                for i, (px, py) in enumerate(points_to_use, 1):
                    shp_filename = os.path.join(scenarios_dir, f"ignition_{i}.shp")
                    
                    driver_shp = ogr.GetDriverByName('ESRI Shapefile')
                    if os.path.exists(shp_filename):
                        driver_shp.DeleteDataSource(shp_filename)
                    
                    ds = driver_shp.CreateDataSource(shp_filename)
                    if ds is None: continue
                    
                    layer = ds.CreateLayer('ignition', srs, ogr.wkbPoint)
                    feature = ogr.Feature(layer.GetLayerDefn())
                    point = ogr.Geometry(ogr.wkbPoint)
                    point.AddPoint(px, py)
                    feature.SetGeometry(point)
                    layer.CreateFeature(feature)
                    
                    feature = None
                    ds = None
            
            self.setProgress(100)
            return True

        except Exception as e:
            self.exception = e
            return False

    def finished(self, result):
        if result:
            if self.generate_only_scenarios:
                msg = f"{self.tr('Scenarios generated in')}:\n{os.path.join(os.path.dirname(self.output_filename), 'flammap_output', 'scenarios')}"
            else:
                msg = f"{self.tr('GeoTIFF files saved to')}:\n{os.path.join(os.path.dirname(self.output_filename), 'flammap_output')}"
                if self.should_generate_scenarios:
                     msg += f"\n\n{self.tr('Scenarios generated in')}:\n.../flammap_output/scenarios/"
            
            QMessageBox.information(
                self.parent.iface.mainWindow(),
                self.tr("FlamMap Export Complete"),
                msg
            )
        else:
            msg = f"{self.tr('An error occurred')}: {self.exception}" if self.exception else self.tr("Export failed")
            QgsMessageLog.logMessage(msg, "MyLiDAR", Qgis.Critical)
            QMessageBox.critical(self.parent.iface.mainWindow(), self.tr("Error Exporting to FlamMap"), msg)

        if self in self.parent.running_tasks:
            self.parent.running_tasks.remove(self)

# ----------------------------------
# --- Main FlamMap Export Method ---
# ----------------------------------

def flammap_export(self):
    # Step 1: Select input/output and parameters via dialog
    dialog = FlamMapExportDialog(self.iface.mainWindow(), translator=self.tr, iface=self.iface)
    if dialog.exec_() != QDialog.Accepted:
        return

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
    ignition_points = dialog.get_ignition_points()
    generate_only_scenarios = dialog.get_generate_only_scenarios()

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