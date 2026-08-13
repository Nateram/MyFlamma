import os
import numpy as np
import traceback
import csv
import datetime

# --- QGIS y PyQt ---
from qgis.core import (
    QgsTask, QgsApplication, Qgis, QgsMessageLog, QgsProject,
    QgsVectorLayer, QgsFeature, QgsGeometry, QgsPointXY, 
    QgsField, QgsCategorizedSymbolRenderer, QgsSymbol, QgsRendererCategory,
    QgsPointCloudLayer
)
from qgis.PyQt.QtWidgets import QMessageBox, QDialog
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor

# --- Librerías Científicas ---
import laspy
from laspy import LazBackend
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

# --- Tu Diálogo ---
from .vegetation_classification_dialog import VegetationClassificationDialog

# =================================================================================
# --- TAREA PRINCIPAL: CHM + GAUSSIAN + PEAK MERGING + METRICAS ---
# =================================================================================

class VegetationClassificationCHMTask(QgsTask):
    """
    Tarea en segundo plano que detecta árboles individuales simulando visión humana (CHM),
    fusiona picos múltiples y extrae métricas forestales (altura, diámetro, color).
    """

    def __init__(self, description, filename, output_path, low_thresh, high_thresh, parent, translator, size_threshold=2.0, export_trees=True, export_shrubs=True, export_grass=False, export_buildings=True, export_reclassified=False, csv_path=None, csv_shrubs_path=None):
        super().__init__(description, QgsTask.CanCancel)
        self.filename = filename
        self.output_path = output_path
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.size_threshold = size_threshold
        self.export_trees = export_trees
        self.export_shrubs = export_shrubs
        self.export_grass = export_grass
        self.export_buildings = export_buildings
        self.export_reclassified = export_reclassified
        self.parent = parent
        self.exception = None
        self.tr = translator
        self.visualization_data = []
        self.stats = {}
        self.epsg = None
        self.reclassified_laz_path = None  # Path del LAZ reclasificado
        
        self.csv_path = csv_path
        self.csv_shrubs_path = csv_shrubs_path

    def run(self):
        try:
            self.setProgress(0)
            
            # --- FUNCIÓN DE RADIO DINÁMICO PARA FUSIÓN ---
            def get_merge_radius(height):
                if height < self.size_threshold: return 1.5
                elif height < 10.0: return 3.5
                elif height < 18.0: return 5.0
                else: return 7.0
            
            def classify_tree_size_by_density(x, y, density_points_in_radius):
                """Clasifica el tamaño del árbol en 5 categorías basadas en densidad."""
                if density_points_in_radius is None or len(density_points_in_radius) == 0:
                    return "Medium", 0
                
                quantiles = [0, 20, 40, 60, 80, 100]
                percentiles = np.percentile(density_points_in_radius, quantiles)
                point_count = len(density_points_in_radius)
                
                if point_count <= percentiles[1]:
                    return "Very Small", point_count
                elif point_count <= percentiles[2]:
                    return "Small", point_count
                elif point_count <= percentiles[3]:
                    return "Medium", point_count
                elif point_count <= percentiles[4]:
                    return "Large", point_count
                else:
                    return "Very Large", point_count
            
            # --- PASO 1: Leer archivo LAS/LAZ ---
            las = laspy.read(self.filename, laz_backend=LazBackend.Lazrs)
            try:
                self.epsg = las.header.epsg if hasattr(las.header, 'epsg') else None
            except:
                self.epsg = None

            # --- PASO 2: Extraer datos ---
            x = np.array(las.x)
            y = np.array(las.y)
            z = np.array(las.z)
            classification = np.array(las.classification)
            
            # --- Extraer color si existe ---
            has_color = False
            red, green, blue = None, None, None
            try:
                if hasattr(las, 'red'):
                    has_color = True
                    r_raw = np.array(las.red)
                    g_raw = np.array(las.green)
                    b_raw = np.array(las.blue)
                    
                    if np.max(r_raw) > 255:
                        red = (r_raw / 256).astype(np.uint8)
                        green = (g_raw / 256).astype(np.uint8)
                        blue = (b_raw / 256).astype(np.uint8)
                    else:
                        red, green, blue = r_raw, g_raw, b_raw
            except:
                has_color = False

            # --- MÁSCARAS ---
            ground_mask = classification == 2
            # Vegetación alta para árboles (Clases 3, 5) + puntos altos
            trees_veg_mask = np.isin(classification, [3, 5]) 
            # Todos los puntos de vegetación para potenciales arbustos (Clases 3, 4, 5)
            all_veg_mask = np.isin(classification, [3, 4, 5])
            # Edificios (Clase 6) para filtrado
            building_mask = classification == 6

            if np.sum(ground_mask) == 0:
                raise ValueError(self.tr("No ground points (class 2) found"))

            ground_xyz = np.column_stack((x[ground_mask], y[ground_mask], z[ground_mask]))
            
            # Para árboles: clases 3, 5
            trees_indices_global = np.where(trees_veg_mask)[0]
            trees_x = x[trees_veg_mask]
            trees_y = y[trees_veg_mask]
            trees_z = z[trees_veg_mask]
            
            if has_color:
                trees_r = red[trees_veg_mask]
                trees_g = green[trees_veg_mask]
                trees_b = blue[trees_veg_mask]
            
            # Para arbustos: TODOS los puntos de vegetación
            shrubs_indices_global = np.where(all_veg_mask)[0]
            shrubs_x = x[all_veg_mask]
            shrubs_y = y[all_veg_mask]
            shrubs_z = z[all_veg_mask]
            
            if has_color:
                shrubs_r = red[all_veg_mask]
                shrubs_g = green[all_veg_mask]
                shrubs_b = blue[all_veg_mask]
            
            build_x = x[building_mask]
            build_y = y[building_mask]
            build_z = z[building_mask]
            has_buildings = len(build_x) > 0

            self.setProgress(20)

            # --- PASO 3: NORMALIZACIÓN DE ALTURA (ÁRBOLES) ---
            norm_h_trees = np.zeros(len(trees_x), dtype=np.float32)
            ground_tree = cKDTree(ground_xyz[:, :2])
            
            batch_size = 50000
            for i in range(0, len(trees_x), batch_size):
                if self.isCanceled(): return False
                end = min(i + batch_size, len(trees_x))
                query_xy = np.column_stack((trees_x[i:end], trees_y[i:end]))
                _, idxs = ground_tree.query(query_xy, k=1)
                norm_h_trees[i:end] = trees_z[i:end] - ground_xyz[idxs, 2]
                self.setProgress(20 + (10 * end / len(trees_x)))

            # Filtro de altura mínima del usuario
            high_mask = norm_h_trees > self.high_thresh
            final_x = trees_x[high_mask]
            final_y = trees_y[high_mask]
            final_h = norm_h_trees[high_mask]
            final_color_indices_trees = np.arange(len(trees_x))[high_mask]

            if len(final_x) == 0 and len(shrubs_x) == 0:
                self.visualization_data = []
                self.stats = {"num_trees": 0, "num_shrubs": 0}
                return True
                self.stats = {"num_tree": 0, "num_high_orig": len(veg_x)}
                return True

            # --- PASO 4: RASTERIZACIÓN CHM ---
            self.setProgress(50)
            pixel_size = 0.5
            
            min_x, max_x = np.min(final_x), np.max(final_x)
            min_y, max_y = np.min(final_y), np.max(final_y)
            
            cols = int((max_x - min_x) / pixel_size) + 1
            rows = int((max_y - min_y) / pixel_size) + 1
            
            chm_grid = np.zeros((rows, cols), dtype=np.float32)
            idx_col = ((final_x - min_x) / pixel_size).astype(int)
            idx_row = ((max_y - final_y) / pixel_size).astype(int)
            
            sort_idx = np.argsort(final_h)
            chm_grid[idx_row[sort_idx], idx_col[sort_idx]] = final_h[sort_idx]

            # --- PASO 5: SUAVIZADO GAUSSIANO ---
            sigma = 1.5  
            smoothed_chm = gaussian_filter(chm_grid, sigma=sigma)

            # --- PASO 6: DETECCIÓN DE PICOS ---
            window_size = 7 
            local_max = maximum_filter(smoothed_chm, size=window_size)
            peaks = (smoothed_chm == local_max) & (smoothed_chm > self.high_thresh)
            
            peak_rows, peak_cols = np.where(peaks)
            raw_peak_x = min_x + (peak_cols * pixel_size) + (pixel_size / 2)
            raw_peak_y = max_y - (peak_rows * pixel_size) - (pixel_size / 2)
            raw_peak_h = smoothed_chm[peak_rows, peak_cols]

            # --- PASO 7: FUSIÓN DE PICOS ---
            self.setProgress(70)
            
            sorted_peak_idx = np.argsort(-raw_peak_h)
            sorted_px = raw_peak_x[sorted_peak_idx]
            sorted_py = raw_peak_y[sorted_peak_idx]
            sorted_ph = raw_peak_h[sorted_peak_idx]
            
            peaks_tree = cKDTree(np.column_stack((sorted_px, sorted_py)))
            processed_peaks = np.zeros(len(sorted_px), dtype=bool)
            
            final_features = []
            
            b_tree = None
            if has_buildings:
                if len(build_x) > 200000:
                    idx_rnd = np.random.choice(len(build_x), 200000, replace=False)
                    b_tree = cKDTree(np.column_stack((build_x[idx_rnd], build_y[idx_rnd])))
                else:
                    b_tree = cKDTree(np.column_stack((build_x, build_y)))
            
            veg_tree = cKDTree(np.column_stack((final_x, final_y)))
            
            total_peaks = len(sorted_px)
            MIN_DIST_TO_BUILDING = 2.0
            
            all_densities = []
            for i in range(total_peaks):
                current_h = sorted_ph[i]
                merge_radius = get_merge_radius(current_h)
                neighbors = veg_tree.query_ball_point([sorted_px[i], sorted_py[i]], r=merge_radius)
                all_densities.append(len(neighbors))

            for i in range(total_peaks):
                if i % 100 == 0:
                    if self.isCanceled(): return False
                    self.setProgress(70 + (30 * i / max(1, total_peaks)))

                if processed_peaks[i]: continue

                current_h = sorted_ph[i]
                merge_radius = get_merge_radius(current_h)
                cluster_idx = peaks_tree.query_ball_point([sorted_px[i], sorted_py[i]], r=merge_radius)
                
                cluster_x = sorted_px[cluster_idx]
                cluster_y = sorted_py[cluster_idx]
                
                centroid_x = float(np.mean(cluster_x))
                centroid_y = float(np.mean(cluster_y))
                max_h = float(current_h)
                
                # Densidad y tamaño (LÓGICA ORIGINAL)
                density_neighbors = veg_tree.query_ball_point([centroid_x, centroid_y], r=merge_radius)
                tree_size, point_count = classify_tree_size_by_density(centroid_x, centroid_y, all_densities)
                
                processed_peaks[cluster_idx] = True
                
                # Check de edificios
                is_valid = True
                if has_buildings:
                    d, _ = b_tree.query([centroid_x, centroid_y], k=1, distance_upper_bound=MIN_DIST_TO_BUILDING)
                    if d != float('inf'): is_valid = False
                
                if is_valid:
                    # --- CALCULAR MÉTRICAS DENDROMÉTRICAS ---
                    tree_points_idx = density_neighbors
                    
                    crown_diam = 0.0
                    crown_area = 0.0
                    avg_rgb = "N/A"
                    density = 0.0
                    
                    if len(tree_points_idx) > 0:
                        tx = final_x[tree_points_idx]
                        ty = final_y[tree_points_idx]
                        
                        range_x = np.max(tx) - np.min(tx)
                        range_y = np.max(ty) - np.min(ty)
                        crown_diam = float((range_x + range_y) / 2.0)
                        
                        radius_est = crown_diam / 2.0
                        crown_area = float(np.pi * (radius_est ** 2))
                        
                        if crown_area > 0:
                            density = len(tree_points_idx) / crown_area
                        
                        if has_color:
                            original_indices = final_color_indices_trees[tree_points_idx]
                            r_val = int(np.mean(trees_r[original_indices]))
                            g_val = int(np.mean(trees_g[original_indices]))
                            b_val = int(np.mean(trees_b[original_indices]))
                            avg_rgb = f"{r_val},{g_val},{b_val}"
                    
                    final_features.append({
                        "x": centroid_x, 
                        "y": centroid_y, 
                        "height": max_h, 
                        "type": "tree",
                        "size": tree_size,
                        "point_count": len(density_neighbors),
                        "crown_diam": crown_diam,
                        "crown_area": crown_area,
                        "color_rgb": avg_rgb,
                        "density": density
                    })

            # --- DETECCIÓN DE ÁRBOLES (Finalizada) ---
            trees_detected = [f for f in final_features if f['height'] >= self.size_threshold]
            
            # --- DETECCIÓN DE ARBUSTOS (SOLO Clase 4 - Vegetación Media) ---
            # SIEMPRE detectar arbustos para visualización (CSV es opcional)
            shrubs_detected = []
            
            self.setProgress(75)
            
            # Extraer SOLO puntos con clasificación 4 (vegetación media)
            med_veg_mask = classification == 4
            med_x = x[med_veg_mask]
            med_y = y[med_veg_mask]
            med_z = z[med_veg_mask]
            med_color_indices = np.arange(len(x))[med_veg_mask]
            
            QgsMessageLog.logMessage(f"DEBUG: Puntos clase 4 encontrados: {len(med_x)}", "MyLiDAR", Qgis.Info)
            
            med_r = red[med_veg_mask] if has_color else None
            med_g = green[med_veg_mask] if has_color else None
            med_b = blue[med_veg_mask] if has_color else None
            
            if len(med_x) > 0:
                # Normalizar altura
                norm_h_med = np.zeros(len(med_x), dtype=np.float32)
                for i in range(0, len(med_x), batch_size):
                    if self.isCanceled(): return False
                    end = min(i + batch_size, len(med_x))
                    query_xy = np.column_stack((med_x[i:end], med_y[i:end]))
                    _, idxs = ground_tree.query(query_xy, k=1)
                    norm_h_med[i:end] = med_z[i:end] - ground_xyz[idxs, 2]
                
                # CLUSTERING: Agrupar SOLO puntos densos (eps=2.0m, mín 3 puntos)
                # Los arbustos reales tienen ≥3 puntos densos sin gaps
                from sklearn.cluster import DBSCAN
                
                xy_coords = np.column_stack((med_x, med_y))
                clustering = DBSCAN(eps=2.0, min_samples=3).fit(xy_coords)
                labels = clustering.labels_
                
                num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                num_noise = list(labels).count(-1)
                QgsMessageLog.logMessage(f"DEBUG: Clusters encontrados: {num_clusters}, Ruido: {num_noise}", "MyLiDAR", Qgis.Info)
                
                # UN PUNTO por cluster (centroide)
                unique_labels = set(labels)
                for cluster_id in unique_labels:
                    if cluster_id == -1:  # Ruido (clusters con <5 puntos)
                        continue
                    
                    cluster_mask = labels == cluster_id
                    cluster_indices = np.where(cluster_mask)[0]
                    
                    # Centroide del cluster
                    cent_x = float(np.mean(med_x[cluster_indices]))
                    cent_y = float(np.mean(med_y[cluster_indices]))
                    cent_h = float(np.mean(norm_h_med[cluster_indices]))
                    
                    # Calcular métricas
                    sx = med_x[cluster_indices]
                    sy = med_y[cluster_indices]
                    
                    range_x = np.max(sx) - np.min(sx)
                    range_y = np.max(sy) - np.min(sy)
                    shrub_crown_diam = float((range_x + range_y) / 2.0)
                    shrub_crown_area = float(np.pi * ((shrub_crown_diam / 2.0) ** 2))
                    
                    shrub_density = len(cluster_indices) / max(shrub_crown_area, 0.01)
                    
                    # Color si disponible
                    shrub_avg_rgb = "N/A"
                    if has_color and med_r is not None:
                        sr_val = int(np.mean(med_r[cluster_indices]))
                        sg_val = int(np.mean(med_g[cluster_indices]))
                        sb_val = int(np.mean(med_b[cluster_indices]))
                        shrub_avg_rgb = f"{sr_val},{sg_val},{sb_val}"
                    
                    shrubs_detected.append({
                        "x": cent_x,
                        "y": cent_y,
                        "height": cent_h,
                        "type": "shrub",
                        "point_count": len(cluster_indices),
                        "crown_diam": shrub_crown_diam,
                        "crown_area": shrub_crown_area,
                        "color_rgb": shrub_avg_rgb,
                        "density": shrub_density
                    })

            # --- DETECCIÓN DE GRASS/CÉSPED (Clase 2 + ExG + Clase 3 existente) - COMO ÁREAS/POLÍGONOS ---
            grass_detected = []
            
            if has_color:
                self.setProgress(80)
                
                # Combinar DOS fuentes de grass:
                # 1. Puntos Clase 2 (Suelo) con ExG > 20 (nuevos detectados por color)
                # 2. Puntos Clase 3 (Vegetación baja) ya existentes en el archivo
                
                grass_mask = classification == 2
                low_veg_mask = classification == 3
                combined_grass_mask = grass_mask | low_veg_mask  # UNIÓN de ambas clases
                
                grass_x = x[combined_grass_mask]
                grass_y = y[combined_grass_mask]
                grass_z = z[combined_grass_mask]
                grass_class = classification[combined_grass_mask]
                
                if len(grass_x) > 0:
                    # Extraer color
                    grass_r = red[combined_grass_mask]
                    grass_g = green[combined_grass_mask]
                    grass_b = blue[combined_grass_mask]
                    
                    # Aplicar filtro ExG SOLO a puntos de clase 2
                    exg = (2 * grass_g.astype(np.float32)) - grass_r.astype(np.float32) - grass_b.astype(np.float32)
                    
                    # Máscara: (Clase 3 SIEMPRE) O (Clase 2 con ExG > 12)
                    # Clase 3 = vegetación baja ya clasificada (NO necesita filtro ExG)
                    # Clase 2 = suelo potencial, solo si muestra características de grass (ExG > 12)
                    is_low_veg = grass_class == 3  # Clase 3 existentes - SIEMPRE incluida
                    is_grass_by_color = (grass_class == 2) & (exg > 12) & (grass_g > grass_r) & (grass_g > grass_b)  # Clase 2 con ExG
                    valid_grass_mask = is_low_veg | is_grass_by_color
                    
                    if np.sum(valid_grass_mask) > 0:
                        grass_x_filt = grass_x[valid_grass_mask]
                        grass_y_filt = grass_y[valid_grass_mask]
                        grass_r_filt = grass_r[valid_grass_mask]
                        grass_g_filt = grass_g[valid_grass_mask]
                        grass_b_filt = grass_b[valid_grass_mask]
                        
                        QgsMessageLog.logMessage(f"DEBUG: Puntos de grass candidatos (Clase 3 existentes + Clase 2 por ExG): {len(grass_x_filt)}", "MyLiDAR", Qgis.Info)
                        
                        # CLUSTERING: Agrupar puntos de grass cercanos
                        from sklearn.cluster import DBSCAN
                        
                        # Separar puntos por origen
                        is_low_veg_mask = grass_class == 3  # Clase 3 existente
                        is_grass_new_mask = (grass_class == 2) & (exg > 20)  # Clase 2 con ExG (nuevos)
                        
                        # Crear KDTree de edificios UNA SOLA VEZ
                        build_tree = None
                        if has_buildings:
                            build_tree = cKDTree(np.column_stack((build_x, build_y)))
                        
                        # OPCIÓN 1: Procesar CLASE 3 EXISTENTE - SIN DBSCAN, incluir todos
                        # Nota: is_low_veg_mask ya está filtrada por valid_grass_mask
                        low_veg_indices_in_filtered = np.where(is_low_veg_mask[valid_grass_mask])[0]
                        if len(low_veg_indices_in_filtered) > 0:
                            # Aplicar clustering SUAVE a clase 3 para agrupar en áreas
                            xy_low_veg = np.column_stack((grass_x_filt[low_veg_indices_in_filtered], grass_y_filt[low_veg_indices_in_filtered]))
                            clustering_low_veg = DBSCAN(eps=5.0, min_samples=5).fit(xy_low_veg)  # min_samples=5 para filtrar ruido
                            labels_low_veg = clustering_low_veg.labels_
                            unique_labels_low_veg = set(labels_low_veg)
                            
                            for cluster_id in unique_labels_low_veg:
                                if cluster_id == -1:
                                    continue
                                
                                cluster_mask = labels_low_veg == cluster_id
                                cluster_local_indices = np.where(cluster_mask)[0]
                                cluster_indices = low_veg_indices_in_filtered[cluster_local_indices]
                                
                                if len(cluster_indices) >= 5:  # Al menos 5 puntos para filtrar ruido
                                    self._create_grass_polygon(cluster_indices, grass_x_filt, grass_y_filt, 
                                                               grass_r_filt, grass_g_filt, grass_b_filt, 
                                                               build_tree, grass_detected)
                        
                        # OPCIÓN 2: Procesar CLASE 2 NUEVOS - CON DBSCAN más estricto
                        # Nota: is_grass_new_mask ya está filtrada por valid_grass_mask
                        grass_new_indices_in_filtered = np.where(is_grass_new_mask[valid_grass_mask])[0]
                        if len(grass_new_indices_in_filtered) > 0:
                            xy_grass_new = np.column_stack((grass_x_filt[grass_new_indices_in_filtered], grass_y_filt[grass_new_indices_in_filtered]))
                            clustering_new = DBSCAN(eps=5.0, min_samples=5).fit(xy_grass_new)
                            labels_new = clustering_new.labels_
                            unique_labels_new = set(labels_new)
                            
                            for cluster_id in unique_labels_new:
                                if cluster_id == -1:
                                    continue
                                
                                cluster_mask = labels_new == cluster_id
                                cluster_local_indices = np.where(cluster_mask)[0]
                                cluster_indices = grass_new_indices_in_filtered[cluster_local_indices]
                                
                                if len(cluster_indices) >= 5:  # Al menos 5 puntos para clase 2 nueva
                                    self._create_grass_polygon(cluster_indices, grass_x_filt, grass_y_filt,
                                                               grass_r_filt, grass_g_filt, grass_b_filt,
                                                               build_tree, grass_detected)

            # --- RECLASIFICACIÓN: Cambiar puntos de grass de Clase 2 a Clase 3 (Low Vegetation) ---
            grass_indices_reclassify = np.array([], dtype=int)
            if has_color:
                # Reclasificar SOLO puntos Clase 2 que tengan ExG > 20
                grass_mask_2 = classification == 2
                if np.sum(grass_mask_2) > 0:
                    grass_x_full = x[grass_mask_2]
                    grass_y_full = y[grass_mask_2]
                    grass_r_full = red[grass_mask_2]
                    grass_g_full = green[grass_mask_2]
                    grass_b_full = blue[grass_mask_2]
                    grass_indices_full = np.where(grass_mask_2)[0]
                    
                    # Aplicar filtro ExG (mismo threshold que en detección)
                    exg_full = (2 * grass_g_full.astype(np.float32)) - grass_r_full.astype(np.float32) - grass_b_full.astype(np.float32)
                    grass_color_mask_full = (exg_full > 12) & (grass_g_full > grass_r_full) & (grass_g_full > grass_b_full)
                    
                    # Obtener índices globales de puntos que serán reclasificados
                    grass_indices_reclassify = grass_indices_full[grass_color_mask_full]
                    
                    # Cambiar clasificación en el array (Clase 2 → Clase 3)
                    classification[grass_indices_reclassify] = 3
            
            # --- DETECCIÓN DE EDIFICIOS (Clase 6) ---
            buildings_detected = []
            
            if has_buildings:
                self.setProgress(85)
                
                # Extraer color de edificios (si disponible)
                if has_color:
                    build_r = red[building_mask]
                    build_g = green[building_mask]
                    build_b = blue[building_mask]
                
                # Normalizar altura de edificios
                norm_h_build = np.zeros(len(build_x), dtype=np.float32)
                for i in range(0, len(build_x), batch_size):
                    if self.isCanceled(): return False
                    end = min(i + batch_size, len(build_x))
                    query_xy = np.column_stack((build_x[i:end], build_y[i:end]))
                    _, idxs = ground_tree.query(query_xy, k=1)
                    norm_h_build[i:end] = build_z[i:end] - ground_xyz[idxs, 2]
                
                # CLUSTERING: Agrupar puntos de edificios cercanos
                from sklearn.cluster import DBSCAN
                
                xy_build = np.column_stack((build_x, build_y))
                clustering_build = DBSCAN(eps=3.0, min_samples=10).fit(xy_build)
                labels_build = clustering_build.labels_
                
                num_build_clusters = len(set(labels_build)) - (1 if -1 in labels_build else 0)
                QgsMessageLog.logMessage(f"DEBUG: Edificios detectados (clusters): {num_build_clusters}", "MyLiDAR", Qgis.Info)
                
                unique_labels_build = set(labels_build)
                for cluster_id in unique_labels_build:
                    if cluster_id == -1:
                        continue
                    
                    cluster_mask = labels_build == cluster_id
                    cluster_indices = np.where(cluster_mask)[0]
                    
                    if len(cluster_indices) < 10:
                        continue
                    
                    # Centroide del edificio
                    bld_cx = float(np.mean(build_x[cluster_indices]))
                    bld_cy = float(np.mean(build_y[cluster_indices]))
                    bld_h_max = float(np.max(norm_h_build[cluster_indices]))
                    bld_h_mean = float(np.mean(norm_h_build[cluster_indices]))
                    
                    # Calcular métricas del edificio
                    bx = build_x[cluster_indices]
                    by = build_y[cluster_indices]
                    
                    range_bx = float(np.max(bx) - np.min(bx))
                    range_by = float(np.max(by) - np.min(by))
                    bld_footprint = range_bx * range_by  # Área aproximada del footprint
                    bld_perimeter = 2 * (range_bx + range_by)
                    
                    # Color si disponible
                    bld_avg_rgb = "N/A"
                    if has_color:
                        bld_r = int(np.mean(build_r[cluster_indices]))
                        bld_g = int(np.mean(build_g[cluster_indices]))
                        bld_b = int(np.mean(build_b[cluster_indices]))
                        bld_avg_rgb = f"{bld_r},{bld_g},{bld_b}"
                    
                    # --- CREAR GRID DE CELDAS CUADRADAS COMO EN GRASS ---
                    grid_size = 2.0  # Celdas de 2m x 2m
                    
                    grid_min_x = np.min(build_x[cluster_indices])
                    grid_max_x = np.max(build_x[cluster_indices])
                    grid_min_y = np.min(build_y[cluster_indices])
                    grid_max_y = np.max(build_y[cluster_indices])
                    
                    # Agrupar puntos en grid
                    grid_dict = {}
                    for i, idx in enumerate(cluster_indices):
                        col = int((build_x[idx] - grid_min_x) / grid_size)
                        row = int((grid_max_y - build_y[idx]) / grid_size)
                        key = (col, row)
                        
                        if key not in grid_dict:
                            grid_dict[key] = []
                        grid_dict[key].append(i)
                    
                    # Crear polígonos Shapely para CADA CELDA
                    from shapely.geometry import Polygon
                    from shapely.ops import unary_union
                    
                    cell_polygons_shapely = []
                    cell_details_build = []  # Guardar detalles de cada celda
                    
                    for (col, row), point_indices in grid_dict.items():
                        # Crear cuadrado de la celda
                        cell_x_min = grid_min_x + col * grid_size
                        cell_x_max = cell_x_min + grid_size
                        cell_y_min = grid_max_y - (row + 1) * grid_size
                        cell_y_max = cell_y_min + grid_size
                        
                        # Crear polígono Shapely del cuadrado
                        cell_poly = Polygon([
                            (cell_x_min, cell_y_min),
                            (cell_x_max, cell_y_min),
                            (cell_x_max, cell_y_max),
                            (cell_x_min, cell_y_max)
                        ])
                        cell_polygons_shapely.append(cell_poly)
                        
                        # Guardar detalles de esta celda
                        cell_indices_in_cluster = [cluster_indices[i] for i in point_indices]
                        if has_color:
                            cell_color_r = int(np.mean(build_r[cell_indices_in_cluster]))
                            cell_color_g = int(np.mean(build_g[cell_indices_in_cluster]))
                            cell_color_b = int(np.mean(build_b[cell_indices_in_cluster]))
                        else:
                            cell_color_r = cell_color_g = cell_color_b = 100
                        
                        cell_h_max = float(np.max(norm_h_build[cell_indices_in_cluster]))
                        cell_h_mean = float(np.mean(norm_h_build[cell_indices_in_cluster]))
                        
                        cell_details_build.append({
                            'x_min': cell_x_min,
                            'y_min': cell_y_min,
                            'x_max': cell_x_max,
                            'y_max': cell_y_max,
                            'size_m2': grid_size ** 2,
                            'color_rgb': f"{cell_color_r},{cell_color_g},{cell_color_b}",
                            'point_count': len(cell_indices_in_cluster),
                            'height_max': cell_h_max,
                            'height_mean': cell_h_mean
                        })
                    
                    # UNIFICAR todos los polígonos de celdas
                    if cell_polygons_shapely:
                        unified_geom = unary_union(cell_polygons_shapely)
                        simplified_geom = unified_geom.simplify(0.3, preserve_topology=True)
                        
                        # Extraer coordenadas del polígono unificado
                        if simplified_geom.geom_type == 'Polygon':
                            # Un solo polígono
                            coords = list(simplified_geom.exterior.coords)
                            final_polygons = [coords]
                        elif simplified_geom.geom_type == 'MultiPolygon':
                            # Múltiples polígonos (si hay huecos o islas)
                            final_polygons = []
                            for poly in simplified_geom.geoms:
                                coords = list(poly.exterior.coords)
                                final_polygons.append(coords)
                        else:
                            # Fallback si es algo inesperado
                            final_polygons = []
                        
                        if final_polygons:
                            # Calcular área total basada en número de celdas
                            total_area = len(grid_dict) * (grid_size ** 2)
                            
                            buildings_detected.append({
                                "x": bld_cx,
                                "y": bld_cy,
                                "height_max": bld_h_max,
                                "height_mean": bld_h_mean,
                                "type": "building",
                                "point_count": len(cluster_indices),
                                "footprint_m2": total_area,
                                "length_m": range_bx,
                                "width_m": range_by,
                                "perimeter_m": bld_perimeter,
                                "color_rgb": bld_avg_rgb,
                                "polygons": final_polygons,
                                "cell_details": cell_details_build
                            })
            
            self.visualization_data = final_features + shrubs_detected + grass_detected + buildings_detected
            self.shrubs_data = shrubs_detected  # Guardar aparte para CSV
            self.grass_data = grass_detected
            self.buildings_data = buildings_detected
            self.reclassified_indices = grass_indices_reclassify  # Guardar para escribir LAS
            self.stats = {
                "num_high_orig": len(final_x),
                "num_trees": len(trees_detected),
                "num_shrubs": len(shrubs_detected),
                "num_grass": len(grass_detected),
                "num_buildings": len(buildings_detected),
                "num_reclassified_to_lowveg": len(grass_indices_reclassify),
                "method": "Hybrid: CHM for Trees + DBSCAN for Shrubs + ExG for Grass + DBSCAN for Buildings"
            }
            
            # Exportar CSVs si se pidió
            if self.export_trees and self.csv_path and trees_detected:
                self._export_trees_to_csv(trees_detected)
            
            if self.export_shrubs and self.csv_shrubs_path and shrubs_detected:
                self._export_shrubs_to_csv(shrubs_detected)
            
            # Exportar grass a CSV (solo si el usuario lo pidió)
            if self.export_grass and grass_detected:
                self._export_grass_to_csv(grass_detected)
            
            # Exportar edificios a CSV (solo si el usuario lo pidió)
            if self.export_buildings and buildings_detected:
                self._export_buildings_to_csv(buildings_detected)
            
            # --- ESCRIBIR LAS MODIFICADO CON RECLASIFICACIÓN (SOLO SI EL USUARIO LO PIDIÓ) ---
            if self.export_reclassified and hasattr(self, 'reclassified_indices') and len(self.reclassified_indices) > 0:
                try:
                    # Crear path para LAZ de salida (con _reclassified)
                    las_base = os.path.splitext(self.filename)[0]
                    las_output = las_base + "_reclassified.laz"
                    
                    # Asignar el array de clasificación modificado al objeto las
                    las.classification = classification
                    
                    # Escribir el nuevo archivo LAZ
                    las.write(las_output, do_compress=True)
                    
                    # Guardar el path para importarlo después
                    self.reclassified_laz_path = las_output
                    
                    num_reclassified = len(self.reclassified_indices)
                    QgsMessageLog.logMessage(f"LAZ Reclasificado guardado: {las_output} ({num_reclassified} puntos de Clase 2→3)", "MyLiDAR", Qgis.Success)
                except Exception as e:
                    QgsMessageLog.logMessage(f"Error escribiendo LAZ reclasificado: {str(e)}", "MyLiDAR", Qgis.Warning)
            
            self.setProgress(100)
            return True

        except Exception as e:
            self.exception = e
            QgsMessageLog.logMessage(str(traceback.format_exc()), "MyLiDAR", Qgis.Critical)
            return False

    def _create_grass_polygon(self, cluster_indices, grass_x, grass_y, grass_r, grass_g, grass_b, build_tree, grass_detected):
        """Crea polígono de grass desde índices de cluster - UNIFICA celdas en forma continua"""
        if len(cluster_indices) < 5:
            return
        
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
        
        # Puntos del cluster
        clust_x = grass_x[cluster_indices]
        clust_y = grass_y[cluster_indices]
        clust_r = grass_r[cluster_indices]
        clust_g = grass_g[cluster_indices]
        clust_b = grass_b[cluster_indices]
        
        # Centroide para verificar proximidad a edificios
        cent_x = float(np.mean(clust_x))
        cent_y = float(np.mean(clust_y))
        
        # Verificar proximidad a edificios
        is_near_building = False
        if build_tree is not None:
            dist, _ = build_tree.query([cent_x, cent_y], k=1)
            if dist < 1.0:
                is_near_building = True
        
        if not is_near_building:
            # Grid de densidad: crear polígonos SOLO donde hay puntos
            grid_size = 2.0  # Celdas de 2m x 2m
            
            grid_min_x = np.min(clust_x)
            grid_max_x = np.max(clust_x)
            grid_min_y = np.min(clust_y)
            grid_max_y = np.max(clust_y)
            
            # Agrupar puntos en grid
            grid_dict = {}
            for i in range(len(clust_x)):
                col = int((clust_x[i] - grid_min_x) / grid_size)
                row = int((grid_max_y - clust_y[i]) / grid_size)
                key = (col, row)
                
                if key not in grid_dict:
                    grid_dict[key] = []
                grid_dict[key].append(i)
            
            # --- Filter: min 5 points per cell ---
            valid_keys = {k for k, v in grid_dict.items() if len(v) >= 5}

            # --- (a) Remove isolated cells (no 4-connected neighbour) ---
            connected_keys = set()
            for key in valid_keys:
                c, r = key
                for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    if (c + dc, r + dr) in valid_keys:
                        connected_keys.add(key)
                        break

            # --- (b) Fill small holes (non-grass cell with >=3 grass neighbours) ---
            for _ in range(2):
                to_fill = set()
                candidates = set()
                for key in connected_keys:
                    c, r = key
                    for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nb = (c + dc, r + dr)
                        if nb not in connected_keys:
                            candidates.add(nb)
                for (c, r) in candidates:
                    grass_n = sum(1 for dc, dr in [(1,0),(-1,0),(0,1),(0,-1)]
                                  if (c+dc, r+dr) in connected_keys)
                    if grass_n >= 3:
                        to_fill.add((c, r))
                connected_keys |= to_fill

            # Crear polígonos Shapely para CADA CELDA
            cell_polygons_shapely = []
            cell_details = []  # Guardar detalles de cada celda
            num_cells = len(connected_keys)
            
            for key in connected_keys:
                col, row = key
                point_indices = grid_dict.get(key, [])
                # Crear cuadrado de la celda
                cell_x_min = grid_min_x + col * grid_size
                cell_x_max = cell_x_min + grid_size
                cell_y_min = grid_max_y - (row + 1) * grid_size
                cell_y_max = cell_y_min + grid_size
                
                # Crear polígono Shapely del cuadrado
                cell_poly = Polygon([
                    (cell_x_min, cell_y_min),
                    (cell_x_max, cell_y_min),
                    (cell_x_max, cell_y_max),
                    (cell_x_min, cell_y_max)
                ])
                cell_polygons_shapely.append(cell_poly)
                
                # Guardar detalles de esta celda
                if len(point_indices) > 0:
                    cell_color_r = int(np.mean(clust_r[point_indices]))
                    cell_color_g = int(np.mean(clust_g[point_indices]))
                    cell_color_b = int(np.mean(clust_b[point_indices]))
                else:
                    # Filled hole cell: use cluster average color
                    cell_color_r = int(np.mean(clust_r))
                    cell_color_g = int(np.mean(clust_g))
                    cell_color_b = int(np.mean(clust_b))
                cell_density = len(point_indices) / (grid_size ** 2)
                
                cell_details.append({
                    'x_min': cell_x_min,
                    'y_min': cell_y_min,
                    'x_max': cell_x_max,
                    'y_max': cell_y_max,
                    'size_m2': grid_size ** 2,
                    'color_rgb': f"{cell_color_r},{cell_color_g},{cell_color_b}",
                    'point_count': max(len(point_indices), 1),
                    'density_pts_m2': cell_density
                })
            
            # UNIFICAR todos los polígonos de celdas en uno solo (o MultiPolygon si hay huecos)
            if cell_polygons_shapely:
                # Unión de todos los polígonos
                unified_geom = unary_union(cell_polygons_shapely)
                
                # Simplificar ligeramente para suavizar bordes (tolerance = 0.3m)
                simplified_geom = unified_geom.simplify(0.3, preserve_topology=True)
                
                # Extraer coordenadas del polígono unificado
                if simplified_geom.geom_type == 'Polygon':
                    # Un solo polígono
                    coords = list(simplified_geom.exterior.coords)
                    final_polygons = [coords]
                elif simplified_geom.geom_type == 'MultiPolygon':
                    # Múltiples polígonos (si hay huecos o islas)
                    final_polygons = []
                    for poly in simplified_geom.geoms:
                        coords = list(poly.exterior.coords)
                        final_polygons.append(coords)
                else:
                    # Fallback si es algo inesperado
                    final_polygons = []
                
                if final_polygons:
                    # Calcular área total (siempre basada en número de celdas)
                    total_area = num_cells * (grid_size ** 2)
                    
                    # Color promedio del cluster
                    grass_avg_rgb = f"{int(np.mean(clust_r))},{int(np.mean(clust_g))},{int(np.mean(clust_b))}"
                    
                    grass_detected.append({
                        "polygons": final_polygons,
                        "area": total_area,
                        "color_rgb": grass_avg_rgb,
                        "point_count": len(clust_x),
                        "type": "grass",
                        "cell_details": cell_details
                    })



    def _export_trees_to_csv(self, features):
        """Exporta métricas a CSV"""
        if not features: return
        
        csv_dir = os.path.dirname(self.csv_path)
        if csv_dir and not os.path.exists(csv_dir): os.makedirs(csv_dir)
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'Tree_ID', 'X', 'Y', 'Height_m', 
                'Crown_Diameter_m', 'Crown_Area_m2', 
                'Point_Count', 'Density_pts_m2', 'Color_RGB'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for idx, f in enumerate(features, 1):
                writer.writerow({
                    'Tree_ID': idx,
                    'X': f"{f['x']:.3f}",
                    'Y': f"{f['y']:.3f}",
                    'Height_m': f"{f['height']:.2f}",
                    'Crown_Diameter_m': f"{f['crown_diam']:.2f}",
                    'Crown_Area_m2': f"{f['crown_area']:.2f}",
                    'Point_Count': f['point_count'],
                    'Density_pts_m2': f"{f['density']:.2f}",
                    'Color_RGB': f['color_rgb']
                })
        
        QgsMessageLog.logMessage(f"CSV Exportado: {self.csv_path}", "MyLiDAR", Qgis.Success)

    def _export_shrubs_to_csv(self, features):
        """Exporta arbustos a CSV"""
        if not features: return
        
        csv_dir = os.path.dirname(self.csv_shrubs_path)
        if csv_dir and not os.path.exists(csv_dir): os.makedirs(csv_dir)
        
        with open(self.csv_shrubs_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'Shrub_ID', 'X', 'Y', 'Height_m', 
                'Crown_Diameter_m', 'Crown_Area_m2', 
                'Point_Count', 'Density_pts_m2', 'Color_RGB'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for idx, f in enumerate(features, 1):
                writer.writerow({
                    'Shrub_ID': idx,
                    'X': f"{f['x']:.3f}",
                    'Y': f"{f['y']:.3f}",
                    'Height_m': f"{f['height']:.2f}",
                    'Crown_Diameter_m': f"{f['crown_diam']:.2f}",
                    'Crown_Area_m2': f"{f['crown_area']:.2f}",
                    'Point_Count': f['point_count'],
                    'Density_pts_m2': f"{f['density']:.2f}",
                    'Color_RGB': f['color_rgb']
                })
        
        QgsMessageLog.logMessage(f"CSV Arbustos Exportado: {self.csv_shrubs_path}", "MyLiDAR", Qgis.Success)

    def _export_grass_to_csv(self, features):
        """Exporta cuadrados de grass a CSV con detalles de cada celda"""
        if not features: return
        
        # Crear path para CSV de grass (basado en el input file, no en csv_path que puede ser None)
        base_path = os.path.splitext(self.filename)[0]
        csv_grass_path = base_path + "_grass_areas.csv"
        
        csv_dir = os.path.dirname(csv_grass_path)
        if csv_dir and not os.path.exists(csv_dir): os.makedirs(csv_dir)
        
        with open(csv_grass_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'Grass_Area_ID', 'Cell_ID', 
                'X_Center', 'Y_Center',
                'X_Min', 'Y_Min', 'X_Max', 'Y_Max',
                'Size_m2', 'Point_Count', 'Density_pts_m2', 'Color_RGB'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            row_idx = 1
            for area_idx, f in enumerate(features, 1):
                # Exportar cada cuadrado del área de grass
                if 'cell_details' in f:
                    for cell_idx, cell in enumerate(f['cell_details'], 1):
                        # Calcular centro de la celda
                        x_center = (cell['x_min'] + cell['x_max']) / 2.0
                        y_center = (cell['y_min'] + cell['y_max']) / 2.0
                        
                        writer.writerow({
                            'Grass_Area_ID': area_idx,
                            'Cell_ID': cell_idx,
                            'X_Center': f"{x_center:.3f}",
                            'Y_Center': f"{y_center:.3f}",
                            'X_Min': f"{cell['x_min']:.3f}",
                            'Y_Min': f"{cell['y_min']:.3f}",
                            'X_Max': f"{cell['x_max']:.3f}",
                            'Y_Max': f"{cell['y_max']:.3f}",
                            'Size_m2': f"{cell['size_m2']:.2f}",
                            'Point_Count': cell['point_count'],
                            'Density_pts_m2': f"{cell['density_pts_m2']:.2f}",
                            'Color_RGB': cell['color_rgb']
                        })
                        row_idx += 1
        
        QgsMessageLog.logMessage(f"CSV Grass Areas Exportado: {csv_grass_path} ({row_idx-1} celdas)", "MyLiDAR", Qgis.Success)

    def _export_buildings_to_csv(self, features):
        """Exporta edificios detectados a CSV con celdas y coordenadas"""
        if not features: return
        
        base_path = os.path.splitext(self.filename)[0]
        csv_buildings_path = base_path + "_buildings.csv"
        
        csv_dir = os.path.dirname(csv_buildings_path)
        if csv_dir and not os.path.exists(csv_dir): os.makedirs(csv_dir)
        
        with open(csv_buildings_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'Building_ID', 'Cell_ID',
                'X_Center', 'Y_Center',
                'X_Min', 'Y_Min', 'X_Max', 'Y_Max',
                'Size_m2', 'Height_Max_m', 'Height_Mean_m',
                'Point_Count', 'Color_RGB'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            row_idx = 1
            for building_idx, f in enumerate(features, 1):
                # Exportar cada celda del edificio
                if 'cell_details' in f:
                    for cell_idx, cell in enumerate(f['cell_details'], 1):
                        # Calcular centro de la celda
                        x_center = (cell['x_min'] + cell['x_max']) / 2.0
                        y_center = (cell['y_min'] + cell['y_max']) / 2.0
                        
                        writer.writerow({
                            'Building_ID': building_idx,
                            'Cell_ID': cell_idx,
                            'X_Center': f"{x_center:.3f}",
                            'Y_Center': f"{y_center:.3f}",
                            'X_Min': f"{cell['x_min']:.3f}",
                            'Y_Min': f"{cell['y_min']:.3f}",
                            'X_Max': f"{cell['x_max']:.3f}",
                            'Y_Max': f"{cell['y_max']:.3f}",
                            'Size_m2': f"{cell['size_m2']:.2f}",
                            'Height_Max_m': f"{cell['height_max']:.2f}",
                            'Height_Mean_m': f"{cell['height_mean']:.2f}",
                            'Point_Count': cell['point_count'],
                            'Color_RGB': cell['color_rgb']
                        })
                        row_idx += 1
        
        QgsMessageLog.logMessage(f"CSV Edificios Exportado: {csv_buildings_path} ({row_idx-1} celdas)", "MyLiDAR", Qgis.Success)

    def finished(self, result):
        if result:
            try:
                crs_str = f"EPSG:{self.epsg}" if self.epsg else "EPSG:4326"
                
                # --- CAPA DE ÁRBOLES (CHM) ---
                trees = [f for f in self.visualization_data if f.get('type') == 'tree']
                if trees:
                    vl_trees = QgsVectorLayer(f"Point?crs={crs_str}", self.tr("Trees (CHM)"), "memory")
                    pr = vl_trees.dataProvider()
                    pr.addAttributes([
                        QgsField("type", QVariant.String),
                        QgsField("height", QVariant.Double),
                        QgsField("crown_diam", QVariant.Double),
                        QgsField("crown_area", QVariant.Double)
                    ])
                    vl_trees.updateFields()
                    
                    feats = []
                    for item in trees:
                        f = QgsFeature()
                        f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(item["x"], item["y"])))
                        f.setAttributes([item["type"], item["height"], item["crown_diam"], item["crown_area"]])
                        feats.append(f)
                    
                    pr.addFeatures(feats)
                    vl_trees.updateExtents()
                    
                    # Simbología árboles (verde oscuro)
                    symbol = QgsSymbol.defaultSymbol(vl_trees.geometryType())
                    symbol.setColor(QColor("#00AA00"))
                    symbol.setSize(4.0)
                    renderer = QgsCategorizedSymbolRenderer("type", [QgsRendererCategory("tree", symbol, "Tree")])
                    vl_trees.setRenderer(renderer)
                    QgsProject.instance().addMapLayer(vl_trees)
                
                # --- CAPA DE ARBUSTOS (Grid Density) ---
                shrubs = [f for f in self.visualization_data if f.get('type') == 'shrub']
                if shrubs:
                    vl_shrubs = QgsVectorLayer(f"Point?crs={crs_str}", self.tr("Shrubs (Grid Density)"), "memory")
                    pr = vl_shrubs.dataProvider()
                    pr.addAttributes([
                        QgsField("type", QVariant.String),
                        QgsField("height", QVariant.Double),
                        QgsField("crown_diam", QVariant.Double),
                        QgsField("crown_area", QVariant.Double)
                    ])
                    vl_shrubs.updateFields()
                    
                    feats = []
                    for item in shrubs:
                        f = QgsFeature()
                        f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(item["x"], item["y"])))
                        f.setAttributes([item["type"], item["height"], item["crown_diam"], item["crown_area"]])
                        feats.append(f)
                    
                    pr.addFeatures(feats)
                    vl_shrubs.updateExtents()
                    
                    # Simbología arbustos (amarillo/naranja)
                    symbol = QgsSymbol.defaultSymbol(vl_shrubs.geometryType())
                    symbol.setColor(QColor("#FFAA00"))
                    symbol.setSize(3.0)
                    renderer = QgsCategorizedSymbolRenderer("type", [QgsRendererCategory("shrub", symbol, "Shrub")])
                    vl_shrubs.setRenderer(renderer)
                    QgsProject.instance().addMapLayer(vl_shrubs)
                
                # --- CAPA DE GRASS/CÉSPED (ExG - Polígonos) ---
                grass = [f for f in self.visualization_data if f.get('type') == 'grass']
                if grass:
                    vl_grass = QgsVectorLayer(f"Polygon?crs={crs_str}", self.tr("Grass Areas (ExG)"), "memory")
                    pr = vl_grass.dataProvider()
                    pr.addAttributes([
                        QgsField("type", QVariant.String),
                        QgsField("area_m2", QVariant.Double),
                        QgsField("point_count", QVariant.Int),
                        QgsField("color_rgb", QVariant.String)
                    ])
                    vl_grass.updateFields()
                    
                    feats = []
                    for item in grass:
                        f = QgsFeature()
                        
                        # Crear polígonos desde el grid
                        if "polygons" in item:
                            # Multipolígono: unir todos los cuadrados de celdas
                            polygon_rings = []
                            for poly_coords in item["polygons"]:
                                coords = [QgsPointXY(pt[0], pt[1]) for pt in poly_coords]
                                polygon_rings.append(coords)
                            # Crear multipolígono
                            geom = QgsGeometry.fromMultiPolygonXY([polygon_rings])
                        else:
                            # Fallback: buffer circular
                            cent_x, cent_y = item["centroid"]
                            radius = item["radius"]
                            import math
                            n_points = 24
                            coords = [QgsPointXY(cent_x + radius * math.cos(2 * math.pi * i / n_points),
                                                 cent_y + radius * math.sin(2 * math.pi * i / n_points))
                                     for i in range(n_points)]
                            coords.append(coords[0])
                            geom = QgsGeometry.fromPolygonXY([coords])
                        
                        f.setGeometry(geom)
                        f.setAttributes([item["type"], item["area"], item["point_count"], item["color_rgb"]])
                        feats.append(f)
                    
                    pr.addFeatures(feats)
                    vl_grass.updateExtents()
                    
                    # Simbología grass (verde claro con transparencia)
                    symbol = QgsSymbol.defaultSymbol(vl_grass.geometryType())
                    symbol.setColor(QColor(144, 238, 144, 100))  # Verde claro con 100/255 opacidad
                    renderer = QgsCategorizedSymbolRenderer("type", [QgsRendererCategory("grass", symbol, "Grass Area")])
                    vl_grass.setRenderer(renderer)
                    QgsProject.instance().addMapLayer(vl_grass)
                
                # --- CAPA DE EDIFICIOS (DBSCAN - Polígonos) ---
                buildings = [f for f in self.visualization_data if f.get('type') == 'building']
                if buildings:
                    vl_buildings = QgsVectorLayer(f"Polygon?crs={crs_str}", self.tr("Buildings (DBSCAN)"), "memory")
                    pr = vl_buildings.dataProvider()
                    pr.addAttributes([
                        QgsField("type", QVariant.String),
                        QgsField("height_max", QVariant.Double),
                        QgsField("height_mean", QVariant.Double),
                        QgsField("footprint_m2", QVariant.Double),
                        QgsField("point_count", QVariant.Int),
                        QgsField("color_rgb", QVariant.String)
                    ])
                    vl_buildings.updateFields()
                    
                    feats = []
                    for item in buildings:
                        f = QgsFeature()
                        
                        # Crear polígonos desde el grid
                        if "polygons" in item:
                            # Multipolígono: unir todos los cuadrados de celdas
                            polygon_rings = []
                            for poly_coords in item["polygons"]:
                                coords = [QgsPointXY(pt[0], pt[1]) for pt in poly_coords]
                                polygon_rings.append(coords)
                            # Crear multipolígono
                            geom = QgsGeometry.fromMultiPolygonXY([polygon_rings])
                        else:
                            # Fallback: punto
                            geom = QgsGeometry.fromPointXY(QgsPointXY(item["x"], item["y"]))
                        
                        f.setGeometry(geom)
                        f.setAttributes([item["type"], item["height_max"], item["height_mean"], 
                                        item["footprint_m2"], item["point_count"], item["color_rgb"]])
                        feats.append(f)
                    
                    pr.addFeatures(feats)
                    vl_buildings.updateExtents()
                    
                    # Simbología edificios (rojo/naranja con transparencia)
                    symbol = QgsSymbol.defaultSymbol(vl_buildings.geometryType())
                    symbol.setColor(QColor(220, 100, 80, 120))  # Rojo-naranja con transparencia
                    renderer = QgsCategorizedSymbolRenderer("type", [QgsRendererCategory("building", symbol, "Building Area")])
                    vl_buildings.setRenderer(renderer)
                    QgsProject.instance().addMapLayer(vl_buildings)
                
                # --- IMPORTAR LAZ RECLASIFICADO ---
                if self.reclassified_laz_path and os.path.exists(self.reclassified_laz_path):
                    try:
                        # Usar QgsPointCloudLayer para cargar nubes de puntos LAZ
                        pc_layer = QgsPointCloudLayer(self.reclassified_laz_path, os.path.basename(self.reclassified_laz_path), "pdal")
                        if pc_layer.isValid():
                            QgsProject.instance().addMapLayer(pc_layer)
                            QgsMessageLog.logMessage(f"LAZ Reclasificado importado: {os.path.basename(self.reclassified_laz_path)}", "MyLiDAR", Qgis.Success)
                        else:
                            QgsMessageLog.logMessage(f"No se pudo cargar LAZ reclasificado", "MyLiDAR", Qgis.Warning)
                    except Exception as e:
                        QgsMessageLog.logMessage(f"Error importando LAZ: {str(e)}", "MyLiDAR", Qgis.Warning)
                
                # Mensaje Final
                s = self.stats
                msg = (f"Árboles detectados (CHM): {s.get('num_trees', 0):,}\n"
                       f"Arbustos detectados (DBSCAN): {s.get('num_shrubs', 0):,}\n"
                       f"Césped detectado (ExG): {s.get('num_grass', 0):,}\n"
                       f"Edificios detectados (DBSCAN): {s.get('num_buildings', 0):,}\n"
                       f"Puntos reclasificados: {s.get('num_reclassified_to_lowveg', 0):,}")
                if self.export_trees and self.csv_path:
                    msg += f"\n✓ CSV Árboles: {os.path.basename(self.csv_path)}"
                if self.export_shrubs and self.csv_shrubs_path:
                    msg += f"\n✓ CSV Arbustos: {os.path.basename(self.csv_shrubs_path)}"
                if self.export_buildings and hasattr(self, 'buildings_data') and self.buildings_data:
                    base_path = os.path.splitext(self.filename)[0]
                    msg += f"\n✓ CSV Edificios: {os.path.basename(base_path + '_buildings.csv')}"
                if self.export_reclassified and self.reclassified_laz_path:
                    msg += f"\n✓ LAZ Reclasificado: {os.path.basename(self.reclassified_laz_path)}"
                
                QMessageBox.information(self.parent.iface.mainWindow(), self.tr("Success"), msg)
                
                # --- GUARDAR DATOS PARA EXPORT OBJ ---
                self.parent.last_classification_data = {
                    'filename': self.filename,
                    'trees': [f for f in self.visualization_data if f.get('type') == 'tree'],
                    'shrubs': [f for f in self.visualization_data if f.get('type') == 'shrub'],
                    'grass': [f for f in self.visualization_data if f.get('type') == 'grass'],
                    'buildings': self.buildings_data if hasattr(self, 'buildings_data') else []
                }
                
            except Exception as e:
                QgsMessageLog.logMessage(str(e), "MyLiDAR", Qgis.Warning)
        else:
            err = str(self.exception) if self.exception else "Unknown error"
            QMessageBox.critical(self.parent.iface.mainWindow(), self.tr("Error"), err)
        
        # Limpieza
        if self in self.parent.running_tasks:
            self.parent.running_tasks.remove(self)


# =================================================================================
# --- FUNCIÓN LANZADORA (CONECTAR AL BOTÓN) ---
# =================================================================================

def classify_vegetation(self):
    # 1. Abrir diálogo
    dialog = VegetationClassificationDialog(self.iface.mainWindow(), translator=self.tr)
    if dialog.exec_() != QDialog.Accepted:
        return

    input_filename, output_filename = dialog.get_input_output()
    if not input_filename:
        QMessageBox.warning(self.iface.mainWindow(), self.tr("Alert"), self.tr("Select input file."))
        return

    # 2. Obtener parámetros
    low_thresh, high_thresh = dialog.get_values()
    size_threshold = dialog.get_size_threshold()
    export_trees, export_shrubs, export_grass, export_buildings, export_reclassified = dialog.get_export_options()
    
    # 3. Configurar rutas CSV
    csv_path = None
    csv_shrubs_path = None
    
    if export_trees or export_shrubs:
        base = os.path.splitext(input_filename)[0]
        if export_trees:
            csv_path = f"{base}_metrics.csv"
        if export_shrubs:
            csv_shrubs_path = f"{base}_shrubs.csv"

    # 4. Lanzar Tarea
    task_desc = f"Classifying data in {os.path.basename(input_filename)}"
    task = VegetationClassificationCHMTask(
        task_desc, input_filename, output_filename, low_thresh, high_thresh, 
        self, self.tr, size_threshold=size_threshold, 
        export_trees=export_trees, export_shrubs=export_shrubs, export_grass=export_grass,
        export_buildings=export_buildings, export_reclassified=export_reclassified,
        csv_path=csv_path, csv_shrubs_path=csv_shrubs_path
    )

    if not hasattr(self, 'running_tasks'):
        self.running_tasks = []
    self.running_tasks.append(task)
    QgsApplication.taskManager().addTask(task)
    
    self.iface.messageBar().pushMessage("Task Started", f"Processing {os.path.basename(input_filename)}...", level=Qgis.Info)
