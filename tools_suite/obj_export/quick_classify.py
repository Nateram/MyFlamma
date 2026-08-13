# -*- coding: utf-8 -*-
"""
Quick Classify - Lightweight classification task for OBJ export.
No LAZ export, no CSV export, just in-memory data generation.
"""

import os
import numpy as np
from qgis.core import QgsTask, QgsMessageLog, Qgis


class QuickClassifyTask(QgsTask):
    """Fast background classification - data only, no file outputs."""

    def __init__(self, description, filename, plugin_ref, translator):
        super().__init__(description, QgsTask.CanCancel)
        self.filename = filename
        self.plugin = plugin_ref
        self.tr = translator
        self.error_msg = None

    def run(self):
        try:
            # Import here to avoid issues if not in QGIS
            import laspy
            from scipy.spatial import cKDTree
            from scipy.ndimage import gaussian_filter, maximum_filter
            from sklearn.cluster import DBSCAN
            from shapely.geometry import Polygon
            from shapely.ops import unary_union

            self.setProgress(5)

            # --- READ LAS ---
            las = laspy.read(self.filename)
            # Convert ScaledArrayView to numpy arrays
            x = np.array(las.x, dtype=np.float64)
            y = np.array(las.y, dtype=np.float64)
            z = np.array(las.z, dtype=np.float64)
            classification = np.array(las.classification, dtype=np.int32)

            has_color = hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue')
            if has_color:
                red = np.array(las.red, dtype=np.uint8)
                green = np.array(las.green, dtype=np.uint8)
                blue = np.array(las.blue, dtype=np.uint8)

            QgsMessageLog.logMessage(self.tr(f"Classify: Read {len(x)} points from {os.path.basename(self.filename)}"), "MyFlamma", Qgis.Info)
            self.setProgress(10)

            if self.isCanceled():
                return False

            # --- GROUND POINTS (Class 2) ---
            ground_mask = classification == 2
            ground_xyz = np.column_stack((x[ground_mask], y[ground_mask], z[ground_mask]))
            ground_tree = None
            if len(ground_xyz) > 0:
                ground_tree = cKDTree(ground_xyz)

            self.setProgress(15)

            # --- BUILD BUILDING KDTree early (for proximity filtering) ---
            building_mask = classification == 6
            build_tree_2d = None
            MIN_DIST_TO_BUILDING = 2.0
            if np.sum(building_mask) > 0:
                bld_x = x[building_mask]
                bld_y = y[building_mask]
                build_tree_2d = cKDTree(np.column_stack((bld_x, bld_y)))

            # --- AUTOMATED ROAD DETECTION (from ground characteristics) ---
            # Roads = flat ground cells with no vegetation above and not green
            roads_detected = []
            road_cell_keys = set()
            road_tree_2d = None
            MIN_DIST_TO_ROAD = 1.5  # nothing grows on roads

            if len(ground_xyz) > 50:
                gx_rd = x[ground_mask]
                gy_rd = y[ground_mask]
                gz_rd = z[ground_mask]
                if has_color:
                    g_red_rd = red[ground_mask]
                    g_green_rd = green[ground_mask]
                    g_blue_rd = blue[ground_mask]

                grid_size_road = 2.0

                # Grid ground points
                road_ground_grid = {}
                for i in range(len(gx_rd)):
                    col = int(gx_rd[i] / grid_size_road)
                    row = int(gy_rd[i] / grid_size_road)
                    key = (col, row)
                    if key not in road_ground_grid:
                        road_ground_grid[key] = []
                    road_ground_grid[key].append(i)

                # Grid vegetation points (class 3,4,5) for overlap check
                veg_any_mask = np.isin(classification, [3, 4, 5])
                veg_grid_count = {}
                if np.sum(veg_any_mask) > 0:
                    veg_x_all = x[veg_any_mask]
                    veg_y_all = y[veg_any_mask]
                    for i in range(len(veg_x_all)):
                        col = int(veg_x_all[i] / grid_size_road)
                        row = int(veg_y_all[i] / grid_size_road)
                        key = (col, row)
                        veg_grid_count[key] = veg_grid_count.get(key, 0) + 1

                # Grid building points for overlap check
                bld_grid_keys = set()
                if np.sum(building_mask) > 0:
                    bldx_all = x[building_mask]
                    bldy_all = y[building_mask]
                    for i in range(len(bldx_all)):
                        col = int(bldx_all[i] / grid_size_road)
                        row = int(bldy_all[i] / grid_size_road)
                        bld_grid_keys.add((col, row))

                # Evaluate each ground cell for road characteristics
                candidate_road_cells = set()
                for key, indices in road_ground_grid.items():
                    # Minimum point density (at least 5 ground pts per 4m² cell)
                    if len(indices) < 5:
                        continue
                    # Flatness: low height standard deviation
                    cell_z = gz_rd[indices]
                    if np.std(cell_z) > 0.15:  # > 15 cm → not flat enough
                        continue
                    # No vegetation above
                    if veg_grid_count.get(key, 0) > 2:
                        continue
                    # Not inside a building
                    if key in bld_grid_keys:
                        continue
                    # Color: not green (if colour available)
                    if has_color:
                        cr = float(np.mean(g_red_rd[indices]))
                        cg = float(np.mean(g_green_rd[indices]))
                        cb = float(np.mean(g_blue_rd[indices]))
                        exg = 2.0 * cg - cr - cb
                        if exg > 10:  # too green → vegetation, not road
                            continue
                    candidate_road_cells.add(key)

                # Connected-component filtering (8-connectivity DFS)
                # Keep only components with >= 5 cells (>= 20 m²)
                MIN_ROAD_CELLS = 5
                visited = set()
                road_cell_details = []

                for seed in candidate_road_cells:
                    if seed in visited:
                        continue
                    stack = [seed]
                    component = []
                    while stack:
                        cell = stack.pop()
                        if cell in visited or cell not in candidate_road_cells:
                            continue
                        visited.add(cell)
                        component.append(cell)
                        col, row = cell
                        for dc in (-1, 0, 1):
                            for dr in (-1, 0, 1):
                                if dc == 0 and dr == 0:
                                    continue
                                nb = (col + dc, row + dr)
                                if nb not in visited and nb in candidate_road_cells:
                                    stack.append(nb)

                    if len(component) < MIN_ROAD_CELLS:
                        continue

                    for key in component:
                        col, row = key
                        road_cell_keys.add(key)
                        road_cell_details.append({
                            'x_min': float(col * grid_size_road),
                            'y_min': float(row * grid_size_road),
                            'x_max': float((col + 1) * grid_size_road),
                            'y_max': float((row + 1) * grid_size_road),
                            'size_m2': grid_size_road ** 2,
                            'point_count': len(road_ground_grid[key])
                        })

                if road_cell_details:
                    roads_detected.append({
                        'area': len(road_cell_details) * (grid_size_road ** 2),
                        'point_count': sum(len(road_ground_grid[k]) for k in road_cell_keys),
                        'type': 'road',
                        'cell_details': road_cell_details
                    })

                    # Build KDTree from ground points in road cells
                    road_pt_idx = []
                    for k in road_cell_keys:
                        road_pt_idx.extend(road_ground_grid[k])
                    rd_pts = np.column_stack((gx_rd[road_pt_idx], gy_rd[road_pt_idx]))
                    road_tree_2d = cKDTree(rd_pts)

            QgsMessageLog.logMessage(self.tr(f"Roads auto-detected: {len(roads_detected)} areas, {len(road_cell_keys)} cells"),
                "MyFlamma", Qgis.Info)

            # --- TREES (Class 3+5 - Vegetation, filtered by height) ---
            trees_detected = []
            high_veg_mask = np.isin(classification, [3, 5])  # Same as main classifier
            
            if np.sum(high_veg_mask) > 100:
                hvx = x[high_veg_mask]
                hvy = y[high_veg_mask]
                hvz = z[high_veg_mask]
                
                if has_color:
                    hvr = red[high_veg_mask]
                    hvg = green[high_veg_mask]
                    hvb = blue[high_veg_mask]

                # Normalize height relative to ground
                norm_h = np.zeros(len(hvx), dtype=np.float32)
                if ground_tree:
                    batch = 50000
                    ground_tree_2d = cKDTree(ground_xyz[:, :2])
                    for i in range(0, len(hvx), batch):
                        end = min(i + batch, len(hvx))
                        qxy = np.column_stack((hvx[i:end], hvy[i:end]))
                        _, idxs = ground_tree_2d.query(qxy, k=1)
                        norm_h[i:end] = hvz[i:end] - ground_xyz[idxs, 2]
                else:
                    norm_h = hvz.copy()

                # Filter only tall points (>2m)
                tall_mask = norm_h > 2.0
                if np.sum(tall_mask) > 50:
                    tx = hvx[tall_mask]
                    ty = hvy[tall_mask]
                    th = norm_h[tall_mask]
                    
                    if has_color:
                        tr = hvr[tall_mask]
                        tg = hvg[tall_mask]
                        tb = hvb[tall_mask]

                    self.setProgress(25)
                    
                    # ============================================================
                    # CHM approach (same as vegetation_classification_chm.py)
                    # ============================================================
                    
                    # Dynamic merge radius (identical to main classifier)
                    def get_merge_radius(height):
                        if height < 2.0:   return 1.5
                        elif height < 10.0: return 3.5
                        elif height < 18.0: return 5.0
                        else:               return 7.0
                    
                    # --- CHM rasterization ---
                    pixel_size = 0.5
                    chm_min_x, chm_max_x = float(np.min(tx)), float(np.max(tx))
                    chm_min_y, chm_max_y = float(np.min(ty)), float(np.max(ty))
                    
                    chm_cols = int((chm_max_x - chm_min_x) / pixel_size) + 1
                    chm_rows = int((chm_max_y - chm_min_y) / pixel_size) + 1
                    
                    chm_grid = np.zeros((chm_rows, chm_cols), dtype=np.float32)
                    idx_col = ((tx - chm_min_x) / pixel_size).astype(int)
                    idx_row = ((chm_max_y - ty) / pixel_size).astype(int)
                    # Clamp indices
                    idx_col = np.clip(idx_col, 0, chm_cols - 1)
                    idx_row = np.clip(idx_row, 0, chm_rows - 1)
                    
                    sort_idx = np.argsort(th)
                    chm_grid[idx_row[sort_idx], idx_col[sort_idx]] = th[sort_idx]
                    
                    # --- Gaussian smoothing ---
                    smoothed_chm = gaussian_filter(chm_grid, sigma=1.5)
                    
                    # --- Peak detection (local maxima) ---
                    local_max = maximum_filter(smoothed_chm, size=7)
                    peaks = (smoothed_chm == local_max) & (smoothed_chm > 2.0)
                    
                    peak_rows, peak_cols = np.where(peaks)
                    raw_peak_x = chm_min_x + (peak_cols * pixel_size) + (pixel_size / 2)
                    raw_peak_y = chm_max_y - (peak_rows * pixel_size) - (pixel_size / 2)
                    raw_peak_h = smoothed_chm[peak_rows, peak_cols]
                    
                    QgsMessageLog.logMessage(f"CHM peaks found: {len(raw_peak_x)}", "MyFlamma", Qgis.Info)
                    self.setProgress(30)
                    
                    # --- Peak merging (tallest first, dynamic radius) ---
                    sorted_peak_idx = np.argsort(-raw_peak_h)
                    sorted_px = raw_peak_x[sorted_peak_idx]
                    sorted_py = raw_peak_y[sorted_peak_idx]
                    sorted_ph = raw_peak_h[sorted_peak_idx]
                    
                    if len(sorted_px) > 0:
                        peaks_kdtree = cKDTree(np.column_stack((sorted_px, sorted_py)))
                        veg_kdtree = cKDTree(np.column_stack((tx, ty)))
                        processed_peaks = np.zeros(len(sorted_px), dtype=bool)
                        
                        for i in range(len(sorted_px)):
                            if processed_peaks[i]:
                                continue
                            
                            current_h = sorted_ph[i]
                            merge_radius = get_merge_radius(current_h)
                            
                            # Merge nearby peaks
                            cluster_idx = peaks_kdtree.query_ball_point(
                                [sorted_px[i], sorted_py[i]], r=merge_radius)
                            
                            centroid_x = float(np.mean(sorted_px[cluster_idx]))
                            centroid_y = float(np.mean(sorted_py[cluster_idx]))
                            max_h = float(current_h)
                            
                            processed_peaks[cluster_idx] = True
                            
                            # Building proximity filter
                            if build_tree_2d is not None:
                                d_bld, _ = build_tree_2d.query([centroid_x, centroid_y], k=1)
                                if d_bld < MIN_DIST_TO_BUILDING:
                                    continue
                            
                            # Road proximity filter
                            if road_tree_2d is not None:
                                d_rd, _ = road_tree_2d.query([centroid_x, centroid_y], k=1)
                                if d_rd < MIN_DIST_TO_ROAD:
                                    continue
                            
                            # Crown metrics from actual vegetation points
                            tree_pts_idx = veg_kdtree.query_ball_point(
                                [centroid_x, centroid_y], r=merge_radius)
                            
                            crown_diam = 1.0
                            crown_area = np.pi * 0.25
                            color = "100,150,100"
                            point_count = len(tree_pts_idx)
                            
                            if len(tree_pts_idx) > 0:
                                tpx = tx[tree_pts_idx]
                                tpy = ty[tree_pts_idx]
                                range_x = float(np.max(tpx) - np.min(tpx))
                                range_y = float(np.max(tpy) - np.min(tpy))
                                crown_diam = max((range_x + range_y) / 2.0, 1.0)
                                crown_diam = min(crown_diam, 15.0)
                                crown_area = float(np.pi * (crown_diam / 2.0) ** 2)
                                
                                if has_color:
                                    cr = int(np.mean(tr[tree_pts_idx]))
                                    cg = int(np.mean(tg[tree_pts_idx]))
                                    cb = int(np.mean(tb[tree_pts_idx]))
                                    color = f"{cr},{cg},{cb}"
                            
                            trees_detected.append({
                                'x': centroid_x,
                                'y': centroid_y,
                                'height': max(max_h, 1.0),
                                'crown_diam': crown_diam,
                                'crown_area': crown_area,
                                'point_count': point_count,
                                'density': float(point_count / max(crown_diam ** 2, 1)),
                                'color_rgb': color,
                                'type': 'tree'
                            })

            QgsMessageLog.logMessage(f"Trees detected: {len(trees_detected)}", "MyFlamma", Qgis.Info)
            self.setProgress(40)

            if self.isCanceled():
                return False

            # --- SHRUBS (Class 4 - Low Vegetation) ---
            shrubs_detected = []
            low_veg_mask = classification == 4
            
            if np.sum(low_veg_mask) > 20:
                self.setProgress(50)
                lvx = x[low_veg_mask]
                lvy = y[low_veg_mask]
                lvz = z[low_veg_mask]
                
                if has_color:
                    lvr = red[low_veg_mask]
                    lvg = green[low_veg_mask]
                    lvb = blue[low_veg_mask]

                # Normalize shrub heights
                norm_h_shrub = np.zeros(len(lvx), dtype=np.float32)
                if ground_tree:
                    ground_tree_2d = cKDTree(ground_xyz[:, :2])
                    batch = 50000
                    for i in range(0, len(lvx), batch):
                        end = min(i + batch, len(lvx))
                        qxy = np.column_stack((lvx[i:end], lvy[i:end]))
                        _, idxs = ground_tree_2d.query(qxy, k=1)
                        norm_h_shrub[i:end] = lvz[i:end] - ground_xyz[idxs, 2]
                else:
                    norm_h_shrub = lvz.copy()

                xy_shrub = np.column_stack((lvx, lvy))
                clustering = DBSCAN(eps=2.0, min_samples=3).fit(xy_shrub)
                labels = clustering.labels_

                for cluster_id in set(labels):
                    if cluster_id == -1:
                        continue
                    
                    mask = labels == cluster_id
                    if np.sum(mask) < 3:
                        continue

                    sx = float(np.mean(lvx[mask]))
                    sy = float(np.mean(lvy[mask]))
                    # Use percentile 75 for more realistic height
                    sh = float(np.percentile(norm_h_shrub[mask], 75))

                    # Filter: reject shrubs inside buildings (only very close)
                    if build_tree_2d is not None:
                        d_bld, _ = build_tree_2d.query([sx, sy], k=1)
                        if d_bld < 1.0:
                            continue

                    # Filter: reject shrubs on roads
                    if road_tree_2d is not None:
                        d_rd, _ = road_tree_2d.query([sx, sy], k=1)
                        if d_rd < MIN_DIST_TO_ROAD:
                            continue

                    if has_color:
                        sr = int(np.mean(lvr[mask]))
                        sg = int(np.mean(lvg[mask]))
                        sb = int(np.mean(lvb[mask]))
                        color = f"{sr},{sg},{sb}"
                    else:
                        color = "100,150,80"

                    # Arithmetic mean of spreads (consistent with main classifier)
                    x_spread = float(np.max(lvx[mask]) - np.min(lvx[mask]))
                    y_spread = float(np.max(lvy[mask]) - np.min(lvy[mask]))
                    diam = max((x_spread + y_spread) / 2.0, 0.5)

                    shrubs_detected.append({
                        'x': sx,
                        'y': sy,
                        'height': max(sh, 0.3),
                        'crown_diam': min(diam, 2.0),
                        'crown_area': np.pi * (diam / 2) ** 2,
                        'point_count': int(np.sum(mask)),
                        'density': float(np.sum(mask) / max(diam ** 2, 1)),
                        'color_rgb': color,
                        'type': 'shrub'
                    })

            QgsMessageLog.logMessage(f"Shrubs detected: {len(shrubs_detected)}", "MyFlamma", Qgis.Info)
            self.setProgress(60)

            if self.isCanceled():
                return False

            # --- RECLASSIFICATION: Class 2 → Class 3 (same as reclassified LAZ) ---
            # This replicates the exact same logic the reclassified LAZ uses,
            # so the OBJ matches it even without generating the reclassified file.
            if has_color:
                reclass_mask = classification == 2
                if np.sum(reclass_mask) > 0:
                    rc_r = red[reclass_mask]
                    rc_g = green[reclass_mask]
                    rc_b = blue[reclass_mask]
                    rc_exg = (2 * rc_g.astype(np.float32)) - rc_r.astype(np.float32) - rc_b.astype(np.float32)
                    rc_green_mask = (rc_exg > 12) & (rc_g > rc_r) & (rc_g > rc_b)
                    # Reclassify matching points: Class 2 → Class 3
                    reclass_indices = np.where(reclass_mask)[0]
                    classification[reclass_indices[rc_green_mask]] = 3
                    n_reclassified = int(np.sum(rc_green_mask))
                    QgsMessageLog.logMessage(
                        f"Reclassified {n_reclassified} ground points (Class 2→3) for grass",
                        "MyFlamma", Qgis.Info)

            # --- GRASS (Class 3 — original + reclassified) ---
            grass_detected = []
            grass_mask_all = classification == 3  # Now includes reclassified pts
            
            if np.sum(grass_mask_all) > 50 and has_color:
                self.setProgress(70)
                gx_f = x[grass_mask_all]
                gy_f = y[grass_mask_all]
                gr_f = red[grass_mask_all]
                gg_f = green[grass_mask_all]
                gb_f = blue[grass_mask_all]

                if len(gx_f) > 0:

                    grid_size = 2.0
                    grid = {}
                    for i in range(len(gx_f)):
                        col = int(gx_f[i] / grid_size)
                        row = int(gy_f[i] / grid_size)
                        key = (col, row)
                        if key not in grid:
                            grid[key] = []
                        grid[key].append(i)

                    cell_details = []
                    # Minimum 5 points per cell to count as grass
                    MIN_PTS_GRASS = 5
                    all_keys = {k for k, v in grid.items() if len(v) >= MIN_PTS_GRASS}

                    # (a) Remove isolated cells (no 4-connected neighbour)
                    connected_keys = set()
                    for key in all_keys:
                        c, r = key
                        for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            if (c + dc, r + dr) in all_keys:
                                connected_keys.add(key)
                                break

                    # (b) Fill small holes: non-grass cell with >=3 grass neighbours
                    filled_keys = set()
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
                                grid.setdefault((c, r), [])
                        connected_keys |= to_fill
                        filled_keys |= to_fill

                    for key in connected_keys:
                        indices = grid.get(key, [])
                        # Skip grass cells that overlap with road cells
                        if key in road_cell_keys:
                            continue
                        col, row = key
                        x_min = col * grid_size
                        x_max = x_min + grid_size
                        y_min = row * grid_size
                        y_max = y_min + grid_size

                        cell_details.append({
                            'x_min': float(x_min),
                            'y_min': float(y_min),
                            'x_max': float(x_max),
                            'y_max': float(y_max),
                            'size_m2': grid_size ** 2,
                            'color_rgb': (f"{int(np.mean(gr_f[indices]))},{int(np.mean(gg_f[indices]))},{int(np.mean(gb_f[indices]))}"
                                          if len(indices) > 0 else f"{int(np.mean(gr_f))},{int(np.mean(gg_f))},{int(np.mean(gb_f))}"),
                            'point_count': max(len(indices), 1),
                            'density_pts_m2': len(indices) / (grid_size ** 2)
                        })

                    if cell_details:
                        grass_detected.append({
                            'area': len(cell_details) * (grid_size ** 2),
                            'color_rgb': f"{int(np.mean(gr_f))},{int(np.mean(gg_f))},{int(np.mean(gb_f))}",
                            'point_count': len(gx_f),
                            'type': 'grass',
                            'cell_details': cell_details
                        })

            QgsMessageLog.logMessage(f"Grass areas detected: {len(grass_detected)}", "MyFlamma", Qgis.Info)
            self.setProgress(80)

            if self.isCanceled():
                return False

            # --- BUILDINGS (Class 6) ---
            buildings_detected = []
            # building_mask already defined above for proximity filtering
            
            if np.sum(building_mask) > 10:
                self.setProgress(85)
                bx = x[building_mask]
                by = y[building_mask]
                bz = z[building_mask]
                
                if has_color:
                    br = red[building_mask]
                    bg = green[building_mask]
                    bb = blue[building_mask]

                # Normalize building height relative to ground
                norm_h_build = np.zeros(len(bx), dtype=np.float32)
                if ground_tree:
                    ground_tree_2d = cKDTree(ground_xyz[:, :2])
                    batch = 50000
                    for i in range(0, len(bx), batch):
                        end = min(i + batch, len(bx))
                        qxy = np.column_stack((bx[i:end], by[i:end]))
                        _, idxs = ground_tree_2d.query(qxy, k=1)
                        norm_h_build[i:end] = bz[i:end] - ground_xyz[idxs, 2]
                else:
                    norm_h_build = bz.copy()

                xy_build = np.column_stack((bx, by))
                clustering = DBSCAN(eps=3.0, min_samples=10).fit(xy_build)
                labels = clustering.labels_

                for cluster_id in set(labels):
                    if cluster_id == -1:
                        continue
                    
                    mask = labels == cluster_id
                    if np.sum(mask) < 10:
                        continue
                    
                    bx_c = bx[mask]
                    by_c = by[mask]
                    bh_c = norm_h_build[mask]  # Use normalized height
                    if has_color:
                        br_c = br[mask]
                        bg_c = bg[mask]
                        bb_c = bb[mask]

                    grid_size = 2.0
                    grid = {}
                    for i in range(len(bx_c)):
                        col = int(bx_c[i] / grid_size)
                        row = int(by_c[i] / grid_size)
                        key = (col, row)
                        if key not in grid:
                            grid[key] = []
                        grid[key].append(i)

                    cell_details = []
                    for key, indices in grid.items():
                        col, row = key
                        x_min = col * grid_size
                        x_max = x_min + grid_size
                        y_min = row * grid_size
                        y_max = y_min + grid_size

                        h_max = np.max(bh_c[indices])
                        h_mean = np.mean(bh_c[indices])

                        cell_details.append({
                            'x_min': float(x_min),
                            'y_min': float(y_min),
                            'x_max': float(x_max),
                            'y_max': float(y_max),
                            'size_m2': grid_size ** 2,
                            'height_max': max(float(h_max), 1.0),
                            'height_mean': float(h_mean),
                            'point_count': len(indices),
                            'color_rgb': f"{int(np.mean(br_c[indices]))},{int(np.mean(bg_c[indices]))},{int(np.mean(bb_c[indices]))}" if has_color else "150,150,150"
                        })

                    if cell_details:
                        buildings_detected.append({
                            'x': float(np.mean(bx_c)),
                            'y': float(np.mean(by_c)),
                            'height_max': max(float(np.max(bh_c)), 1.0),
                            'height_mean': float(np.mean(bh_c)),
                            'type': 'building',
                            'point_count': len(bx_c),
                            'footprint_m2': len(cell_details) * (grid_size ** 2),
                            'color_rgb': f"{int(np.mean(br_c))},{int(np.mean(bg_c))},{int(np.mean(bb_c))}" if has_color else "150,150,150",
                            'cell_details': cell_details
                        })

            QgsMessageLog.logMessage(f"Buildings detected: {len(buildings_detected)}", "MyFlamma", Qgis.Info)
            self.setProgress(90)

            # --- TERRAIN DEM GRID (from ground points, class 2) ---
            terrain_grid = None
            if len(ground_xyz) > 50:
                QgsMessageLog.logMessage("Generating terrain DEM grid for OBJ export...", "MyFlamma", Qgis.Info)
                terrain_cell = 2.0  # 2m resolution grid

                gx_t = ground_xyz[:, 0]
                gy_t = ground_xyz[:, 1]
                gz_t = ground_xyz[:, 2]

                t_x_min = float(np.min(gx_t))
                t_x_max = float(np.max(gx_t))
                t_y_min = float(np.min(gy_t))
                t_y_max = float(np.max(gy_t))

                t_cols = int(np.ceil((t_x_max - t_x_min) / terrain_cell)) + 1
                t_rows = int(np.ceil((t_y_max - t_y_min) / terrain_cell)) + 1

                # Limit grid size to avoid memory issues (max ~2000x2000)
                if t_cols > 2000 or t_rows > 2000:
                    terrain_cell = max((t_x_max - t_x_min) / 2000, (t_y_max - t_y_min) / 2000, terrain_cell)
                    t_cols = int(np.ceil((t_x_max - t_x_min) / terrain_cell)) + 1
                    t_rows = int(np.ceil((t_y_max - t_y_min) / terrain_cell)) + 1

                # Grid: accumulate heights per cell (use mean Z)
                z_sum = np.zeros((t_rows, t_cols), dtype=np.float64)
                z_count = np.zeros((t_rows, t_cols), dtype=np.int32)

                col_idx = np.clip(((gx_t - t_x_min) / terrain_cell).astype(int), 0, t_cols - 1)
                row_idx = np.clip(((gy_t - t_y_min) / terrain_cell).astype(int), 0, t_rows - 1)

                for i in range(len(gx_t)):
                    c, r = col_idx[i], row_idx[i]
                    z_sum[r, c] += gz_t[i]
                    z_count[r, c] += 1

                # Mean height where we have data
                valid = z_count > 0
                z_mean = np.zeros_like(z_sum)
                z_mean[valid] = z_sum[valid] / z_count[valid]

                # Fill holes (cells with no ground points) using nearest valid cell
                if np.sum(~valid) > 0 and np.sum(valid) > 0:
                    from scipy.ndimage import distance_transform_edt
                    # distance_transform_edt with indices gives us the nearest valid cell
                    _, nearest_idx = distance_transform_edt(~valid, return_distances=True, return_indices=True)
                    z_mean = z_mean[nearest_idx[0], nearest_idx[1]]

                # Store the minimum elevation to normalize later
                z_base = float(np.min(z_mean[valid])) if np.sum(valid) > 0 else 0.0

                terrain_grid = {
                    'x_min': t_x_min,
                    'x_max': t_x_max,
                    'y_min': t_y_min,
                    'y_max': t_y_max,
                    'cell_size': terrain_cell,
                    'cols': t_cols,
                    'rows': t_rows,
                    'z_grid': z_mean.tolist(),  # 2D list of elevations
                    'z_base': z_base,           # minimum elevation for normalization
                }
                QgsMessageLog.logMessage(
                    f"Terrain DEM: {t_cols}x{t_rows} cells, cell={terrain_cell:.1f}m, "
                    f"Z range: {z_base:.1f} - {float(np.max(z_mean)):.1f}m",
                    "MyFlamma", Qgis.Info
                )

            self.setProgress(95)

            # --- SAVE TO PLUGIN ---
            self.plugin.last_classification_data = {
                'filename': self.filename,
                'trees': trees_detected,
                'shrubs': shrubs_detected,
                'grass': grass_detected,
                'buildings': buildings_detected,
                'roads': roads_detected,
                'terrain': terrain_grid
            }

            self.setProgress(100)
            return True

        except Exception as e:
            self.error_msg = str(e)
            import traceback
            QgsMessageLog.logMessage(
                f"Quick Classify error: {e}\n{traceback.format_exc()}",
                "MyFlamma", Qgis.Critical
            )
            return False

    def finished(self, result):
        if result:
            QgsMessageLog.logMessage("Quick classification completed. Ready to export.", "MyFlamma", Qgis.Success)
        else:
            QgsMessageLog.logMessage(f"Quick classification failed: {self.error_msg}", "MyFlamma", Qgis.Critical)
