# -*- coding: utf-8 -*-
"""
OBJ Export - Generates a 3D .obj + .mtl from in-memory classified data.

Geometry conventions (Y-up for Unity):
  - LiDAR X → OBJ X
  - LiDAR Y → OBJ Z  (depth)
  - Height  → OBJ Y  (up)

Shapes:
  Ground   - triangulated terrain mesh with real elevation (double-sided)
  Grass    - terrain faces painted green (same mesh, different material)
  Shrubs   - cubes (width = crown_diam, height = Height_m)
  Trees    - rectangular trunk + cube crown
  Buildings - extruded boxes per cell (width from cell bounds, height from Height_Max_m)
"""

import os

from qgis.core import QgsTask, QgsApplication, QgsMessageLog, Qgis


# ---------------------------------------------------------------------------
#  Helper: simple OBJ geometry writer
# ---------------------------------------------------------------------------

class ObjWriter:
    """Accumulates vertices, normals and faces, then writes .obj + .mtl."""

    def __init__(self):
        self.vertices = []      # (x, y, z)
        self.normals = []       # (nx, ny, nz)
        self.groups = []        # (group_name, material_name, faces[])
        # faces: list of tuples of (v_idx, vn_idx) - 1-based
        self._current_group = None

    # --- low-level helpers ---
    def _vi(self, x, y, z):
        """Add a vertex and return its 1-based index."""
        self.vertices.append((x, y, z))
        return len(self.vertices)

    def _ni(self, nx, ny, nz):
        """Add a normal and return its 1-based index."""
        self.normals.append((nx, ny, nz))
        return len(self.normals)

    def begin_group(self, name, material):
        self._current_group = {
            'name': name,
            'material': material,
            'faces': []
        }
        self.groups.append(self._current_group)

    def _add_face(self, verts_normals):
        """verts_normals: list of (v_idx, vn_idx)"""
        self._current_group['faces'].append(verts_normals)

    # --- high-level shape builders ---

    def add_quad(self, x0, z0, x1, z1, y=0.0):
        """Double-sided flat horizontal quad (visible from both sides)."""
        n_up = self._ni(0, 1, 0)
        n_down = self._ni(0, -1, 0)
        v1 = self._vi(x0, y, z0)
        v2 = self._vi(x1, y, z0)
        v3 = self._vi(x1, y, z1)
        v4 = self._vi(x0, y, z1)
        # Top face (CCW seen from above)
        self._add_face([(v1, n_up), (v4, n_up), (v3, n_up), (v2, n_up)])
        # Bottom face (CCW seen from below)
        self._add_face([(v1, n_down), (v2, n_down), (v3, n_down), (v4, n_down)])

    def add_inclined_quad(self, v1, v2, v3, v4):
        """Double-sided inclined quad with per-corner elevation.
        v1..v4 are (x, y, z) tuples; v1=front-left, v2=front-right,
        v3=back-right, v4=back-left.  Winding is CCW from above."""
        i1 = self._vi(*v1)
        i2 = self._vi(*v2)
        i3 = self._vi(*v3)
        i4 = self._vi(*v4)

        # Normal via cross(v4-v1, v2-v1) → points up for CCW face
        e1 = (v4[0]-v1[0], v4[1]-v1[1], v4[2]-v1[2])
        e2 = (v2[0]-v1[0], v2[1]-v1[1], v2[2]-v1[2])
        nx = e1[1]*e2[2] - e1[2]*e2[1]
        ny = e1[2]*e2[0] - e1[0]*e2[2]
        nz = e1[0]*e2[1] - e1[1]*e2[0]
        ln = (nx**2 + ny**2 + nz**2)**0.5
        if ln > 0:
            nx, ny, nz = nx/ln, ny/ln, nz/ln
        else:
            nx, ny, nz = 0, 1, 0
        n_top = self._ni(nx, ny, nz)
        n_bot = self._ni(-nx, -ny, -nz)

        # Top face (CCW from above)
        self._add_face([(i1, n_top), (i4, n_top), (i3, n_top), (i2, n_top)])
        # Bottom face (reversed winding)
        self._add_face([(i1, n_bot), (i2, n_bot), (i3, n_bot), (i4, n_bot)])

    def add_box(self, x_min, z_min, x_max, z_max, y_base, y_top):
        """Closed axis-aligned box with 6 faces, all CCW winding (outward normals)."""
        # 8 vertices
        v = [
            self._vi(x_min, y_base, z_min),  # 0 bottom-front-left
            self._vi(x_max, y_base, z_min),  # 1 bottom-front-right
            self._vi(x_max, y_base, z_max),  # 2 bottom-back-right
            self._vi(x_min, y_base, z_max),  # 3 bottom-back-left
            self._vi(x_min, y_top,  z_min),  # 4 top-front-left
            self._vi(x_max, y_top,  z_min),  # 5 top-front-right
            self._vi(x_max, y_top,  z_max),  # 6 top-back-right
            self._vi(x_min, y_top,  z_max),  # 7 top-back-left
        ]
        # 6 normals
        n_down  = self._ni( 0, -1,  0)
        n_up    = self._ni( 0,  1,  0)
        n_front = self._ni( 0,  0, -1)
        n_back  = self._ni( 0,  0,  1)
        n_left  = self._ni(-1,  0,  0)
        n_right = self._ni( 1,  0,  0)

        # All faces CCW when viewed from outside
        # bottom (normal -Y, viewed from below: CCW)
        self._add_face([(v[0], n_down), (v[1], n_down), (v[2], n_down), (v[3], n_down)])
        # top (normal +Y, viewed from above: CCW)
        self._add_face([(v[4], n_up), (v[7], n_up), (v[6], n_up), (v[5], n_up)])
        # front (normal -Z, viewed from front: CCW)
        self._add_face([(v[0], n_front), (v[4], n_front), (v[5], n_front), (v[1], n_front)])
        # back (normal +Z, viewed from back: CCW)
        self._add_face([(v[2], n_back), (v[6], n_back), (v[7], n_back), (v[3], n_back)])
        # left (normal -X, viewed from left: CCW)
        self._add_face([(v[3], n_left), (v[7], n_left), (v[4], n_left), (v[0], n_left)])
        # right (normal +X, viewed from right: CCW)
        self._add_face([(v[1], n_right), (v[5], n_right), (v[6], n_right), (v[2], n_right)])

    # --- write to disk ---

    def write(self, obj_path, mtl_materials):
        """
        Write .obj and .mtl files.
        mtl_materials: dict  name → {'Kd': (r,g,b), 'd': alpha}
        """
        mtl_path = os.path.splitext(obj_path)[0] + '.mtl'
        mtl_name = os.path.basename(mtl_path)

        # --- Write .mtl ---
        with open(mtl_path, 'w', encoding='utf-8') as f:
            f.write("# MyFlamma OBJ Export - Materials\n")
            for mat_name, props in mtl_materials.items():
                kd = props.get('Kd', (0.5, 0.5, 0.5))
                alpha = props.get('d', 1.0)
                f.write(f"\nnewmtl {mat_name}\n")
                f.write(f"Kd {kd[0]:.4f} {kd[1]:.4f} {kd[2]:.4f}\n")
                f.write(f"Ka {kd[0]*0.2:.4f} {kd[1]*0.2:.4f} {kd[2]*0.2:.4f}\n")
                f.write(f"d {alpha:.2f}\n")
                f.write("illum 1\n")

        # --- Write .obj ---
        with open(obj_path, 'w', encoding='utf-8') as f:
            f.write("# MyFlamma OBJ Export\n")
            f.write(f"mtllib {mtl_name}\n\n")

            # vertices
            for vx, vy, vz in self.vertices:
                f.write(f"v {vx:.4f} {vy:.4f} {vz:.4f}\n")
            f.write("\n")

            # normals
            for nx, ny, nz in self.normals:
                f.write(f"vn {nx:.4f} {ny:.4f} {nz:.4f}\n")
            f.write("\n")

            # groups + faces
            for grp in self.groups:
                f.write(f"g {grp['name']}\n")
                f.write(f"usemtl {grp['material']}\n")
                for face in grp['faces']:
                    indices = " ".join(f"{vi}//{ni}" for vi, ni in face)
                    f.write(f"f {indices}\n")
                f.write("\n")


# ---------------------------------------------------------------------------
#  Main export task
# ---------------------------------------------------------------------------

class ObjExportTask(QgsTask):
    """Background task that generates .obj + .mtl from in-memory data."""

    def __init__(self, description, trees, shrubs, grass, buildings, params, output_path, plugin_ref, translator, terrain=None):
        super().__init__(description, QgsTask.CanCancel)
        self.trees = trees or []
        self.shrubs = shrubs or []
        self.grass = grass or []
        self.buildings = buildings or []
        self.terrain = terrain  # terrain DEM dict or None
        self.params = params
        self.output_path = output_path
        self.plugin = plugin_ref
        self.tr = translator
        self.error_msg = None

    def run(self):
        try:
            return self._generate()
        except Exception as e:
            self.error_msg = str(e)
            import traceback
            QgsMessageLog.logMessage(
                f"OBJ Export error: {e}\n{traceback.format_exc()}",
                "MyFlamma", Qgis.Critical
            )
            return False

    def _generate(self):
        import numpy as np

        p = self.params
        scale = p.get('scale', 1.0)
        padding = p.get('ground_padding', 10) * scale
        include_terrain = p.get('include_terrain', True)

        total = len(self.trees) + len(self.shrubs) + len(self.grass) + len(self.buildings)
        if total == 0:
            self.error_msg = "No classification data available."
            return False

        QgsMessageLog.logMessage(
            f"OBJ Export: {len(self.trees)} trees, {len(self.shrubs)} shrubs, "
            f"{len(self.grass)} grass areas, {len(self.buildings)} buildings, "
            f"terrain={'yes' if self.terrain else 'no'}",
            "MyFlamma", Qgis.Info
        )

        # --- Determine scene bounding box ---
        all_x = []
        all_z = []  # LiDAR Y → OBJ Z

        for tree in self.trees:
            all_x.append(tree.get('x', 0))
            all_z.append(tree.get('y', 0))
        for shrub in self.shrubs:
            all_x.append(shrub.get('x', 0))
            all_z.append(shrub.get('y', 0))
        for grass in self.grass:
            for cell in grass.get('cell_details', []):
                all_x.append(cell['x_min'])
                all_x.append(cell['x_max'])
                all_z.append(cell['y_min'])
                all_z.append(cell['y_max'])
        for building in self.buildings:
            for cell in building.get('cell_details', []):
                all_x.append(cell['x_min'])
                all_x.append(cell['x_max'])
                all_z.append(cell['y_min'])
                all_z.append(cell['y_max'])
        if not all_x:
            self.error_msg = "Could not determine scene bounds."
            return False

        # Use a local origin (offset) so coordinates are small
        origin_x = min(all_x)
        origin_z = max(all_z)   # MAX so we can flip Z: (origin_z - y)

        scene_x_min = 0.0
        scene_z_min = 0.0
        scene_x_max = (max(all_x) - origin_x) * scale
        scene_z_max = (origin_z - min(all_z)) * scale  # flipped

        # --- Terrain elevation helper ---
        # Build a numpy array for fast terrain lookups
        terrain_grid_np = None
        terrain_z_base = 0.0
        terrain_x_min = 0.0
        terrain_y_min = 0.0
        terrain_cell = 2.0
        terrain_cols = 0
        terrain_rows = 0

        if self.terrain and include_terrain:
            terrain_z_base = self.terrain['z_base']
            terrain_x_min = self.terrain['x_min']
            terrain_y_min = self.terrain['y_min']
            terrain_cell = self.terrain['cell_size']
            terrain_cols = self.terrain['cols']
            terrain_rows = self.terrain['rows']
            terrain_grid_np = np.array(self.terrain['z_grid'], dtype=np.float64)

        def get_terrain_elev(lidar_x, lidar_y):
            """Get terrain elevation at a LiDAR coordinate using bilinear
            interpolation, returns OBJ Y value.  This ensures grass quads
            track the actual terrain surface on steep slopes."""
            if terrain_grid_np is None:
                return 0.0
            # Fractional grid position
            fc = (lidar_x - terrain_x_min) / terrain_cell
            fr = (lidar_y - terrain_y_min) / terrain_cell
            # Integer corners (clamped)
            c0 = max(0, min(int(np.floor(fc)), terrain_cols - 2))
            r0 = max(0, min(int(np.floor(fr)), terrain_rows - 2))
            c1 = c0 + 1
            r1 = r0 + 1
            # Fractional part
            t = fc - c0  # 0..1 along column axis
            s = fr - r0  # 0..1 along row axis
            t = max(0.0, min(t, 1.0))
            s = max(0.0, min(s, 1.0))
            # Bilinear interpolation
            z00 = terrain_grid_np[r0, c0]
            z01 = terrain_grid_np[r0, c1]
            z10 = terrain_grid_np[r1, c0]
            z11 = terrain_grid_np[r1, c1]
            z_interp = (1 - s) * ((1 - t) * z00 + t * z01) + s * ((1 - t) * z10 + t * z11)
            return (z_interp - terrain_z_base) * scale

        writer = ObjWriter()

        # --- Materials ---
        materials = {
            'mat_ground':       {'Kd': (0.12, 0.12, 0.12), 'd': 1.0},   # very dark grey
            'mat_grass':        {'Kd': (0.20, 0.65, 0.15), 'd': 1.0},   # green
            'mat_shrub':        {'Kd': (0.08, 0.35, 0.05), 'd': 1.0},   # very dark green
            'mat_trunk':        {'Kd': (0.45, 0.28, 0.10), 'd': 1.0},   # brown
            'mat_crown':        {'Kd': (0.10, 0.55, 0.10), 'd': 1.0},   # green
            'mat_building':     {'Kd': (0.95, 0.64, 0.14), 'd': 1.0},   # orange
        }

        progress_done = 0
        progress_total = (1 if p.get('include_ground') else 0) + \
                         len(self.grass) + len(self.shrubs) + len(self.trees) + \
                         len(self.buildings)
        if progress_total == 0:
            progress_total = 1

        # =====================================================================
        #  1. GROUND / TERRAIN MESH  +  2. GRASS (painted on terrain)
        # =====================================================================
        #  Grass is NOT a separate layer. Instead, terrain faces that fall
        #  inside a classified grass cell are assigned mat_grass. This
        #  guarantees 100 % alignment (same vertices, same triangles).
        #
        #  Before painting, we:
        #   a) Remove isolated grass cells (no 4-connected neighbour)
        #   b) Fill small holes (non-grass cell surrounded by ≥3 grass
        #      neighbours in 4-connectivity)
        # =====================================================================

        # --- Build grass lookup set in terrain-grid coordinates ----------
        grass_terrain_keys = set()   # (terrain_col, terrain_row)

        if p.get('include_grass', True) and self.grass and terrain_grid_np is not None:
            for grass_area in self.grass:
                for cell in grass_area.get('cell_details', []):
                    # Map the grass cell centre to the terrain grid
                    cx = (cell['x_min'] + cell['x_max']) / 2.0
                    cy = (cell['y_min'] + cell['y_max']) / 2.0
                    tc = int((cx - terrain_x_min) / terrain_cell)
                    tr = int((cy - terrain_y_min) / terrain_cell)
                    tc = max(0, min(tc, terrain_cols - 2))
                    tr = max(0, min(tr, terrain_rows - 2))
                    grass_terrain_keys.add((tc, tr))

            # (a) Remove isolated cells (must have ≥1 neighbour in 4-conn.)
            filtered = set()
            for (c, r) in grass_terrain_keys:
                neighbours = 0
                for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    if (c + dc, r + dr) in grass_terrain_keys:
                        neighbours += 1
                if neighbours >= 1:
                    filtered.add((c, r))
            grass_terrain_keys = filtered

            # (b) Fill small holes: if a non-grass cell has ≥3 grass
            #     neighbours, fill it in.  Repeat twice for cascade fills.
            for _ in range(2):
                to_fill = set()
                # Only check cells that are adjacent to existing grass
                candidates = set()
                for (c, r) in grass_terrain_keys:
                    for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nb = (c + dc, r + dr)
                        if nb not in grass_terrain_keys:
                            if 0 <= nb[0] < terrain_cols - 1 and 0 <= nb[1] < terrain_rows - 1:
                                candidates.add(nb)
                for (c, r) in candidates:
                    grass_n = 0
                    for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        if (c + dc, r + dr) in grass_terrain_keys:
                            grass_n += 1
                    if grass_n >= 3:
                        to_fill.add((c, r))
                grass_terrain_keys |= to_fill

            QgsMessageLog.logMessage(
                f"Grass terrain cells: {len(grass_terrain_keys)} (after isolate-removal + hole-fill)",
                "MyFlamma", Qgis.Info
            )

        # --- Build the terrain mesh ----------------------------------------
        if p.get('include_ground', True):
            if terrain_grid_np is not None and include_terrain:
                QgsMessageLog.logMessage(
                    f"Building terrain mesh: {terrain_cols}x{terrain_rows} vertices",
                    "MyFlamma", Qgis.Info
                )

                # Create vertex grid (row, col) → vertex index (shared by both materials)
                vert_indices = np.zeros((terrain_rows, terrain_cols), dtype=np.int32)

                for r in range(terrain_rows):
                    if self.isCanceled():
                        return False
                    for c in range(terrain_cols):
                        lidar_x = terrain_x_min + c * terrain_cell
                        lidar_y = terrain_y_min + r * terrain_cell

                        obj_x = (lidar_x - origin_x) * scale
                        obj_z = (origin_z - lidar_y) * scale  # flipped
                        obj_y = (terrain_grid_np[r, c] - terrain_z_base) * scale

                        vert_indices[r, c] = writer._vi(obj_x, obj_y, obj_z)

                # We need two groups so we can start one, switch to the other,
                # and keep adding faces.  Pre-create both groups.
                writer.begin_group('Ground', 'mat_ground')
                ground_group = writer._current_group
                writer.begin_group('Grass', 'mat_grass')
                grass_group = writer._current_group

                n_up = writer._ni(0, 1, 0)
                n_down = writer._ni(0, -1, 0)

                for r in range(terrain_rows - 1):
                    if self.isCanceled():
                        return False
                    for c in range(terrain_cols - 1):
                        v00 = vert_indices[r, c]
                        v10 = vert_indices[r + 1, c]
                        v11 = vert_indices[r + 1, c + 1]
                        v01 = vert_indices[r, c + 1]

                        # Choose material: grass or ground
                        if (c, r) in grass_terrain_keys:
                            target = grass_group
                        else:
                            target = ground_group

                        # Top face (CCW from above) - two triangles
                        target['faces'].append([(v00, n_up), (v01, n_up), (v11, n_up)])
                        target['faces'].append([(v00, n_up), (v11, n_up), (v10, n_up)])
                        # Bottom face (reversed winding, visible from below)
                        target['faces'].append([(v00, n_down), (v11, n_down), (v01, n_down)])
                        target['faces'].append([(v00, n_down), (v10, n_down), (v11, n_down)])

                # Remove grass group if it ended up empty
                if not grass_group['faces']:
                    writer.groups.remove(grass_group)
                # Remove ground group if it ended up empty (unlikely)
                if not ground_group['faces']:
                    writer.groups.remove(ground_group)

                QgsMessageLog.logMessage("Terrain mesh built (ground + grass painted).", "MyFlamma", Qgis.Info)
            else:
                # Flat ground plane fallback
                writer.begin_group('Ground', 'mat_ground')
                writer.add_quad(
                    scene_x_min - padding, scene_z_min - padding,
                    scene_x_max + padding, scene_z_max + padding,
                    y=0.0
                )

            progress_done += 1
            self.setProgress(int(progress_done / progress_total * 100))

        if self.isCanceled():
            return False

        # =====================================================================
        #  2. SHRUBS - small green cubes at terrain elevation
        # =====================================================================
        if p.get('include_shrubs', True) and self.shrubs:
            writer.begin_group('Shrubs', 'mat_shrub')
            for shrub in self.shrubs:
                if self.isCanceled():
                    return False

                sx = (shrub.get('x', 0) - origin_x) * scale
                sz = (origin_z - shrub.get('y', 0)) * scale  # flipped
                elev = get_terrain_elev(shrub.get('x', 0), shrub.get('y', 0))
                # Limitar altura a máximo 2m para arbustos
                s_height = min(max(shrub.get('height', 0.3), 0.2), 2.0) * scale
                # Arbustos típicamente son 0.3-1.5m de diámetro
                s_diam = min(max(shrub.get('crown_diam', 0.5), 0.3), 1.5) * scale
                half = s_diam / 2.0

                writer.add_box(
                    sx - half, sz - half,
                    sx + half, sz + half,
                    y_base=elev,
                    y_top=elev + s_height
                )

                progress_done += 1
                if progress_done % 200 == 0:
                    self.setProgress(int(progress_done / progress_total * 100))

        if self.isCanceled():
            return False

        # =====================================================================
        #  3. TREES - rectangular trunk + cube crown at terrain elevation
        # =====================================================================
        if p.get('include_trees', True) and self.trees:
            for tree in self.trees:
                if self.isCanceled():
                    return False

                tx = (tree.get('x', 0) - origin_x) * scale
                tz = (origin_z - tree.get('y', 0)) * scale  # flipped
                elev = get_terrain_elev(tree.get('x', 0), tree.get('y', 0))
                t_height = max(tree.get('height', 1.0), 1.0) * scale
                t_crown_diam = max(tree.get('crown_diam', 0.5), 0.5) * scale

                # Trunk: narrow rectangle from ground to crown base
                trunk_width = max(t_crown_diam * 0.15, 0.15 * scale)
                trunk_half = trunk_width / 2.0
                crown_height = t_crown_diam * 0.8  # Crown is roughly spherical
                trunk_top = t_height - crown_height

                if trunk_top < 0.5 * scale:
                    trunk_top = t_height * 0.4

                tree_id = tree.get('Tree_ID', tree.get('Tree_ID', 'x'))
                # Trunk
                writer.begin_group(f'Tree_{tree_id}_trunk', 'mat_trunk')
                writer.add_box(
                    tx - trunk_half, tz - trunk_half,
                    tx + trunk_half, tz + trunk_half,
                    y_base=elev,
                    y_top=elev + trunk_top
                )

                # Crown: cube at top
                crown_half = t_crown_diam / 2.0
                writer.begin_group(f'Tree_{tree_id}_crown', 'mat_crown')
                writer.add_box(
                    tx - crown_half, tz - crown_half,
                    tx + crown_half, tz + crown_half,
                    y_base=elev + trunk_top,
                    y_top=elev + t_height
                )

                progress_done += 1
                if progress_done % 100 == 0:
                    self.setProgress(int(progress_done / progress_total * 100))

        if self.isCanceled():
            return False

        # =====================================================================
        #  4. BUILDINGS - extruded boxes at terrain elevation
        # =====================================================================
        if p.get('include_buildings', True) and self.buildings:
            for building in self.buildings:
                if self.isCanceled():
                    return False

                bid = building.get('x', 'x')
                writer.begin_group(f'Building_walls', 'mat_building')

                for cell in building.get('cell_details', []):
                    cx = (cell['x_min'] + cell['x_max']) / 2.0
                    cy = (cell['y_min'] + cell['y_max']) / 2.0
                    elev = get_terrain_elev(cx, cy)

                    bx_min = (cell['x_min'] - origin_x) * scale
                    bx_max = (cell['x_max'] - origin_x) * scale
                    bz_min = (origin_z - cell['y_max']) * scale  # flipped
                    bz_max = (origin_z - cell['y_min']) * scale  # flipped
                    b_height = min(max(cell.get('height_max', 3), 1.0), 50.0) * scale

                    writer.add_box(
                        bx_min, bz_min,
                        bx_max, bz_max,
                        y_base=elev,
                        y_top=elev + b_height
                    )

                progress_done += 1
                if progress_done % 50 == 0:
                    self.setProgress(int(progress_done / progress_total * 100))

        self.setProgress(95)

        # =====================================================================
        #  Write to disk
        # =====================================================================
        writer.write(self.output_path, materials)

        num_verts = len(writer.vertices)
        num_faces = sum(len(g['faces']) for g in writer.groups)
        QgsMessageLog.logMessage(
            f"OBJ Export complete: {num_verts} vertices, {num_faces} faces → {self.output_path}",
            "MyFlamma", Qgis.Success
        )
        self.setProgress(100)
        return True

    def finished(self, result):
        from qgis.PyQt.QtWidgets import QMessageBox
        if result:
            mtl_path = os.path.splitext(self.output_path)[0] + '.mtl'
            msg = (
                f"OBJ exported successfully!\n\n"
                f"Files:\n"
                f"  • {self.output_path}\n"
                f"  • {mtl_path}\n\n"
                f"You can import these into Unity, Blender, or any 3D tool."
            )
            self.plugin.iface.messageBar().pushMessage(
                "OBJ Export", "Export completed successfully!",
                level=Qgis.Success, duration=8
            )
            QgsMessageLog.logMessage(msg, "MyFlamma", Qgis.Success)
        else:
            err = self.error_msg or "Unknown error"
            self.plugin.iface.messageBar().pushMessage(
                "OBJ Export", f"Export failed: {err}",
                level=Qgis.Critical, duration=10
            )


# ---------------------------------------------------------------------------
#  Entry point (called from my_lidar.py)
# ---------------------------------------------------------------------------

def export_to_obj(plugin_ref):
    """Show dialog to select input file + export options, then process."""
    from .obj_export_dialog import ObjExportDialog
    from .quick_classify import QuickClassifyTask

    # Always show dialog with file selector
    dlg = ObjExportDialog(plugin_ref.iface.mainWindow(), translator=plugin_ref.tr, require_input=True, iface=plugin_ref.iface)

    if dlg.exec_() != ObjExportDialog.Accepted:
        return

    input_file = dlg.get_input_file()
    params = dlg.get_params()
    output_path = dlg.get_output_file()

    if not input_file:
        from qgis.PyQt.QtWidgets import QMessageBox
        QMessageBox.warning(plugin_ref.iface.mainWindow(), plugin_ref.tr("Error"),
                            plugin_ref.tr("Please select a LAS/LAZ file."))
        return

    if not output_path:
        from qgis.PyQt.QtWidgets import QMessageBox
        QMessageBox.warning(plugin_ref.iface.mainWindow(), plugin_ref.tr("Error"),
                            plugin_ref.tr("Please choose an output OBJ file."))
        return

    # Normalize path for comparison
    input_file_norm = os.path.normpath(input_file)

    # Check if cached data matches the selected file
    has_matching_data = (
        hasattr(plugin_ref, 'last_classification_data')
        and plugin_ref.last_classification_data is not None
        and os.path.normpath(plugin_ref.last_classification_data.get('filename', '')) == input_file_norm
    )

    if has_matching_data:
        # Cached data matches selected file, export directly
        _do_obj_export(plugin_ref, params, output_path=output_path)
    else:
        # Different file or no cache: run quick classify first
        task_desc = f"Classifying {os.path.basename(input_file)} for OBJ export"
        task = QuickClassifyTask(task_desc, input_file, plugin_ref, plugin_ref.tr)

        # Store callback to launch OBJ export after classification
        original_finished = task.finished
        def on_classify_done(result):
            original_finished(result)
            if result:
                _do_obj_export(plugin_ref, params, output_path=output_path)
        task.finished = on_classify_done

        plugin_ref.running_tasks.append(task)
        QgsApplication.taskManager().addTask(task)

        plugin_ref.iface.messageBar().pushMessage(
            plugin_ref.tr("Task Started"),
            plugin_ref.tr("Classifying LAS file for OBJ export..."),
            level=Qgis.Info, duration=-1
        )


def _do_obj_export(plugin_ref, params, output_path=None):
    """Launch the OBJ export task."""

    data = plugin_ref.last_classification_data

    if output_path is None:
        output_path = params.get('output_obj')
    if output_path is None:
        # Auto-generate output path in project directory
        base_path = os.path.splitext(data['filename'])[0]
        output_path = base_path + "_3d_scene.obj"
    if not os.path.splitext(output_path)[1].lower() == '.obj':
        output_path += '.obj'

    task_desc = f"Exporting to OBJ → {os.path.basename(output_path)}"
    task = ObjExportTask(
        task_desc,
        data.get('trees', []),
        data.get('shrubs', []),
        data.get('grass', []),
        data.get('buildings', []),
        params,
        output_path,
        plugin_ref,
        plugin_ref.tr,
        terrain=data.get('terrain', None)
    )
    plugin_ref.running_tasks.append(task)
    QgsApplication.taskManager().addTask(task)

    plugin_ref.iface.messageBar().pushMessage(
        plugin_ref.tr("Task Started"),
        plugin_ref.tr("Exporting 3D scene to OBJ in the background..."),
        level=Qgis.Info, duration=-1
    )
