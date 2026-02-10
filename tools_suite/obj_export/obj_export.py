# -*- coding: utf-8 -*-
"""
OBJ Export - Generates a 3D .obj + .mtl from in-memory classified data.

Geometry conventions (Y-up for Unity):
  - LiDAR X → OBJ X
  - LiDAR Y → OBJ Z  (depth)
  - Height   → OBJ Y  (up)

Shapes:
  Ground   – flat quad at Y=0
  Grass    – grid of 2m×2m quads at Y=0.05 (slightly above ground)
  Shrubs   – cubes (width = crown_diam, height = Height_m)
  Trees    – rectangular trunk + cube crown
  Buildings – extruded boxes per cell (width from cell bounds, height from Height_Max_m)
"""

import os
from collections import defaultdict

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
        # faces: list of tuples of (v_idx, vn_idx) – 1-based
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

    def __init__(self, description, trees, shrubs, grass, buildings, roads, params, output_path, plugin_ref, translator):
        super().__init__(description, QgsTask.CanCancel)
        self.trees = trees or []
        self.shrubs = shrubs or []
        self.grass = grass or []
        self.buildings = buildings or []
        self.roads = roads or []
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
        p = self.params
        scale = p.get('scale', 1.0)
        padding = p.get('ground_padding', 10) * scale

        total = len(self.trees) + len(self.shrubs) + len(self.grass) + len(self.buildings) + len(self.roads)
        if total == 0:
            self.error_msg = "No classification data available."
            return False

        QgsMessageLog.logMessage(
            f"OBJ Export: {len(self.trees)} trees, {len(self.shrubs)} shrubs, "
            f"{len(self.grass)} grass areas, {len(self.buildings)} buildings, "
            f"{len(self.roads)} road areas",
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
        for road in self.roads:
            for cell in road.get('cell_details', []):
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

        writer = ObjWriter()

        # --- Materials ---
        materials = {
            'mat_ground':       {'Kd': (0.25, 0.25, 0.25), 'd': 1.0},   # dark grey
            'mat_grass':        {'Kd': (0.20, 0.65, 0.15), 'd': 1.0},   # green
            'mat_shrub':        {'Kd': (0.08, 0.35, 0.05), 'd': 1.0},   # very dark green
            'mat_trunk':        {'Kd': (0.45, 0.28, 0.10), 'd': 1.0},   # brown
            'mat_crown':        {'Kd': (0.10, 0.55, 0.10), 'd': 1.0},   # green
            'mat_building':     {'Kd': (0.95, 0.64, 0.14), 'd': 1.0},   # orange
            'mat_road':         {'Kd': (0.10, 0.10, 0.10), 'd': 1.0},   # near-black
        }

        progress_done = 0
        progress_total = (1 if p.get('include_ground') else 0) + \
                         len(self.grass) + len(self.shrubs) + len(self.trees) + \
                         len(self.buildings) + len(self.roads)
        if progress_total == 0:
            progress_total = 1

        # =====================================================================
        #  1. GROUND PLANE
        # =====================================================================
        if p.get('include_ground', True):
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
        #  2. GRASS – flat squares slightly above ground
        # =====================================================================
        if p.get('include_grass', True) and self.grass:
            writer.begin_group('Grass', 'mat_grass')
            for grass_area in self.grass:
                if self.isCanceled():
                    return False

                for cell in grass_area.get('cell_details', []):
                    gx_min = (cell['x_min'] - origin_x) * scale
                    gx_max = (cell['x_max'] - origin_x) * scale
                    gz_min = (origin_z - cell['y_max']) * scale  # flipped
                    gz_max = (origin_z - cell['y_min']) * scale  # flipped

                    writer.add_quad(gx_min, gz_min, gx_max, gz_max, y=0.05 * scale)

                progress_done += 1
                if progress_done % 100 == 0:
                    self.setProgress(int(progress_done / progress_total * 100))

        if self.isCanceled():
            return False

        # =====================================================================
        #  2b. ROADS – flat black squares slightly above ground (below grass)
        # =====================================================================
        if p.get('include_roads', True) and self.roads:
            writer.begin_group('Roads', 'mat_road')
            for road_area in self.roads:
                if self.isCanceled():
                    return False

                for cell in road_area.get('cell_details', []):
                    rx_min = (cell['x_min'] - origin_x) * scale
                    rx_max = (cell['x_max'] - origin_x) * scale
                    rz_min = (origin_z - cell['y_max']) * scale  # flipped
                    rz_max = (origin_z - cell['y_min']) * scale  # flipped

                    writer.add_quad(rx_min, rz_min, rx_max, rz_max, y=0.03 * scale)

                progress_done += 1
                if progress_done % 100 == 0:
                    self.setProgress(int(progress_done / progress_total * 100))

        if self.isCanceled():
            return False

        # =====================================================================
        #  3. SHRUBS – small green cubes
        # =====================================================================
        if p.get('include_shrubs', True) and self.shrubs:
            writer.begin_group('Shrubs', 'mat_shrub')
            for shrub in self.shrubs:
                if self.isCanceled():
                    return False

                sx = (shrub.get('x', 0) - origin_x) * scale
                sz = (origin_z - shrub.get('y', 0)) * scale  # flipped
                # Limitar altura a máximo 2m para arbustos
                s_height = min(max(shrub.get('height', 0.3), 0.2), 2.0) * scale
                # Arbustos típicamente son 0.3-1.5m de diámetro
                s_diam = min(max(shrub.get('crown_diam', 0.5), 0.3), 1.5) * scale
                half = s_diam / 2.0

                writer.add_box(
                    sx - half, sz - half,
                    sx + half, sz + half,
                    y_base=0.0,
                    y_top=s_height
                )

                progress_done += 1
                if progress_done % 200 == 0:
                    self.setProgress(int(progress_done / progress_total * 100))

        if self.isCanceled():
            return False

        # =====================================================================
        #  4. TREES – rectangular trunk + cube crown
        # =====================================================================
        if p.get('include_trees', True) and self.trees:
            for tree in self.trees:
                if self.isCanceled():
                    return False

                tx = (tree.get('x', 0) - origin_x) * scale
                tz = (origin_z - tree.get('y', 0)) * scale  # flipped
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
                    y_base=0.0,
                    y_top=trunk_top
                )

                # Crown: cube at top
                crown_half = t_crown_diam / 2.0
                writer.begin_group(f'Tree_{tree_id}_crown', 'mat_crown')
                writer.add_box(
                    tx - crown_half, tz - crown_half,
                    tx + crown_half, tz + crown_half,
                    y_base=trunk_top,
                    y_top=t_height
                )

                progress_done += 1
                if progress_done % 100 == 0:
                    self.setProgress(int(progress_done / progress_total * 100))

        if self.isCanceled():
            return False

        # =====================================================================
        #  5. BUILDINGS – extruded boxes per cell (grid squares)
        # =====================================================================
        if p.get('include_buildings', True) and self.buildings:
            # Group by building ID
            for building in self.buildings:
                if self.isCanceled():
                    return False

                bid = building.get('x', 'x')  # use centroid as fallback ID
                writer.begin_group(f'Building_walls', 'mat_building')
                
                for cell in building.get('cell_details', []):
                    bx_min = (cell['x_min'] - origin_x) * scale
                    bx_max = (cell['x_max'] - origin_x) * scale
                    bz_min = (origin_z - cell['y_max']) * scale  # flipped
                    bz_max = (origin_z - cell['y_min']) * scale  # flipped
                    # Limitar altura edificios a máximo 50m realista
                    b_height = min(max(cell.get('height_max', 3), 1.0), 50.0) * scale

                    writer.add_box(
                        bx_min, bz_min,
                        bx_max, bz_max,
                        y_base=0.0,
                        y_top=b_height
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

    # Check if we have classification data
    has_data = hasattr(plugin_ref, 'last_classification_data') and plugin_ref.last_classification_data is not None
    
    # Show dialog - request input file if no data
    dlg = ObjExportDialog(plugin_ref.iface.mainWindow(), translator=plugin_ref.tr, require_input=not has_data, iface=plugin_ref.iface)

    if dlg.exec_() != ObjExportDialog.Accepted:
        return

    input_file = dlg.get_input_file()
    params = dlg.get_params()

    # If no existing data, run quick classify first
    if not has_data:
        if not input_file:
            from qgis.PyQt.QtWidgets import QMessageBox
            QMessageBox.warning(plugin_ref.iface.mainWindow(), plugin_ref.tr("Error"),
                                plugin_ref.tr("Please select a LAS/LAZ file."))
            return

        # Launch quick classify task in background
        task_desc = f"Classifying {os.path.basename(input_file)} for OBJ export"
        task = QuickClassifyTask(task_desc, input_file, plugin_ref, plugin_ref.tr)
        
        # Store callback to launch OBJ export after classification
        original_finished = task.finished
        def on_classify_done(result):
            original_finished(result)
            if result:
                # Now proceed with OBJ export
                _do_obj_export(plugin_ref, params)
        task.finished = on_classify_done
        
        plugin_ref.running_tasks.append(task)
        QgsApplication.taskManager().addTask(task)

        plugin_ref.iface.messageBar().pushMessage(
            plugin_ref.tr("Task Started"),
            plugin_ref.tr("Classifying LAS file for OBJ export..."),
            level=Qgis.Info, duration=-1
        )
    else:
        # Data already available, proceed directly
        _do_obj_export(plugin_ref, params)


def _do_obj_export(plugin_ref, params):
    """Launch the OBJ export task."""
    from .obj_export_dialog import ObjExportDialog

    data = plugin_ref.last_classification_data

    # Auto-generate output path in project directory
    base_path = os.path.splitext(data['filename'])[0]
    output_path = base_path + "_3d_scene.obj"

    task_desc = f"Exporting to OBJ → {os.path.basename(output_path)}"
    task = ObjExportTask(
        task_desc,
        data.get('trees', []),
        data.get('shrubs', []),
        data.get('grass', []),
        data.get('buildings', []),
        data.get('roads', []),
        params,
        output_path,
        plugin_ref,
        plugin_ref.tr
    )
    plugin_ref.running_tasks.append(task)
    QgsApplication.taskManager().addTask(task)

    plugin_ref.iface.messageBar().pushMessage(
        plugin_ref.tr("Task Started"),
        plugin_ref.tr("Exporting 3D scene to OBJ in the background..."),
        level=Qgis.Info, duration=-1
    )
