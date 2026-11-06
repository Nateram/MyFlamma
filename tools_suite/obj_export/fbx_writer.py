# -*- coding: utf-8 -*-
"""
FBX Export - Writes the scene accumulated by ObjWriter as a binary FBX 7.4 file.

No external SDK is required: the binary node layout is written directly
(the same layout produced by Blender's FBX exporter). Binary is used because
Blender cannot import ASCII FBX.

Scene structure:
  - One Model + Mesh per material (Ground, Grass, Shrubs, Trunks, Crowns, Buildings)
  - One Lambert material per mesh (DiffuseColor = MTL Kd)
  - Per-vertex normals and vertex colours
  - Y-up, right-handed, 1 unit = 1 metre (same coordinates as the OBJ)
"""

import struct
import time
import zlib

import numpy as np


_VERSION = 7400
_HEAD_MAGIC = b'Kaydara FBX Binary  \x00\x1a\x00'
_FOOT_ID = b'\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e'
_FOOT_MAGIC = b'\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b'
# The FBX SDK checks FileId against CreationTime, so a known-valid pair is used.
_FILE_ID = b'\x28\xb3\x2a\xeb\xb6\x24\xcc\xc2\xbf\xc8\xb0\x2a\xa9\x2b\xfc\xf1'
_CREATION_TIME = "1970-01-01 10:00:00:000"
_SENTINEL = b'\x00' * 13  # null record closing a nested list (FBX < 7500)

# Readable object names for each material group
_MESH_NAMES = {
    'mat_ground': 'Ground',
    'mat_grass': 'Grass',
    'mat_shrub': 'Shrubs',
    'mat_trunk': 'Tree_Trunks',
    'mat_crown': 'Tree_Crowns',
    'mat_building': 'Buildings',
}


# ---------------------------------------------------------------------------
#  Typed property helpers
# ---------------------------------------------------------------------------

def _C(v): return ('C', bool(v))
def _I(v): return ('I', int(v))
def _L(v): return ('L', int(v))
def _D(v): return ('D', float(v))
def _S(v): return ('S', v.encode('utf-8') if isinstance(v, str) else v)
def _R(v): return ('R', v)
def _Ad(v): return ('d', np.asarray(v, dtype='<f8'))
def _Ai(v): return ('i', np.asarray(v, dtype='<i4'))


def _encode_prop(prop):
    code, value = prop
    if code == 'C':
        return b'C' + struct.pack('<?', value)
    if code == 'I':
        return b'I' + struct.pack('<i', value)
    if code == 'L':
        return b'L' + struct.pack('<q', value)
    if code == 'D':
        return b'D' + struct.pack('<d', value)
    if code in ('S', 'R'):
        return code.encode() + struct.pack('<I', len(value)) + value
    # Arrays: count, encoding (1 = zlib), byte length, payload
    data = value.tobytes()
    encoding = 0
    if len(data) > 128:
        data = zlib.compress(data, 1)
        encoding = 1
    return code.encode() + struct.pack('<III', len(value), encoding, len(data)) + data


class _Node:
    def __init__(self, name, *props):
        self.name = name.encode('ascii')
        self.props = list(props)
        self.children = []

    def add(self, name, *props):
        node = _Node(name, *props)
        self.children.append(node)
        return node

    def p70(self, name, type_name, label, flags, *values):
        """Add a 'P' entry to a Properties70 node."""
        return self.add('P', _S(name), _S(type_name), _S(label), _S(flags), *values)


def _write_node(f, node, is_last):
    props = b''.join(_encode_prop(p) for p in node.props)
    start = f.tell()
    f.write(struct.pack('<III', 0, len(node.props), len(props)))
    f.write(struct.pack('<B', len(node.name)))
    f.write(node.name)
    f.write(props)
    if node.children:
        last = len(node.children) - 1
        for i, child in enumerate(node.children):
            _write_node(f, child, i == last)
        f.write(_SENTINEL)
    elif not node.props and not is_last:
        f.write(_SENTINEL)
    end = f.tell()
    f.seek(start)
    f.write(struct.pack('<I', end))
    f.seek(end)


def _fbx_name(name, cls):
    # Binary FBX stores "Class::Name" as "Name\x00\x01Class"
    return _S(name.encode('utf-8') + b'\x00\x01' + cls.encode('ascii'))


# ---------------------------------------------------------------------------
#  Mesh extraction from ObjWriter
# ---------------------------------------------------------------------------

def _build_mesh(writer, faces):
    """Convert ObjWriter faces into FBX arrays.

    A local vertex is created per (vertex, normal) pair so that hard edges and
    double-sided faces keep their own normals."""
    local = {}
    verts, normals, colors, pvi, color_idx = [], [], [], [], []
    for face in faces:
        last = len(face) - 1
        for k, (vi, ni) in enumerate(face):
            key = (vi, ni)
            idx = local.get(key)
            if idx is None:
                idx = len(local)
                local[key] = idx
                verts.extend(writer.vertices[vi - 1])
                normals.extend(writer.normals[ni - 1])
                r, g, b = writer.colors[vi - 1]
                colors.extend((r, g, b, 1.0))
            # The last index of each polygon is stored as -(index + 1)
            pvi.append(~idx if k == last else idx)
            color_idx.append(idx)
    return verts, normals, colors, pvi, color_idx


# ---------------------------------------------------------------------------
#  Public entry point
# ---------------------------------------------------------------------------

def write_fbx(fbx_path, writer, materials, creator="MyFlamma QGIS plugin"):
    """Write the geometry held by an ObjWriter to a binary FBX file."""
    # Merge groups by material (one mesh per element type)
    faces_by_mat = {}
    for grp in writer.groups:
        if grp['faces']:
            faces_by_mat.setdefault(grp['material'], []).extend(grp['faces'])

    next_id = [1000000]

    def new_id():
        next_id[0] += 1
        return next_id[0]

    root = _Node('')

    # --- Header ---
    now = time.localtime()
    header = root.add('FBXHeaderExtension')
    header.add('FBXHeaderVersion', _I(1003))
    header.add('FBXVersion', _I(_VERSION))
    header.add('EncryptionType', _I(0))
    stamp = header.add('CreationTimeStamp')
    stamp.add('Version', _I(1000))
    for key, val in (('Year', now.tm_year), ('Month', now.tm_mon), ('Day', now.tm_mday),
                     ('Hour', now.tm_hour), ('Minute', now.tm_min), ('Second', now.tm_sec),
                     ('Millisecond', 0)):
        stamp.add(key, _I(val))
    header.add('Creator', _S(creator))
    root.add('FileId', _R(_FILE_ID))
    root.add('CreationTime', _S(_CREATION_TIME))
    root.add('Creator', _S(creator))

    # --- Global settings: Y-up, right-handed, centimetre-based unit x100 = metres ---
    gs = root.add('GlobalSettings')
    gs.add('Version', _I(1000))
    p = gs.add('Properties70')
    for name, val in (('UpAxis', 1), ('UpAxisSign', 1), ('FrontAxis', 2), ('FrontAxisSign', 1),
                      ('CoordAxis', 0), ('CoordAxisSign', 1),
                      ('OriginalUpAxis', 1), ('OriginalUpAxisSign', 1)):
        p.p70(name, 'int', 'Integer', '', _I(val))
    p.p70('UnitScaleFactor', 'double', 'Number', '', _D(100.0))
    p.p70('OriginalUnitScaleFactor', 'double', 'Number', '', _D(100.0))

    # --- Documents / References ---
    docs = root.add('Documents')
    docs.add('Count', _I(1))
    doc = docs.add('Document', _L(new_id()), _S(''), _S('Scene'))
    p = doc.add('Properties70')
    p.p70('SourceObject', 'object', '', '')
    p.p70('ActiveAnimStackName', 'KString', '', '', _S(''))
    doc.add('RootNode', _L(0))
    root.add('References')

    # --- Definitions ---
    n = len(faces_by_mat)
    defs = root.add('Definitions')
    defs.add('Version', _I(100))
    defs.add('Count', _I(1 + 3 * n))
    for obj_type, count in (('GlobalSettings', 1), ('Model', n), ('Geometry', n), ('Material', n)):
        ot = defs.add('ObjectType', _S(obj_type))
        ot.add('Count', _I(count))

    # --- Objects + Connections ---
    objects = root.add('Objects')
    connections = root.add('Connections')

    for mat_name, faces in faces_by_mat.items():
        mesh_name = _MESH_NAMES.get(mat_name, mat_name)
        verts, normals, colors, pvi, color_idx = _build_mesh(writer, faces)

        geom_id, model_id, mat_id = new_id(), new_id(), new_id()

        geom = objects.add('Geometry', _L(geom_id), _fbx_name(mesh_name, 'Geometry'), _S('Mesh'))
        geom.add('Properties70')
        geom.add('GeometryVersion', _I(124))
        geom.add('Vertices', _Ad(verts))
        geom.add('PolygonVertexIndex', _Ai(pvi))

        le = geom.add('LayerElementNormal', _I(0))
        le.add('Version', _I(101))
        le.add('Name', _S(''))
        le.add('MappingInformationType', _S('ByVertice'))
        le.add('ReferenceInformationType', _S('Direct'))
        le.add('Normals', _Ad(normals))

        le = geom.add('LayerElementColor', _I(0))
        le.add('Version', _I(101))
        le.add('Name', _S('Col'))
        le.add('MappingInformationType', _S('ByPolygonVertex'))
        le.add('ReferenceInformationType', _S('IndexToDirect'))
        le.add('Colors', _Ad(colors))
        le.add('ColorIndex', _Ai(color_idx))

        le = geom.add('LayerElementMaterial', _I(0))
        le.add('Version', _I(101))
        le.add('Name', _S(''))
        le.add('MappingInformationType', _S('AllSame'))
        le.add('ReferenceInformationType', _S('IndexToDirect'))
        le.add('Materials', _Ai([0]))

        layer = geom.add('Layer', _I(0))
        layer.add('Version', _I(100))
        for le_type in ('LayerElementNormal', 'LayerElementColor', 'LayerElementMaterial'):
            entry = layer.add('LayerElement')
            entry.add('Type', _S(le_type))
            entry.add('TypedIndex', _I(0))

        model = objects.add('Model', _L(model_id), _fbx_name(mesh_name, 'Model'), _S('Mesh'))
        model.add('Version', _I(232))
        p = model.add('Properties70')
        p.p70('DefaultAttributeIndex', 'int', 'Integer', '', _I(0))
        model.add('Shading', _C(True))
        model.add('Culling', _S('CullingOff'))

        kd = materials.get(mat_name, {}).get('Kd', (0.5, 0.5, 0.5))
        alpha = materials.get(mat_name, {}).get('d', 1.0)
        mat = objects.add('Material', _L(mat_id), _fbx_name(mat_name, 'Material'), _S(''))
        mat.add('Version', _I(102))
        mat.add('ShadingModel', _S('lambert'))
        mat.add('MultiLayer', _I(0))
        p = mat.add('Properties70')
        p.p70('DiffuseColor', 'Color', '', 'A', _D(kd[0]), _D(kd[1]), _D(kd[2]))
        p.p70('DiffuseFactor', 'Number', '', 'A', _D(1.0))
        p.p70('AmbientColor', 'Color', '', 'A', _D(kd[0] * 0.2), _D(kd[1] * 0.2), _D(kd[2] * 0.2))
        p.p70('Opacity', 'double', 'Number', '', _D(alpha))

        connections.add('C', _S('OO'), _L(model_id), _L(0))
        connections.add('C', _S('OO'), _L(geom_id), _L(model_id))
        connections.add('C', _S('OO'), _L(mat_id), _L(model_id))

    takes = root.add('Takes')
    takes.add('Current', _S(''))

    # --- Write file ---
    with open(fbx_path, 'wb') as f:
        f.write(_HEAD_MAGIC)
        f.write(struct.pack('<I', _VERSION))
        last = len(root.children) - 1
        for i, child in enumerate(root.children):
            _write_node(f, child, i == last)
        f.write(_SENTINEL)
        # Footer
        f.write(_FOOT_ID)
        f.write(b'\x00' * 4)
        pad = ((f.tell() + 15) & ~15) - f.tell()
        f.write(b'\x00' * (pad or 16))
        f.write(struct.pack('<I', _VERSION))
        f.write(b'\x00' * 120)
        f.write(_FOOT_MAGIC)
