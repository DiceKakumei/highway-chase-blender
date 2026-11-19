"""手の調査: rider_rigged.glb の右手(と銃)を、単純な照明で数方向から描く。 blender --background --python tools/hand_view.py -- OUT"""
import bpy, os, sys, math
import numpy as np
from mathutils import Vector, Matrix
OUT = sys.argv[sys.argv.index('--') + 1]; os.makedirs(OUT, exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=os.path.join(ROOT, 'assets/bike_B2/rider_rigged.glb'))
ao = next(o for o in bpy.data.objects if o.type == 'ARMATURE'); body = next(o for o in bpy.data.objects if o.type == 'MESH')
for m in body.data.materials:
    for n in m.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            for l in list(n.inputs['Alpha'].links): m.node_tree.links.remove(l)
sc = bpy.context.scene; sc.render.engine = 'CYCLES'; sc.cycles.samples = 16; sc.cycles.device = 'CPU'
sc.render.resolution_x = 800; sc.render.resolution_y = 800
w = bpy.data.worlds.new('w'); w.use_nodes = True; w.node_tree.nodes['Background'].inputs[1].default_value = 1.2; sc.world = w
for loc, e in (((-2, -2, 3), 500), ((2, -1, 2), 300)):
    ld = bpy.data.lights.new('l', 'AREA'); ld.energy = e; ld.size = 2; lo = bpy.data.objects.new('l', ld); lo.location = loc; sc.collection.objects.link(lo)
    lo.rotation_euler = (Vector((0, 0, 0.9)) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
bpy.context.view_layer.update()
# 右手の頂点(重み>0.5)の中心
hi = body.vertex_groups['RightHand'].index
vs = [v for v in body.data.vertices if any(g.group == hi and g.weight > 0.5 for g in v.groups)]
c = sum((body.matrix_world @ v.co for v in vs), Vector()) / len(vs)
print('hand verts', len(vs), 'center', c)
pb = ao.pose.bones['RightHand']; print('RightHand head', ao.matrix_world @ pb.head, 'tail', ao.matrix_world @ pb.tail)
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.lens = 60
for name, d in (('front', (0, -1, 0)), ('back', (0, 1, 0)), ('left', (-1, 0, 0)), ('right', (1, 0, 0)), ('top', (0, 0, 1)), ('under', (0, 0, -1))):
    d = Vector(d); cam.location = c + d * 0.7
    cam.rotation_euler = (c - cam.location).to_track_quat('-Z', 'Y' if abs(d.z) < 0.5 else 'X').to_euler()
    sc.render.filepath = os.path.join(OUT, 'hand_%s.png' % name); bpy.ops.render.render(write_still=True)
