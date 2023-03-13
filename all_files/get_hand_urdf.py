import trimesh
from graspsampler.common import Scene, PandaGripper
from graspsampler.utils import gripper_bd

scene = Scene()

gripper = PandaGripper()

# obbs = gripper.get_obbs()

# for bb in obbs:
    # bb.visual.face_colors = [255,255,255, 75]
    # scene.add_geometry(bb)

base = trimesh.load('assets/urdf_files/meshes/collision/hand.obj').bounding_box
finger = trimesh.load('assets/urdf_files/meshes/collision/finger.obj').bounding_box

# scene.add_geometry(base)
# scene.add_geometry(finger)
trimesh.exchange.export.export_mesh(finger, 'finger2.obj', 'obj')

# scene.add_gripper(gripper, face_colors=[125,125,125,255])
# scene.show()