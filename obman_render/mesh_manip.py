from copy import deepcopy
import sys
import os
import random
import numpy as np
from scipy.spatial.transform import Rotation

root = '.'
sys.path.insert(0, root)
mano_path = os.environ.get('MANO_LOCATION', None)
if mano_path is None:
    raise ValueError('Environment variable MANO_LOCATION not defined'
                     'Please follow the README.md instructions')
sys.path.insert(0, os.path.join(mano_path, 'webuser'))

from lbs import global_rigid_transformation


def alter_mesh(obj, verts):
    import bmesh
    import bpy
    from mathutils import Vector
    bpy.context.view_layer.objects.active = obj
    mesh = bpy.context.object.data

    bm = bmesh.new()

    # convert the current mesh to a bmesh (must be in edit mode)
    bpy.ops.object.mode_set(mode='EDIT')
    bm.from_mesh(mesh)
    bpy.ops.object.mode_set(mode='OBJECT')  # return to object mode

    for v, bv in zip(verts, bm.verts):
        bv.co = Vector(v)

    # make the bmesh the object's mesh
    bm.to_mesh(mesh)
    bm.select_flush(True)
    bm.free()  # always do this when finished


def load_body_data(smpl_data, gender='female', idx=0, n_sh_bshapes=10):
    """
    Loads MoSHed pose data from CMU Mocap (only the given idx is loaded), and loads all CAESAR shape data.
    Args:
        smpl_data: Files with *trans, *shape, *pose parameters
        gender: female | male. CAESAR data has 2K male, 2K female shapes
        idx: index of the mocap sequence
        n_sh_bshapes: number of shape blendshapes (number of PCA components)
    """
    # create a dictionary with key the sequence name and values the pose and trans
    cmu_keys = []
    for seq in smpl_data.files:
        if seq.startswith('pose_'):
            cmu_keys.append(seq.replace('pose_', ''))

    name = sorted(cmu_keys)[idx % len(cmu_keys)]

    cmu_parms = {}
    for seq in smpl_data.files:
        if seq == ('pose_' + name):
            cmu_parms[seq.replace('pose_', '')] = {
                'poses': smpl_data[seq],
                'trans': smpl_data[seq.replace('pose_', 'trans_')]
            }

    # load all SMPL shapes
    fshapes = smpl_data['%sshapes' % gender][:, :n_sh_bshapes]

    return (cmu_parms, fshapes, name)


def load_smpl(template='assets/models/basicModel_{}_lbs_10_207_0_v1.0.2.fbx',
              gender='f'):
    """
    Loads smpl model, deleted armature and renames mesh to 'Body'
    """
    import bpy
    filepath = template.format(gender)
    bpy.ops.import_scene.fbx(
        filepath=filepath, axis_forward='Y', axis_up='Z', global_scale=100)
    obname = '{}_avg'.format(gender)
    ob = bpy.data.objects[obname]
    ob.parent = None

    # Delete armature
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Armature'].select_set(True)
    bpy.ops.object.delete(use_global=False)

    # Rename mesh
    #bpy.data.meshes['Untitled'].name = 'Body'
    return ob


def random_global_rotation():
    """
    Creates global random rotation in axis-angle rotation format.
    """
    # 1. We will pick random axis: random azimuth and random elevation in spherical coordinates.
    # 2. We will pick random angle.
    # Random azimuth
    randazimuth = np.arccos(2 * np.random.rand(1) - 1)
    # Random elevation
    randelevation = 2 * np.pi * np.random.rand(1)
    # Random axis in cartesian coordinate (this already has norm 1)
    randaxis = np.asarray([
        np.cos(randelevation) * np.cos(randazimuth),
        np.cos(randelevation) * np.sin(randazimuth),
        np.sin(randelevation)
    ], dtype=np.float)
    # Random angle
    randangle = 2.0 * np.pi * np.random.rand(1)

    # Construct axis-angle vector
    randaxisangle = randangle * randaxis

    return np.squeeze(randaxisangle)


def egocentric_viewpoint(global_joint_positions, head_idx=15, hand_idx=40, pelvis_idx=0, handNoise=np.array([[0.0], [0.0], [0.0]])):
    '''
    Rotate head->hand vector into z axis (same direction as camera)
    :param global_joint_positions: array of global joint positions for the body skeleton
    :param head_idx: index of the head joint in the global_joint_positions array
    :param hand_idx: index of the right hand joint in the global_joint_positions array
    :param pelvis_idx: index of the pelvis joint in the global_joint_positions array
    :param head_noise_factor:
    :return: global rotation in axis-angle representation
    '''

    randpose = np.reshape(global_joint_positions, [-1, 3])

    hand = np.array(global_joint_positions[hand_idx, :], dtype=np.float).reshape([-1,1])
    head = np.array(global_joint_positions[head_idx, :], dtype=np.float).reshape([-1,1])
    pelvis = np.array(global_joint_positions[pelvis_idx, :], dtype=np.float).reshape([-1, 1])
    headNoise = 1.0 * (np.random.random(size=(3, 1)) - 0.5)
    headNoise[2] = 0.0
    head = head + headNoise
    #print("Head noise: {}".format(headNoise))
    hand = hand + handNoise
    headToHand = hand - head

    # Rotation angle around X axis (rotate vector into XZ plane)
    psi = np.arctan(headToHand[1] / headToHand[2])
    cosPsi = np.cos(psi)
    sinPsi = np.sin(psi)
    rotX = Rotation.from_matrix([[1.0, 0.0, 0.0], [0.0, cosPsi, -sinPsi], [0.0, sinPsi, cosPsi]]).as_matrix()
    rotHeadToHand = np.dot(rotX, headToHand)

    # Rotation angle around Y axis (rotate vector into YZ plane => orient is parallel to Z axis)
    phi = np.arctan(- rotHeadToHand[0] / rotHeadToHand[2])
    cosPhi = np.cos(phi)
    sinPhi = np.sin(phi)
    rotY = Rotation.from_matrix([[cosPhi, 0.0, sinPhi], [0.0, 1.0, 0.0], [-sinPhi, 0.0, cosPhi]]).as_matrix()
    rotHeadToHand = np.dot(rotY, rotHeadToHand)
    rot = np.matmul(rotY, rotX)

    # Ensure orientation points along negative z-axis (hand is in front of the camera)
    if (rotHeadToHand[2] > 0.0):
        rotX180 = Rotation.from_matrix([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]).as_matrix()
        rot = np.matmul(rotX180, rot)
        rotHeadToHand = np.dot(rotX180, rotHeadToHand)

    # Rotation around Z axis (spine should be parallel to negative Y axis, so rotate it into YZ plane)
    headToPelvis = pelvis - head
    rotHeadToPelvis = np.dot(rot, headToPelvis)
    beta = np.arctan(rotHeadToPelvis[0] / rotHeadToPelvis[1])
    cosBeta = np.cos(beta)
    sinBeta = np.sin(beta)
    rotZ = Rotation.from_matrix([[cosBeta, -sinBeta, 0.0], [sinBeta, cosBeta, 0.0], [0.0, 0.0, 1.0]]).as_matrix()
    rot = np.matmul(rotZ, rot)
    rotHeadToPelvis = np.dot(rot, headToPelvis)

    if (rotHeadToPelvis[1] > 0.0):
        rotZ180 = Rotation.from_matrix([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]).as_matrix()
        rot = np.matmul(rotZ180, rot)

    #print("head->hand: {}".format(rot.dot(hand - head)))

    axisangle = Rotation.from_matrix(rot).as_rotvec()
    #print("Norm: {}".format(np.linalg.norm(axisangle)))
    return axisangle


def randomized_verts(model,
                     smpl_data,
                     ncomps=12,
                     pose_var=2,
                     hand_pose=None,
                     hand_pose_offset=3,
                     z_min=0.5,
                     z_max=0.8,
                     head_idx=15,
                     shape_val=2.0,
                     random_shape=False,
                     random_pose=False,
                     body_rot=True,
                     side='right',
                     split='train'):
    """
    Args:
        model: SMPL+H chumpy model
        smpl_data: 72-dim SMPL pose parameters from CMU and 10-dim shape parameteres from CAESAR
        center_idx: hand root joint on which to center, 25 for left hand, 40 for right
        z_min: min distance to camera in world coordinates
        z_max: max distance to camera in world coordinates
        ncomps: number of principal components used for both hands
        hand_pose: pca coeffs of hand pose
        hand_pose_offset: 3 is hand_pose contains global rotation
            0 if only pca coeffs are provided
    """

    if side == 'left':
        center_idx = 25
    else:
        center_idx = 40
    # Load smpl
    if split == 'test':
        cmu_idx = random.choice(list(range(4000, 4700)))
    elif split == 'val':
        cmu_idx = random.choice(list(range(4700, 5338)))
    else:
        cmu_idx = random.choice(list(range(0, 4000)))

    cmu_parms, fshapes, name = load_body_data(smpl_data, idx=cmu_idx)
    pose_data = cmu_parms[name]['poses']
    nframes = pose_data.shape[0]
    randframe = np.random.randint(nframes)

    # Init with zero trans
    model.trans[:] = np.zeros(model.trans.size)

    # Set random shape param
    if random_shape:
        #randshape = random.choice(fshapes)
        randshape = np.random.uniform(
            low=-shape_val, high=shape_val, size=model.betas.shape)
        model.betas[:] = randshape
    else:
        randshape = np.zeros(model.betas.shape)

    # Random body pose (except hand)
    randpose = np.zeros(model.pose.size)
    if random_pose:
        body_idx = 72
        randpose[:body_idx] = pose_data[randframe]

    # Overwrite global rotation with uniform random rotation
    randpose[0:3] = 0
    _, global_joint_positions = global_rigid_transformation(randpose, model.J, model.kintree_table)
    global_joint_positions = np.vstack([g[:3, 3] for g in global_joint_positions])

    noiseFactor = 0.02
    handNoise = noiseFactor * 2.0 * (np.random.random(size=(3, 1)) - 0.5)
    if body_rot:
        randpose[0:3] = egocentric_viewpoint(global_joint_positions, handNoise=handNoise)
    else:
        randpose[0:3] = [-np.pi/2, 0, 0]

    hand_comps = int(ncomps / 2)
    hand_idx = 66
    if hand_pose is not None:
        if side == 'left':
            randpose[hand_idx:hand_idx + hand_comps:] = hand_pose[
                hand_pose_offset:]
            left_rand = hand_pose[hand_pose_offset:]
        elif side == 'right':
            randpose[hand_idx + hand_comps:] = hand_pose[hand_pose_offset:]
            right_rand = hand_pose[hand_pose_offset:]
    else:
        # Alter right hand
        right_rand = np.random.uniform(
            low=-pose_var, high=pose_var, size=(hand_comps, ))
        randpose[hand_idx:hand_idx + hand_comps:] = right_rand

        # Alter left hand
        left_rand = np.random.uniform(
            low=-pose_var, high=pose_var, size=(hand_comps, ))
        randpose[hand_idx + hand_comps:] = left_rand

    model.pose[:] = randpose

    # Center on the hand
    hand = np.array(randpose[hand_idx : hand_idx+3], dtype=np.float).reshape([-1, 1])
    head = np.array(randpose[head_idx : head_idx+3], dtype=np.float).reshape([-1, 1])
    hand_head_dist = np.linalg.norm(hand - head) - 0.1
    #print("hand_head_dist: {}".format(hand_head_dist))

    if hand_head_dist > z_min:
        z_max = min(z_max, hand_head_dist)
    rand_z = random.uniform(z_min, z_max)
    trans = np.array(
        [model.J_transformed[center_idx, :].r[i] for i in range(3)])
    trans[0] = trans[0] + handNoise[0]
    trans[1] = trans[1] + handNoise[1]
    # Offset in z direction
    trans[2] = trans[2] + rand_z
    model.trans[:] = -trans

    new_verts = model.r
    if side == 'right':
        hand_pose = right_rand
    else:
        hand_pose = left_rand
    meta_info = {
        'z': rand_z,
        'trans': (-trans).astype(np.float32),
        'pose': randpose.astype(np.float32),
        'shape': randshape.astype(np.float32),
        'mano_pose': hand_pose.astype(np.float32)
    }
    return new_verts, model, meta_info


def load_obj(filename_obj, normalization=True, texture_size=4):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices = []
    for line in open(filename_obj).readlines():
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = np.vstack(vertices).astype('float32')

    # load faces
    faces = []
    for line in open(filename_obj).readlines():
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype('int32') - 1

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[None, :]
        vertices /= np.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[None, :] / 2

    return vertices, faces
