

import json
import io
import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R

def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [ndarray_to_list(item) for item in obj]
    else:
        return obj

def euler_to_quaternion(euler_angles, sequence='xyz'):
    rotation = R.from_euler(sequence, euler_angles, degrees=False)
    return rotation.as_quat()

def construct_pose(transl, orientation, orientation_type='euler', sequence='xyz'):
    if orientation_type == 'euler':
        quaternion = euler_to_quaternion(orientation, sequence)
    elif orientation_type == 'quaternion':
        quaternion = orientation
    elif orientation_type == 'rotmat':
        rotation = R.from_matrix(orientation)
        quaternion = rotation.as_quat()
    else:
        raise ValueError(f"Unsupported orientation type: {orientation_type}")
    pose = np.concatenate([transl, quaternion])
    return ndarray_to_list(pose)

def convert_mat_to_json(mat_file):
    
    # Initialize list to hold grasp data
    json_data_list = []

    # Get the number of grasps (assuming length of 'zgen' corresponds to number of grasps)
    num_grasps = len(mat_file['zgen'])

    # Iterate over each grasp
    for idx in range(num_grasps):
        json_data = {
            "body": mat_file.get('body', "/root/grasp_renderer/assets/voluson_painted_downscaled.ply"),
            "pose": construct_pose(ndarray_to_list(mat_file['transl'][idx]), ndarray_to_list(mat_file['rotmat'][idx]), orientation_type='rotmat'),
            "dofs": ndarray_to_list(mat_file['zgen'][idx]),
            "contacts": ndarray_to_list(mat_file.get('contacts', [])),
            "epsilon": mat_file.get('epsilon', 0.0),
            "volume": mat_file.get('volume', 0.0),
            "link_in_contact": ndarray_to_list(mat_file.get('link_in_contact', [])),
            "quality": mat_file.get('quality', 0.0),
            "mano_trans": [ndarray_to_list(mat_file['transl'][idx])],  # Change here to ensure a list of lists
            "mano_pose": ndarray_to_list(np.concatenate([ndarray_to_list(mat_file['global_orient'][idx]), mat_file['hand_pose'][idx]])),
            "rotmat": ndarray_to_list(mat_file['rotmat'][idx]),
            "selected_grasps":[1, 2, 5, 7] # This list needs to be added manually
        }

        json_data_list.append(json_data)

    return json_data_list
