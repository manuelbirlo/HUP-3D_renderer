from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import pickle
import trimesh
import os
from matplotlib.patches import Circle
from PIL import Image, ImageDraw

def visualize_corners_2d(
    ax, corners, joint_idxs=False, links=None, alpha=1, linewidth=2
):
    visualize_joints_2d(
        ax,
        corners,
        alpha=alpha,
        joint_idxs=joint_idxs,
        linewidth=linewidth,
        links=[
            [0, 1, 3, 2],
            [4, 5, 7, 6],
            [1, 5],
            [3, 7],
            [4, 0],
            [0, 2, 6, 4],
        ],
    )


def visualize_joints_2d(
    ax,
    joints,
    joint_idxs=True,
    links=None,
    alpha=1,
    scatter=True,
    linewidth=2,
    color=None,
    joint_labels=None,
    axis_equal=False, #True,
):
    if links is None:
        links = [
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    if scatter:
        ax.scatter(x, y, 1, "r")

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            if joint_labels is None:
                joint_label = str(row_idx)
            else:
                joint_label = str(joint_labels[row_idx])
            ax.annotate(joint_label, (row[0], row[1]))
    _draw2djoints(
        ax, joints, links, alpha=alpha, linewidth=linewidth, color=color
    )
    if axis_equal:
        ax.axis("equal")


def _draw2djoints(ax, annots, links, alpha=1, linewidth=1, color=None):
    colors = ["r", "m", "b", "c", "g", "y", "b"]

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            if color is not None:
                link_color = color[finger_idx]
            else:
                link_color = colors[finger_idx]
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=link_color,
                alpha=alpha,
                linewidth=linewidth,
            )


def _draw2dseg(ax, annot, idx1, idx2, c="r", alpha=1, linewidth=1):
    ax.plot(
        [annot[idx1, 0], annot[idx2, 0]],
        [annot[idx1, 1], annot[idx2, 1]],
        c=c,
        alpha=alpha,
        linewidth=linewidth,
    )

def get_camintr(cam_calib):
        return np.array(cam_calib).astype(np.float32) # shape: (3, 3)

def transform(verts, trans, convert_to_homogeneous=False):
    assert len(verts.shape) == 2, "Expected 2 dimensions for verts, got: {}.".format(len(verts.shape))
    assert len(trans.shape) == 2, "Expected 2 dimensions for trans, got: {}.".format(len(trans.shape))
    if convert_to_homogeneous:
        hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
    else:
        hom_verts = verts

    assert trans.shape[1] == hom_verts.shape[1], \
        "Incompatible shapes: verts.shape: {}, trans.shape: {}".format(verts.shape, trans.shape)

    trans_verts = np.dot(trans, hom_verts.transpose()).transpose()
    return trans_verts

def get_obj_mesh(obj_path): 
     mesh = trimesh.load(obj_path)
     return mesh

def get_obj_verts(obj_path):
        verts = get_obj_mesh(obj_path).vertices
        return np.array(verts).astype(np.float32)

def get_obj_verts_trans(obj_path, cam_extr, affine_transform):
        # Get object 3d vertices (n,3) in the camera coordinate frame
        verts = get_obj_verts(obj_path) # shape: (n, 3)
        trans_verts = transform(verts, get_obj_pose(cam_extr, affine_transform), convert_to_homogeneous=True)
        return np.array(trans_verts).astype(np.float32)

def get_obj_pose(cam_extr, affine_transform):
        # Get object pose (3,4) matrix in world coordinate frame
        transform = cam_extr @ affine_transform
        return np.array(transform).astype(np.float32)

def get_objcorners2d(cam_calib, obj_path, cam_extr, affine_transform):
        corners_3d = get_obj_corners3d(obj_path, cam_extr, affine_transform)
        intr = get_camintr(cam_calib)
        corners_2d_hom = transform(corners_3d, intr)
        corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:]
        return np.array(corners_2d).astype(np.float32)

# Probably not needed
def transform_coords(pts, affine_trans, invert=False):
    """
    Args:
        pts(np.ndarray): (point_nb, 2)
    """
    if invert:
        affine_trans = np.linalg.inv(affine_trans)
    hom2d = np.concatenate([pts, np.ones([np.array(pts).shape[0], 1])], 1)
    transformed_rows = affine_trans.dot(hom2d.transpose()).transpose()[:, :2]
    return transformed_rows

def get_obj_corners3d(obj_path, cam_extr, affine_transform):
        model = get_obj_verts_trans(obj_path, cam_extr, affine_transform)
        min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
        min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
        min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
        corners_3d = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ])
        return np.array(corners_3d).astype(np.float32)

def get_objverts2d(cam_calib, obj_path, cam_extr, affine_transform):
        objpoints3d = get_obj_verts_trans(obj_path, cam_extr, affine_transform).transpose() # shape: (3, n)
        hom_2d = np.dot(get_camintr(cam_calib), objpoints3d).transpose() # shape: (n, 3)
        verts2d = (hom_2d / hom_2d[:, 2:])[:, :2] # shape: (n, 2)
        return np.array(verts2d).astype(np.float32)

img_name = "GE_Voluson_2D_aligned_downscaled.ply_grasp001_0008_cam_view_x0_y3.141592653589793_z0"

img = Image.open("/graspit_ros_ws/grasp_renderer/datasets/synthetic/train/rgb/{}.jpg".format(img_name))
img = np.array(img)
meta_path = "/graspit_ros_ws/grasp_renderer/datasets/synthetic/train/meta/{}.pkl".format(img_name)
with open(meta_path, 'rb') as meta_f:
    metainfo = pickle.load(meta_f)
# visualize_corners_2d(
#     ax, corners, joint_idxs=False, links=None, alpha=1, linewidth=2
# )
gt_handjoints2d = metainfo["coords_2d"]
affine_transform = metainfo["affine_transform"]
trans = metainfo['trans']
pca_pose = metainfo['pca_pose']
cam_calib = metainfo['cam_calib']
obj_path = metainfo['obj_path']
cam_extr = metainfo['cam_extr']

object_verts_2d = get_objverts2d(cam_calib, obj_path, cam_extr, affine_transform)
object_corners_2d = get_objcorners2d(cam_calib, obj_path, cam_extr, affine_transform)
#object_corners_3d = get_obj_corners3d(obj_path, cam_extr, affine_transform)

radius = 2
                
# Blue color in BGR 
color = (0, 0, 255) 
   
# Line thickness of 2 px 
thickness = 1


fig, ax = plt.subplots(1,1)
if gt_handjoints2d is not None:
    visualize_joints_2d(ax, gt_handjoints2d, alpha=0.5, joint_idxs=False)



object_verts_2d = [object_verts_2d[index] for index in range(1, len(object_verts_2d), 100)]
x_values=[i[0] for i in object_verts_2d]
y_values=[i[1] for i in object_verts_2d]

ax.scatter(x_values, y_values, 1, "r")

x_values= object_corners_2d[:, 0]
y_values=object_corners_2d[:, 1]

#x_val = [gt_objcorners2d[0][0]]
ax.scatter(x_values, y_values, 10, "b")
#ax.plot([x_values[0], x_values[1]], [y_values[0], y_values[1]], "r")
#ax.plot([x_values[1], x_values[3]], [y_values[1], y_values[3]], "m")
#ax.plot([x_values[3], x_values[2]], [y_values[3], y_values[2]], "b")

#ax.plot([x_values[4], x_values[5]], [y_values[4], y_values[5]], "c")
#ax.plot([x_values[5], x_values[7]], [y_values[5], y_values[7]], "g")
#ax.plot([x_values[7], x_values[6]], [y_values[7], y_values[6]], "y")

#ax.plot([x_values[1], x_values[5]], [y_values[1], y_values[5]], "b")

#ax.plot([x_values[3], x_values[7]], [y_values[3], y_values[7]], "m")

#ax.plot([x_values[4], x_values[0]], [y_values[4], y_values[0]], "g")

#ax.plot([x_values[0], x_values[2]], [y_values[0], y_values[2]], "y")
#ax.plot([x_values[2], x_values[6]], [y_values[2], y_values[6]], "b")
#ax.plot([x_values[6], x_values[4]], [y_values[6], y_values[4]], "r")
#"r", "m", "b", "c", "g", "y", "b"
#for coords in sparse_coords[0:2000]:
     #current_coords = (int(coords[0]), int(coords[1]))
 #    circ = Circle((coords[0], coords[1]),1)
 #    ax.add_patch(circ)
     #ax.plot(coords[0], coords[1], line) 

# Column 1
ax.imshow(img)
ax.axis("off")
#plt.show()
# Visualize 2D object bounding box
visualize_joints_2d(
        ax,
        object_corners_2d,
        alpha=0.5,
        joint_idxs=False,
        links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]]
)
ax.imshow(img)
plt.show()

a = 1
