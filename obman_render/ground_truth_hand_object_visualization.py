from PIL import Image
import numpy as np
#from matplotlib import pyplot as plt
import trimesh

class GroundTruthVisualization:

    def __init__(self, blender_scene, folder_rgb, folder_rgb_hand_with_skeleton):
        self.blender_scene = blender_scene,
        self.folder_rgb = folder_rgb,
        self.folder_rgb_hand_with_skeleton = folder_rgb_hand_with_skeleton

    
    def visualize_joints_2d(
        self,
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
        self._draw2djoints(
              ax, joints, links, alpha=alpha, linewidth=linewidth, color=color
        )
        if axis_equal:
            ax.axis("equal")
            
    def visualize_corners_2d(
        self, ax, corners, joint_idxs=False, links=None, alpha=1, linewidth=2
    ):
        self.visualize_joints_2d(
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

    def _draw2djoints(self, ax, annots, links, alpha=1, linewidth=1, color=None):
        colors = ["r", "m", "b", "c", "g", "y", "b"]

        for finger_idx, finger_links in enumerate(links):
            for idx in range(len(finger_links) - 1):
                if color is not None:
                    link_color = color[finger_idx]
                else:
                    link_color = colors[finger_idx]
                self._draw2dseg(
                        ax,
                        annots,
                        finger_links[idx],
                        finger_links[idx + 1],
                        c=link_color,
                        alpha=alpha,
                        linewidth=linewidth,
                )

    def _draw2dseg(self, ax, annot, idx1, idx2, c="r", alpha=1, linewidth=1):
        ax.plot(
            [annot[idx1, 0], annot[idx2, 0]],
            [annot[idx1, 1], annot[idx2, 1]],
            c=c,
            alpha=alpha,
            linewidth=linewidth,
        )

    def get_camintr(self, cam_calib):
            return np.array(cam_calib).astype(np.float32) # shape: (3, 3)

    def transform(self, verts, trans, convert_to_homogeneous=False):
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

    def get_obj_mesh(self, obj_path): 
        mesh = trimesh.load(obj_path)
        return mesh

    def get_obj_verts(self, obj_path):
            verts = self.get_obj_mesh(obj_path).vertices
            return np.array(verts).astype(np.float32)

    def get_obj_verts_trans(self, obj_path, cam_extr, affine_transform):
            # Get object 3d vertices (n,3) in the camera coordinate frame
            verts = self.get_obj_verts(obj_path) # shape: (n, 3)
            trans_verts = self.transform(verts, self.get_obj_pose(cam_extr, affine_transform), convert_to_homogeneous=True)
            return np.array(trans_verts).astype(np.float32)

    def get_obj_pose(self, cam_extr, affine_transform):
            # Get object pose (3,4) matrix in world coordinate frame
            transform = cam_extr @ affine_transform
            return np.array(transform).astype(np.float32)

    def get_objcorners2d(self, cam_calib, obj_path, cam_extr, affine_transform):
            corners_3d = self.get_obj_corners3d(obj_path, cam_extr, affine_transform)
            intr = self.get_camintr(cam_calib)
            corners_2d_hom = self.transform(corners_3d, intr)
            corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:]
            return np.array(corners_2d).astype(np.float32)


    def get_obj_corners3d(self, obj_path, cam_extr, affine_transform):
            #model = self.get_obj_verts_trans(obj_path, cam_extr, affine_transform)
            model = self.get_obj_verts(obj_path) # shape: (n, 3)
            x_values = model[:, 0]
            y_values = model[:, 1]
            z_values = model[:, 2]
            min_x, max_x = np.min(x_values), np.max(x_values)
            min_y, max_y = np.min(y_values), np.max(y_values)
            min_z, max_z = np.min(z_values), np.max(z_values)
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
            corners_3d = self.transform(corners_3d, self.get_obj_pose(cam_extr, affine_transform), convert_to_homogeneous=True)
            return np.array(corners_3d).astype(np.float32)

    def get_objverts2d(self, cam_calib, obj_path, cam_extr, affine_transform):
            objpoints3d = self.get_obj_verts_trans(obj_path, cam_extr, affine_transform).transpose() # shape: (3, n)
            hom_2d = np.dot(self.get_camintr(cam_calib), objpoints3d).transpose() # shape: (n, 3)
            verts2d = (hom_2d / hom_2d[:, 2:])[:, :2] # shape: (n, 2)
            return np.array(verts2d).astype(np.float32)
    
    def visualize_hand_skeleton(self, metainfo, ax):
         gt_handjoints2d = metainfo["coords_2d"]
         
         if gt_handjoints2d is not None:
            self.visualize_joints_2d(ax, joints=gt_handjoints2d, joint_idxs=False, links=None, alpha=0.5, scatter=True, linewidth=2, color=None, joint_labels=None, axis_equal=False)

    def visualize_object_vertices_and_bounding_box(self, metainfo, ax):
        affine_transform = metainfo["affine_transform"]

        cam_calib = metainfo['cam_calib']
        obj_path = metainfo['obj_path']
        cam_extr = metainfo['cam_extr']

        #object_verts_2d = self.get_objverts2d(cam_calib, obj_path, cam_extr, affine_transform)
        object_corners_2d = self.get_objcorners2d(cam_calib, obj_path, cam_extr, affine_transform)

        # visualize sparse object vertices (every 100th vertex due to runtime performance)
        #object_verts_2d = [object_verts_2d[index] for index in range(1, len(object_verts_2d), 100)]
        #x_values=[i[0] for i in object_verts_2d]
        #y_values=[i[1] for i in object_verts_2d]
        
        #ax.scatter(x_values, y_values, 1, "b", alpha=0.02)

        self.visualize_joints_2d(
            ax,
            object_corners_2d,
            alpha=0.5,
            joint_idxs=False,
            links=[[0, 1, 3, 2], [4, 5, 7, 6], [1, 5], [3, 7], [4, 0], [0, 2, 6, 4]]
)
         

