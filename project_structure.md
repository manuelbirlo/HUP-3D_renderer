# Project Structure
COMING SOON.

# Dataset Structure
COMING SOON.

## Metadata
All attributes given in the metadata are explained below. Unless noted otherwise, all values are given in meters and radians.

| Attribute | Contents | Type/Shape |
| ------ | ------ | ------ |
| obj_path | Path to the object model | string |
| obj_scale | Scale factor for model | float |
| side | Left or right hand | string |
| coords_2d | Hand 2D positions (ordering defined below) | (21, 2) |
| coords_3d | Hand 3D positions (ordering defined below) | (21, 3) |
| verts_3d |  | (778, 3) |
| full_body_3d |  | (22, 3) |
| full_body_2d |  | (22, 2) |
| z |  |  |
| trans |  | (3, 1) |
| pose |  | (156, 1) |
| shape |  | (10, 1) |
| mano_pose |  | (45, 1) |
| cam_extr | Camera extrinsics matrix | (3, 4) |
| cam_calib | Camera intrinsics matrix | (3, 3) |
| affine_transform | Object transformation matrix | (4, 4) |
| pca_pose | Pose in PCA representation (45 principle components) | (45, 1) |
| hand_trans | Global hand translation | (3, 1) |
| hand_global_rot | Global hand rotation | (3, 1) |
| hand_pose |  | (48, 1) |
| bg_path | Path to background image | string |
| body_tex | Path to body texture file | string |
| sh_coeffs |  | (9,1) |
| obj_visibility_ratio | in \% | float |
| depth_min | Minimal depth of hand and object | float | 
| depth_max | Maximal depth of hand and object | float |
| hand_depth_min | Minimal depth of hand | float |
| hand_depth_max | Maximal depth of hand | float |
| obj_depth_min | Minimal depth of object | float |
| obj_depth_max | Maximal depth of object | float |

### Hand Joint Ordering
The ordering of the hand joints (for both coords_2d and coords_3d) is as follows:
```
[Wrist, TMCP, TPIP, TDIP, TTIP, IMCP, IPIP, IDIP, ITIP, MMCP, MPIP, MDIP, MTIP, RMCP, RPIP, RDIP, RTIP, PMCP, PPIP, PDIP, PTIP]
```
![](assets/hand_model.png)

(Hand model visualization taken from [First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations](https://github.com/guiggh/hand_pose_action))