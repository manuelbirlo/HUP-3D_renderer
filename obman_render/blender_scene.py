import bpy
import numpy as np
import os
import sys
import random

root = '.'
sys.path.insert(0, root)
mano_path = os.environ.get('MANO_LOCATION', None)
if mano_path is None:
    raise ValueError('Environment variable MANO_LOCATION not defined'
                     'Please follow the README.md instructions')
sys.path.insert(0, os.path.join(mano_path, 'webuser'))

from obman_render import mesh_manip, render, texturing, camutils, coordutils
from serialization import load_model
from smpl_handpca_wrapper import load_model as smplh_load_model
from mathutils import Matrix

from mathutils import Matrix, Vector, Quaternion
from scipy.spatial.transform import Rotation as R

class BlenderScene:

    def __init__(self, render_body=True, ambiant_mean=0.7, ambiant_add=0.5):
        self.render_body = render_body
        self.scene = bpy.data.scenes['Scene']
        # Clear default scene cube
        bpy.ops.object.delete()

        # Set rendering params
        self.scene.use_nodes = True
        self.scene.render.engine = 'CYCLES'
        self.scene.cycles.device = 'GPU'
        self.scene.cycles.film_transparent = True
        self.scene.cycles.shading_system = False
        self.scene.render.resolution_x = 256
        self.scene.render.resolution_y = 256
        self.scene.render.resolution_percentage = 100

        # Use AI denoising
        bpy.context.view_layer.cycles.use_denoising = True

        # Get camera info
        self.camera_name = 'Camera'
        camutils.set_camera(self.camera_name)
        camutils.check_camera(self.camera_name) 
        self.cam_calib = np.array(camutils.get_calib_matrix(self.camera_name))
        self.cam_extr = np.array(camutils.get_extrinsic(self.camera_name))


    def addObjectTexture(self, obj_path, obj_textures, random_obj_textures, texture_zoom=1):
        # Create object material if none is present
        if len(self.obj_mesh.materials) == 0:
            mat = bpy.data.materials.new(name='{}_mat'.format(self.obj_mesh.name))
            bpy.ops.object.material_slot_add()
            self.obj.material_slots[0].material = mat

        for mat_idx, obj_mat in enumerate(self.obj_mesh.materials):
            if random_obj_textures:
                obj_texture = random.choice(obj_textures)
                generated_uv = True
            else:
                obj_texture = os.path.join(os.path.dirname(obj_path), "texture_" + obj_mat.name + ".jpg")
                generated_uv = False

            print("obj_mat: " + str(obj_mat))
            print("obj_texture: " + str(obj_texture))
            print("down_scale: " + str(texture_zoom))
            print("generated_uv: " + str(generated_uv))

            texturing.add_obj_texture(
                obj_mat,
                obj_texture,
                down_scale=texture_zoom,
                tmp_suffix='tmp',
                generated_uv=generated_uv)

        meta_infos = {}

        if random_obj_textures:
            meta_infos['obj_texture'] = obj_texture

        return meta_infos


    def clearUnused(self):
        for item in bpy.data.meshes:
            if item.users == 0:
                bpy.data.meshes.remove(item)
        for item in bpy.data.images:
            if item.users == 0:
                bpy.data.images.remove(item)


    def setToolMaterialPassIndices(self):
        for mat_idx, obj_mat in enumerate(self.obj_mesh.materials):
            #obj_mat.use_nodes = True
            obj_mat.pass_index = 100


    def checkCamera(self):
        camutils.check_camera(camera_name=self.camera_name)


    def deleteMaterials(self):
        # Remove materials
        for material in self.obj_mesh.materials:
            material.user_clear()
            bpy.data.materials.remove(material, do_unlink=True)


    def deleteObject(self):
        if self.obj is not None:
            bpy.data.objects.remove(self.obj, do_unlink=True)
            self.obj = None


    def loadSMPLHModel(self, mano_right_path, smpl_model_path, smpl_data_path,
                      texture_zoom = 1):

        # Load smpl2mano correspondences
        self.right_smpl2mano = np.load('assets/models/smpl2righthand_verts.npy')

        # Load SMPL+H model
        self.ncomps = 45
        self.smplh_model = smplh_load_model(
            smpl_model_path, ncomps=2 * self.ncomps, flat_hand_mean=True)
        self.mano_model = load_model(mano_right_path)
        mano_mesh = bpy.data.meshes.new('Mano')
        mano_mesh.from_pydata(list(np.array(self.mano_model.r)), [], list(self.mano_model.f))
        self.mano_obj = bpy.data.objects.new('Mano', mano_mesh)
        bpy.context.scene.collection.objects.link(self.mano_obj)
        self.mano_obj.hide_render = True

        print('Loaded mano model')

        # Load smpl info
        self.smpl_data = np.load(smpl_data_path)
        self.smplh_obj = mesh_manip.load_smpl()
        # Smooth the edges of the body model
        bpy.ops.object.shade_smooth()

        self.materials = texturing.initialize_texture(
            self.smplh_obj, texture_zoom=texture_zoom, tmp_suffix='tmp')

    def setup_vertex_color_material(self, obj, layer_name="Col"):
        # Create a new material with vertex color support
        mat = bpy.data.materials.new(name="VertexColorMaterial")
        obj.data.materials.append(mat)
        
        # Enable use of nodes for the material
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear existing nodes
        while nodes:
            nodes.remove(nodes[0])
        
        # Create necessary nodes and link them
        vertex_color_node = nodes.new(type='ShaderNodeVertexColor')
        vertex_color_node.layer_name = layer_name
        shader_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        links.new(vertex_color_node.outputs['Color'], shader_node.inputs['Base Color'])
        links.new(shader_node.outputs['BSDF'], output_node.inputs['Surface'])

    def loadObject(self, ply_path, obj_scale=1.0):
        # Ensure the path is for a .ply file
        ply_path = os.path.splitext(ply_path)[0] + ".ply"
        print("ply_path: {}".format(ply_path))

        # Load PLY object model
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.import_mesh.ply(filepath=ply_path)

        # Assuming the imported PLY object is the only selected object
        self.obj = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = self.obj

        # Assign the mesh data to self.obj_mesh
        self.obj_mesh = self.obj.data

        # Apply scale and rotation
        obj_scale = float(obj_scale)
        self.obj.scale = (obj_scale, obj_scale, obj_scale)
        self.obj.rotation_euler = (0, 0, 0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

        bpy.ops.object.shade_smooth()  # Set shading to smooth

        # Setup vertex color material
        if len(self.obj.data.vertex_colors) > 0:
            self.setup_vertex_color_material(self.obj, "Col")

        if self.obj.data.vertex_colors:
            print("Vertex Color Layer Name:", self.obj.data.vertex_colors[0].name)

        # Metadata about the object
        meta_infos = {
            'obj_path': ply_path,
            'obj_scale': obj_scale
        }

        return meta_infos

    def renderRGB(self, img_path, bg_path, depth_path, folder_tmp_segm,
                  hide_smplh=False, hide_obj=False):
        self.scene.render.filepath = img_path
        self.scene.render.image_settings.file_format = 'JPEG'
        self.smplh_obj.hide_render = hide_smplh
        self.obj.hide_render = hide_obj

        tmp_segm_path = render.set_cycle_nodes(
            self.scene,
            bg_path,
            segm_path=folder_tmp_segm,
            depth_path=depth_path,
            bg_scale=1.0)
            #bg_scale=0.5)
        bpy.ops.render.render(write_still=True)
        return tmp_segm_path

    def apply_additional_rotation(self, obj, rotmat):
        # Convert the rotmat list to a Blender Matrix
        rotation_matrix = Matrix(rotmat)

        # Ensure it's a 4x4 matrix by appending a [0, 0, 0, 1] row, if needed
        if len(rotation_matrix) < 4:
            rotation_matrix.resize_4x4()

        # Apply the rotation to the object
        obj.matrix_world = rotation_matrix @ obj.matrix_world

    def apply_rotation_to_mesh(self, mesh, rotmat):
        # Convert the rotation matrix to a Blender Matrix object
        rot_matrix_blender = Matrix(rotmat)
        
        # Ensure it's a 4x4 matrix by appending a [0, 0, 0, 1] row, if needed
        if len(rot_matrix_blender) < 4:
            rot_matrix_blender.resize_4x4()

        # Apply the rotation to each vertex of the mesh
        for vert in mesh.vertices:
            vert.co = rot_matrix_blender @ vert.co

    def setHandAndObjectPose(self, grasp, z_min, z_max, cam_view, z_distance, debug_data_file_writer = None):
        assert len(cam_view) == 3, "Input tuple 'cam_view' does not have required length 3"

        # Set hand pose
        if 'mano_trans' in grasp:
            print("_____________ if 'mano_trans' in grasp:_________________")
            self.mano_model.trans[:] = [val for val in grasp['mano_trans']]
        else:
            self.mano_model.trans[:] = grasp['hand_trans']
            
            # Adjust the probe location (x = 0, y = 0, z = z of hand_trans)
            #self.obj.location = (0, 0, grasp['hand_trans'][2])
            
            #self.mano_model.trans[:] = np.array(grasp['hand_trans']) #grasp['hand_trans']

            print("______grasp['hand_trans']____{}".format(grasp['hand_trans']))
            print("______self.mano_model.trans____{}".format(self.mano_model.trans))

        print("_______________ ROTMAT {}".format(np.array(grasp['rotmat'])))
        # Define the local translation vector (assuming translation along the z-axis)
        #local_translation = np.array([0, 0, -grasp['hand_trans'][1]])
        z_axis_offset = 0.07189549170510294 # A fixed z-axis offset between the default input voluson_painted.ply located at the origin and any voluson_painted.ply that is the result of POV_Surgery's grabnet-based grasp generation.
        local_translation = np.array([0, 0, -z_axis_offset])
        
        # Transform the local translation vector using the rotation matrix
        global_translation = np.dot(np.array(grasp['rotmat']), local_translation)

        # Apply the transformed translation vector to the object's global location
        self.obj.location += Vector(global_translation)
        #self.obj.location += Vector(local_translation)

        self.mano_model.pose[:] = grasp['hand_pose'] # passing 48 values of mano_pose
        mesh_manip.alter_mesh(self.mano_obj, self.mano_model.r.tolist())


        # Center mesh on center_idx
        # You can even pass random_shape and random_pose =False
        smplh_verts, posed_model, meta_info = mesh_manip.randomized_verts(
            self.smplh_model,
            self.smpl_data,
            ncomps=2 * self.ncomps,
            z_min=z_min,
            z_max=z_max,
            z_distance=z_distance,
            side='right',
            hand_pose= grasp['pca_pose'],
            hand_pose_offset=0,
            random_shape=False,
            random_pose=False,
            cam_viewpoint_x = cam_view[0],
            cam_viewpoint_y=cam_view[1],
            cam_viewpoint_z=cam_view[2],
            debug_data_file_writer=debug_data_file_writer)

        mesh_manip.alter_mesh(self.smplh_obj, smplh_verts.tolist())

        hand_info = coordutils.get_hand_body_info(
            posed_model,
            render_body=self.render_body,
            side='right',
            cam_extr=self.cam_extr,
            cam_calib=self.cam_calib,
            right_smpl2mano=self.right_smpl2mano)

        meta_infos = {**hand_info, **meta_info}
        meta_infos['cam_extr'] = self.cam_extr
        meta_infos['cam_calib'] = self.cam_calib

        # Apply transform to object
        rigid_transform = coordutils.get_rigid_transform_posed_mano(posed_model, self.mano_model)
        self.mano_obj.matrix_world = Matrix(rigid_transform)
        
        obj_transform = rigid_transform.dot(self.obj.matrix_world)
        self.obj.matrix_world = Matrix(obj_transform)

        # desired_world_translation = Vector((0, 0, grasp['hand_trans'][2]))

        # Apply the translation after considering the object's rotation
        #self.apply_translation_after_rotation(self.obj, desired_world_translation)

        print("______________________________________ object location after rigid transform:_______{}_______".format(self.obj.location))
        print("______________________________________ hand location after rigid transform:_______{}_______".format(self.mano_obj.location))
        """
        additional_rotmat = [
            [0.9999619126319885, -0.00013189652236178517, -0.008725538849830627, 0],
            [0.0, 0.9998857975006104, -0.015114419162273407, 0],
            [0.008726535364985466, 0.015113843604922295, 0.9998477101325989, 0],
            [0, 0, 0, 1]  # homogeneous coordinate for 4x4 matrix
        ]
            
        # Apply rotmat to the hand mesh vertices
        #self.apply_rotation_to_mesh(self.mano_obj.data, additional_rotmat)

        # Calculate the transformation for the object
        obj_transform = rigid_transform @ self.obj.matrix_world
        
        # Apply rotmat to the object mesh vertices
        self.apply_rotation_to_mesh(self.obj.data, additional_rotmat)

        # Update the world transformation of the object
        self.obj.matrix_world = Matrix(obj_transform)
        """



        # self.obj.matrix_world is an identity matrix at this point because a value hasn't been assigned to it yet.
        #obj_transform = rigid_transform.dot(self.obj.matrix_world)
        #self.obj.matrix_world = Matrix(obj_transform)


        # --------------------------------------------------------------
        # Apply rotmat to the hand mesh vertices
        #self.apply_rotation_to_mesh(self.mano_obj.data, grasp['rotmat'])
        # --------------------------------------------------------------

        # Save object info
        meta_infos['affine_transform'] = obj_transform.astype(np.float32)
        """
        # Load the additional rotation matrix (rotmat)
      
        additional_rotmat = [
            [0.9999619126319885, -0.00013189652236178517, -0.008725538849830627, 0],
            [0.0, 0.9998857975006104, -0.015114419162273407, 0],
            [0.008726535364985466, 0.015113843604922295, 0.9998477101325989, 0],
            [0, 0, 0, 1]  # homogeneous coordinate for 4x4 matrix
        ]
       
        # Apply the additional rotation to obj_transform
        # Convert your NumPy array (additional_rotation_matrix) to a Blender Matrix
        additional_rotation_matrix_blender = Matrix(additional_rotmat)
 
        # Convert obj_transform (if it's a numpy.ndarray) to a Blender Matrix
        if isinstance(obj_transform, np.ndarray):
            obj_transform = Matrix(obj_transform.tolist())
 
        # Now multiply the Blender Matrices
        obj_transform = additional_rotation_matrix_blender @ obj_transform
 
        # Apply the final transformation to the tool object
        self.obj.matrix_world = obj_transform

        #self.obj.matrix_world = Matrix(obj_transform)
        # After applying all transformations to obj_transform
        # Convert Blender Matrix to numpy array for storing in meta_infos
        obj_transform_np = np.array(obj_transform)  # Converts the Blender Matrix to a NumPy array

        # Save object info
        meta_infos['affine_transform'] = obj_transform_np.astype(np.float32)
        # Save object info
        #meta_infos['affine_transform'] = obj_transform.astype(np.float32)
        """


        return meta_infos


    def setHandTextures(self):
        self.setHandMaterial(self.materials['rightForeArm'].node_tree,
                             #color=(0.315, 0.395, 0.5),
                             color=(1.0, 0.6784313725, 0.3764705882), # skin color
                             #roughness=0.8)
                             roughness=1.0)
        self.setHandMaterial(self.materials['rightHand'].node_tree,
                             #color=(0.37, 0.458, 0.5),
                             color=(0.5647058824, 0.5921568627, 0.768627451),
                             roughness=0.8)
        self.setHandMaterial(self.materials['rightHandIndex1'].node_tree,
                             #color=(0.37, 0.458, 0.5),
                             color=(0.5647058824, 0.5921568627, 0.768627451),
                             roughness=0.8)


    def setHandMaterial(self, tree, color=(1.0, 1.0, 1.0), roughness=0.5):
        for n in tree.nodes:
            tree.nodes.remove(n)

        rgbNode = tree.nodes.new('ShaderNodeRGB')
        rgbNode.location = (0,0)
        rgbNode.outputs[0].default_value[:3] = color
        valNode = tree.nodes.new('ShaderNodeValue')
        valNode.location = (0, 200)
        valNode.outputs[0].default_value = roughness

        refNode = tree.nodes.new('ShaderNodeBsdfRefraction')
        refNode.location = (200, 0)
        tree.links.new(rgbNode.outputs[0], refNode.inputs[0])
        tree.links.new(valNode.outputs[0], refNode.inputs[1])

        gloNode = tree.nodes.new('ShaderNodeBsdfGlossy')
        gloNode.location = (200, 200)
        tree.links.new(rgbNode.outputs[0], gloNode.inputs[0])
        tree.links.new(valNode.outputs[0], gloNode.inputs[1])

        mixNode = tree.nodes.new('ShaderNodeMixShader')
        mixNode.location = (400, 0)
        mixNode.inputs[0].default_value = 0.5
        tree.links.new(refNode.outputs[0], mixNode.inputs[1])
        tree.links.new(gloNode.outputs[0], mixNode.inputs[2])

        outNode = tree.nodes.new('ShaderNodeOutputMaterial')
        outNode.location = (600, 0)
        tree.links.new(mixNode.outputs[0], outNode.inputs[0])


    def setLighting(self):
        # get lamp object
        lamps = [obj for obj in bpy.data.objects if obj.type == 'LIGHT']
        self.lamp_obj = lamps[0]

        # set random lamp location
        self.lamp_obj.location = np.random.uniform([-0.6, 0.1, 0.5], [0.6, 0.4, 0.5], 3)
        self.lamp_obj.rotation_euler = np.radians([0.0, 0.0, 0.0])
        #self.lamp_obj.location = np.random.uniform([-0.15, -0.1, 0.0], [-0.05, 0.1, 0.0], 3)
        #rot = np.random.uniform([-10.0, -20.0, 0.0], [10.0, -10.0, 0.0], 3)
        #self.lamp_obj.rotation_euler = np.radians(rot)
        #self.lamp_obj.data.type = 'SPOT'
        #self.lamp_obj.data.spot_size = np.radians(25.0)

        rg = np.random.uniform(0.6, 1.0)
        b = np.random.uniform(0.6, 1.0)
        self.lamp_obj.data.color = (rg, rg, b)
        pwr = np.random.uniform(50, 300)
        self.lamp_obj.data.energy = pwr

        meta_infos = {}
        return meta_infos


    def setSMPLTexture(self, tex_path):
        # Update body+hands image
        tex_img = bpy.data.images.load(tex_path)
        for part, material in self.materials.items():
            if 'Image Texture' in material.node_tree.nodes:
                material.node_tree.nodes['Image Texture'].image = tex_img



