import os
import pickle

import numpy as np


# create one material per part as defined in a pickle with the segmentation
# this is useful to render the segmentation in a material pass
def create_segmentation(obj):
    import bpy
    print('creating segmentation')
    materials = {}
    vgroups = {}
    with open('assets/segms/segm_per_v_overlap.pkl', 'rb') as f:
        vsegm = pickle.load(f)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.material_slot_remove()
    parts = sorted(vsegm.keys())
    sorted_parts = [
        'hips', 'leftUpLeg', 'rightUpLeg', 'spine', 'leftLeg', 'rightLeg',
        'spine1', 'leftFoot', 'rightFoot', 'spine2', 'leftToeBase',
        'rightToeBase', 'neck', 'leftShoulder', 'rightShoulder', 'head',
        'leftArm', 'rightArm', 'leftForeArm', 'rightForeArm', 'leftHand',
        'rightHand', 'leftHandIndex1', 'rightHandIndex1'
    ]
    part2num = {part: (ipart + 1) for ipart, part in enumerate(sorted_parts)}
    for part in parts:
        vs = vsegm[part]
        vgroups[part] = obj.vertex_groups.new(name=part)
        vgroups[part].add(vs, 1.0, 'ADD')
        bpy.ops.object.vertex_group_set_active(group=part)

        # Duplicates sh_material to all body parts
        mater = bpy.data.materials['Material'].copy()
        materials[part] = mater

        materials[part].pass_index = part2num[part]
        bpy.ops.object.material_slot_add()
        obj.material_slots[-1].material = materials[part]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.vertex_group_select()
        bpy.ops.object.material_slot_assign()
        bpy.ops.object.mode_set(mode='OBJECT')
    return (materials)


def create_sh_material(tree, down_scale=1):
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # rgb = tree.nodes.new('ShaderNodeRGB')
    uv = tree.nodes.new('ShaderNodeTexCoord')
    uv.location = -800, 400

    uv_xform = tree.nodes.new('ShaderNodeMapping')
    uv_xform.location = -600, 400
    uv_xform.inputs['Scale'].default_value = (down_scale, down_scale, down_scale)

    uv_im = tree.nodes.new('ShaderNodeTexImage')
    uv_im.location = -400, 400

    emission = tree.nodes.new('ShaderNodeEmission')
    emission.location = -60, 400
    emission.inputs[1].default_value = 0.5

    mat_out = tree.nodes.new('ShaderNodeOutputMaterial')
    mat_out.location = 110, 400

    tree.links.new(uv.outputs[2], uv_xform.inputs[0])
    tree.links.new(uv_xform.outputs[0], uv_im.inputs[0])
    tree.links.new(uv_im.outputs[0], emission.inputs[0])
    tree.links.new(emission.outputs[0], mat_out.inputs[0])


def add_obj_texture(mater,
                    obj_texture_path,
                    down_scale=1.0,
                    tmp_suffix='',
                    generated_uv=True):
    import bpy

    # Initialize sh texture
    mater.use_nodes = True
    mater.pass_index = 100
    tree = mater.node_tree
    for n in tree.nodes:
        tree.nodes.remove(n)

    # rgb = tree.nodes.new('ShaderNodeRGB')
    uv = tree.nodes.new('ShaderNodeTexCoord')
    uv.location = -800, 400

    uv_xform = tree.nodes.new('ShaderNodeMapping')
    uv_xform.location = -600, 400
    uv_xform.inputs['Scale'].default_value = (down_scale, down_scale, down_scale)

    uv_im = tree.nodes.new('ShaderNodeTexImage')
    # uv_im.projection = 'TUBE'
    uv_im.location = -400, 400

    # the emission node makes it independent of the scene lighting
    emission = tree.nodes.new('ShaderNodeEmission')
    emission.location = -60, 400

    mat_out = tree.nodes.new('ShaderNodeOutputMaterial')
    mat_out.location = 110, 400

    if generated_uv:
        tree.links.new(uv.outputs[0], uv_xform.inputs[0])
    else:
        tree.links.new(uv.outputs[2], uv_xform.inputs[0])
    tree.links.new(uv_xform.outputs[0], uv_im.inputs[0])
    tree.links.new(uv_im.outputs[0], emission.inputs[0])
    tree.links.new(emission.outputs[0], mat_out.inputs[0])

    tex_img = bpy.data.images.load(obj_texture_path)
    tree.nodes['Image Texture'].image = tex_img


def initialize_texture(obj, texture_zoom=1, tmp_suffix=''):
    import bpy
    mater = bpy.data.materials['Material']
    mater.use_nodes = True
    node_tree = mater.node_tree

    create_sh_material(node_tree, down_scale=texture_zoom)
    materials = create_segmentation(obj)
    return materials
