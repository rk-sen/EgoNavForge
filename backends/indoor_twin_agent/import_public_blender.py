"""Batch import public assets, bake the complete assembly in one canonical coordinate system."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import math

import bpy
from mathutils import Matrix,Vector

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from usd_writer import write_material,write_mesh,identifier

NOMINAL_Z={'wall':2.7,'floor':.10,'door':2.10,'window':2.7,'wall_component':2.7,'baseboard':.13,
'chair':.94,'stool':.71,'desk':.75,'table':.74,'bench':.48,'sofa':.89,'bed':.82,'shelf':1.84,
'cabinet':.94,'monitor':.44,'keyboard':.045,'mouse':.042,'laptop':.25,'speaker':.31,'television':.61,
'radio':.19,'microwave':.34,'coffee_machine':.39,'fridge':.91,'washer':.87,'box':.33,'bin':.43,
'pillow':.16,'rug':.025,'teddy_bear':.39,'plant':.53,'book':.11,'lamp':.44,'ceiling_light':.18,
'floor_lamp':1.67,'ceiling_fan':.28,'coat_rack':1.75,'mirror':.65,'person':1.72}
ARCH={'wall','floor','door','window','wall_component','baseboard'}
LIGHTS={'ceiling_light','lamp','floor_lamp'}
FURNITURE={'chair','stool','desk','table','bench','sofa','bed','shelf','cabinet','plant','pillow','rug','coat_rack'}


def traced(socket):
    while socket and socket.is_linked:
        v=socket.links[0].from_socket
        node=v.node
        if node.type=='TEX_IMAGE' and node.image:
            return node.image,'rgb'
        if node.type in {'SEPARATE_COLOR','SEPRGB'}:
            im,_=traced(node.inputs[0])
            return im,{'Red':'r','Green':'g','Blue':'b','R':'r','G':'g','B':'b'}.get(v.name,'r')
        if node.type=='NORMAL_MAP':
            socket=node.inputs['Color']
        else:
            break
    return None,None


def save_material(mat,key,out,index):
    name=identifier(key+'_mat_'+str(index))
    mat.name=name
    base,rough,metal=list(mat.diffuse_color)[:3],.67,0.
    nodes=mat.node_tree.nodes if mat.use_nodes else ()
    bsdf=next((x for x in nodes if x.type=='BSDF_PRINCIPLED'),None)
    maps={}
    if bsdf:
        base=list(bsdf.inputs['Base Color'].default_value)[:3]
        rough=float(bsdf.inputs['Roughness'].default_value)
        metal=float(bsdf.inputs['Metallic'].default_value)
        for tag,in_name in [('Base','Base Color'),('Normal','Normal'),('Roughness','Roughness'),('Metallic','Metallic')]:
            im,channel=traced(bsdf.inputs[in_name])
            if im is None: continue
            file=f'{name}_{tag}.png'
            path=out/'Textures'/file
            was_filepath=im.filepath_raw
            was_format=im.file_format
            im.file_format='PNG'
            im.filepath_raw=str(path)
            im.save()
            im.file_format=was_format
            im.filepath_raw=was_filepath
            maps[tag]=dict(file=file,channel='rgb' if tag in {'Base','Normal'} else ('r' if channel=='rgb' else channel))
    return mat.name,dict(color=base,roughness=rough,metallic=metal,emission=0.,maps=maps,
                         files=[data['file'] for data in maps.values()])


def import_one(meta,src,out):
    key=meta['prototype_key']
    bpy.ops.wm.read_factory_settings(use_empty=True)
    suffix=src.suffix.lower()
    if suffix in {'.gltf','.glb'}:
        bpy.ops.import_scene.gltf(filepath=str(src))
    elif suffix=='.obj':
        bpy.ops.wm.obj_import(filepath=str(src))
    else:
        bpy.ops.import_scene.fbx(filepath=str(src),use_anim=False)
        rig=next(ob for ob in bpy.context.scene.objects if ob.type=='ARMATURE')
        # Pose the downloaded deform rig without rebinding its separate animation skeleton.
        inv=rig.matrix_world.to_3x3().inverted()
        for side,sign in [('Left',1.),('Right',-1.)]:
            for suffix,target in [('Arm',(sign*.14,-.06,-1.)),('ForeArm',(sign*.04,-.13,-1.))]:
                bone=rig.pose.bones[side+suffix]
                current=bone.matrix.to_3x3()@Vector((0,1,0))
                desired=inv@Vector(target)
                swing=current.rotation_difference(desired).to_matrix().to_4x4()
                posed=swing@bone.matrix
                posed.translation=bone.matrix.translation
                bone.matrix=posed
                bpy.context.view_layer.update()
        m=bpy.data.materials.new('Kenney_survivor')
        m.use_nodes=True
        inf=m.node_tree.nodes.new('ShaderNodeTexImage')
        inf.image=bpy.data.images.load(str(src.parents[1]/'Skins/survivorMaleB.png'))
        m.node_tree.links.new(inf.outputs['Color'],m.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
        for ob in bpy.context.scene.objects:
            if ob.type=='MESH': ob.data.materials.clear();ob.data.materials.append(m)
    bpy.context.view_layer.update()
    graph=bpy.context.evaluated_depsgraph_get()
    assembly=bpy.data.collections.new(key)
    bpy.context.scene.collection.children.link(assembly)
    baked=[]
    for i,ob in enumerate(list(bpy.context.scene.objects)):
        if ob.type!='MESH':continue
        evaluated=ob.evaluated_get(graph)
        data=bpy.data.meshes.new_from_object(evaluated,preserve_all_data_layers=True,depsgraph=graph)
        if not data.polygons:
            bpy.data.meshes.remove(data);continue
        data.transform(evaluated.matrix_world)
        inst=bpy.data.objects.new(f'{key}_part_{i:02}',data)
        assembly.objects.link(inst)
        baked.append(inst)
    if not baked: raise ValueError(f'No geometry in {src}')
    for ob in list(bpy.context.scene.objects):
        if ob not in baked: bpy.data.objects.remove(ob,do_unlink=True)
    lo=Vector([min(v.co[j] for ob in baked for v in ob.data.vertices) for j in range(3)])
    hi=Vector([max(v.co[j] for ob in baked for v in ob.data.vertices) for j in range(3)])
    if min(hi-lo)<1e-6: raise ValueError('Degenerate 3D bounds')
    center=Vector(((hi.x+lo.x)/2,(hi.y+lo.y)/2,lo.z))
    scale=NOMINAL_Z.get(meta['semantic'],hi.z-lo.z)/(hi.z-lo.z) if meta['provider']=='Kenney' else 1.
    # Bake all child transforms into the geometry together. No child is independently repositioned.
    for ob in baked: ob.data.transform(Matrix.Scale(scale,4)@Matrix.Translation(-center))
    if meta['semantic']=='person':
        if (hi-lo).x*scale>1.25 or (hi-lo).y*scale>1.25:
            raise ValueError('Expected the prebuilt idle pose, received a spread-limb pose')
    fallback=bpy.data.materials.new(key+'_untextured')
    fallback.diffuse_color=(.50,.48,.44,1)
    for ob in baked:
        if not ob.data.materials: ob.data.materials.append(fallback)
        else:
            for i,mat in enumerate(ob.data.materials):
                if mat is None: ob.data.materials[i]=fallback
    out.mkdir(parents=True,exist_ok=True)
    (out/'Textures').mkdir(exist_ok=True)
    (out/'PrototypeBlends').mkdir(exist_ok=True)
    used=sorted({mat for ob in baked for mat in ob.data.materials},key=lambda x:x.name)
    mats=dict(save_material(mat,key,out,i) for i,mat in enumerate(used))
    bpy.ops.file.pack_all()
    bpy.data.libraries.write(str(out/'PrototypeBlends'/(key+'.blend')),{assembly},path_remap='NONE',fake_user=True,compress=True)
    with (out/'Materials.usda').open('w') as f:
        f.write('#usda 1.0\ndef Scope "Materials" {\n')
        for name,record in mats.items(): write_material(f,name,record)
        f.write('}\n')
    with (out/'AssetPrototypes.usda').open('w') as f:
        f.write('#usda 1.0\ndef Scope "Prototypes" {\n')
        f.write(f'    def Xform "{key}" {{\n')
        f.write('        def Scope "Looks" {\n')
        for name in mats: f.write(f'            def Material "{name}" (prepend references = @Materials.usda@</Materials/{name}>) {{}}\n')
        f.write('        }\n')
        bpy.context.view_layer.update()
        graph=bpy.context.evaluated_depsgraph_get()
        for ob in baked: write_mesh(f,ob,graph,key)
        f.write('    }\n}\n')
    (out/'MaterialDefinitions.json').write_text(json.dumps(mats,indent=2))
    if meta['semantic']=='person':
        meta.update(preferred_pose=True,forward_axis='-Y',pose='standing; arms down; original rig posed and baked as one group')
    meta.update(dimensions_m=[float(x)*scale for x in hi-lo],canonical_root='bottom center, Z-up',
                input_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),mesh_count=len(baked),
                material=None,shape_parts=[meta['semantic']],motion='static')
    (out/'Intake.json').write_text(json.dumps(meta,indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sources',type=Path,required=True);p.add_argument('--staging',type=Path,required=True)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    mapping=json.loads((ROOT/'asset_sources.json').read_text())
    source=a.sources.resolve()
    staging=a.staging.resolve()
    todo=[]
    for provider,rows in mapping.items():
        for tag,semantic in rows.items():
            if provider=='innerscene':
                f=source/'innerscene'/tag/'model.glb'
                if f.is_file(): todo.append(('Innerscene',tag,semantic,f,'https://www.innerscene.com/tools/library/3d-parts/'+tag))
                continue
            pattern=(source/provider/tag).glob('*.gltf') if provider=='polyhaven' else (source/provider).rglob(tag+'.obj')
            f=next(pattern,None)
            if f:
                page='https://polyhaven.com/a/'+tag if provider=='polyhaven' else 'https://opengameart.org/content/furniture-kit'
                todo.append(('Poly Haven' if provider=='polyhaven' else 'Kenney',tag,semantic,f,page))
    human=source/'human/Model/characterMedium.fbx'
    todo.append(('Kenney','SurvivorMaleStanding','person',human,'https://opengameart.org/content/animated-characters-1'))
    failures=[]
    for provider,tag,semantic,src,page in todo:
        key=identifier({'Poly Haven':'PH_','Kenney':'KN_','Innerscene':'IS_'}[provider]+tag)
        category='Architecture' if semantic in ARCH else 'Lighting' if semantic in LIGHTS else 'Furniture' if semantic in FURNITURE else 'Equipment'
        meta=dict(query_id=key,prototype_key=key,semantic=semantic,category=category,
                  provider=provider,source_page=page,license='CC0-1.0',source_file=src.name,
                  description=re.sub(r'[_-]',' ',tag),public_prefab=True)
        if (staging/key/'Intake.json').exists(): continue
        try:
            import_one(meta,src,staging/key)
            print('imported',key,flush=True)
        except Exception as e:
            failures.append([key,str(e)])
            print('failed',key,repr(e),flush=True)
    (staging/'import_failures.json').write_text(json.dumps(failures,indent=2))
    if failures: raise RuntimeError('Some curated source assets failed intake')


if __name__=='__main__': main()
