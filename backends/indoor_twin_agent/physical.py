"""Write and verify portable Isaac-oriented OpenUSD, with independent rigid roots."""
import hashlib
import math
from pathlib import Path
import shutil
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade

from evidence import put_json, read_json, sha

# Prior masses, in kg. They describe plausible proxies for simulation.
DYNAMIC={'ExerciseBall':1.05,'LooseThrowPillow':.65,'StorageBox':1.8,'MobileEquipmentRig':15.}
MASSES={'TerracottaSofa':58.,'TrefoilWoodBench':16.,'OfficeDesk':34.,'RollingChair':13.,
        'OfficeShelf':38.,'FabricBarrier':7.,'ComputerTower':11.,'Beanbag':5.0}
RUBBER={'ball','beanbag'}
COLLIDER_OMIT=('hemp','hem','piping','vent','tape','indicator','stem','rail','mullion','overlap')


def rows(mat):
    return np.array(mat,dtype=float)


def vec3(v):
    return Gf.Vec3d(*[float(x) for x in v])


def mkdir_prim(layer,path,type_name):
    x=Sdf.CreatePrimInLayer(layer,path)
    x.specifier=Sdf.SpecifierDef
    x.typeName=type_name
    return x


def stage_meta(stage):
    UsdGeom.SetStageMetersPerUnit(stage,1)
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage,1)


def parts_from(stage,key):
    root=stage.GetPrimAtPath(f'/Prototypes/{key}')
    to_root=UsdGeom.XformCache()
    parts=[]
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh): continue
        mesh=UsdGeom.Mesh(prim)
        m=to_root.GetLocalToWorldTransform(prim)
        p=np.array([list(m.Transform(Gf.Vec3d(*map(float,v)))) for v in mesh.GetPointsAttr().Get()])
        bound=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        mat=bound.GetPath().name if bound else ''
        parts.append(dict(name=prim.GetName(),point=p,faces=list(mesh.GetFaceVertexCountsAttr().Get()),
                          indices=list(mesh.GetFaceVertexIndicesAttr().Get()),material=mat))
    if not parts: raise ValueError(f'{key} has no Blender geometry.')
    return parts


def signature(parts):
    h=hashlib.sha256()
    # Content hash independent of part names, order, materials and stage identities.
    payloads=[]
    for p in parts:
        blob=np.ascontiguousarray(p['point'],dtype='<f4').tobytes()
        blob+=np.ascontiguousarray(p['faces'],dtype='<i4').tobytes()
        blob+=np.ascontiguousarray(p['indices'],dtype='<i4').tobytes()
        payloads.append(hashlib.sha256(blob).digest())
    for item in sorted(payloads):
        h.update(item)
    return h.hexdigest()


def geometry_topology_signature(parts):
    """Ignore face emission order and cyclic first-vertex changes across Blender sessions."""
    pieces=[]
    for part in parts:
        faces=[]
        i=0
        for n in part['faces']:
            ids=tuple(part['indices'][i:i+n])
            i+=n
            pivot=ids.index(min(ids))
            faces.append(ids[pivot:]+ids[:pivot])
        payload=np.ascontiguousarray(part['point'],dtype='<f4').tobytes()
        payload+=repr(sorted(faces)).encode()
        pieces.append(hashlib.sha256(payload).digest())
    return hashlib.sha256(b''.join(sorted(pieces))).hexdigest()


def stored_topology(asset_dir,key):
    root=Path(asset_dir)
    file=root/'Geometry.usda'
    if not file.exists():
        file=root/'Geometry.usdc'
    stage=Usd.Stage.Open(str(file))
    return geometry_topology_signature(parts_from(stage,key))


def create_thumbnail(parts,materials,path):
    all_points=np.concatenate([p['point'] for p in parts])
    center=(all_points.max(0)+all_points.min(0))*.5
    right=np.array([.707,-.707,0])
    up=np.array([-.32,-.32,.89])
    look=np.cross(right,up)
    m=np.stack([right,up,look])
    points=(all_points-center)@m.T
    scale=180/max(0.01,float(np.ptp(points[:,:2],axis=0).max()))
    img=Image.new('RGB',(236,236),(35,42,49))
    d=ImageDraw.Draw(img)
    drawing=[]
    for p in parts:
        xyz=(p['point']-center)@m.T
        rgba=np.array(materials.get(p['material'],{}).get('color',[.58,.56,.53]))*255
        start=0
        for size in p['faces']:
            ids=p['indices'][start:start+size]
            start+=size
            vs=xyz[ids]
            xy=vs[:,:2]*[scale,-scale]+[118,117]
            normal=np.cross(vs[1]-vs[0],vs[2]-vs[0])
            k=.56+.44*min(1,abs(normal[1])/max(np.linalg.norm(normal),1e-7))
            color=tuple(np.clip(rgba*k+24,0,255).astype(int))
            drawing.append((vs[:,2].mean(),[tuple(v) for v in xy],color))
    for z,xy,c in sorted(drawing,key=lambda x:x[0]):
        d.polygon(xy,fill=c)
    img.save(path,quality=91,optimize=True)


def physics_info(key,semantic,query):
    native=query.get('template') or key
    if native in DYNAMIC:
        mode='dynamic'
        mass=DYNAMIC[native]
    elif query.get('motion')=='dynamic' and not native in MASSES and semantic not in {'person','barrier','window','door','light'}:
        mode='dynamic'
        mass=None
    else:
        mode='static'
        mass=MASSES.get(native)
    return dict(mode=mode,mass_kg=mass,density_kg_m3=320 if semantic in {'sofa','pillow','beanbag'} else 625,
                static_friction=.68 if semantic in RUBBER else .52,
                dynamic_friction=.54 if semantic in RUBBER else .41,
                restitution=.3 if semantic=='ball' else .055,
                collider_policy='sphere' if semantic=='ball' else 'compound_component_boxes',
                evidence='simulation prior, not measured physical properties')


def collider_recipe(parts,profile,semantic):
    all_pts=np.concatenate([p['point'] for p in parts])
    lo,hi=all_pts.min(0),all_pts.max(0)
    if profile['collider_policy']=='sphere':
        return [{'type':'sphere','center':((lo+hi)/2).tolist(),'size':(hi-lo).tolist()}]
    result=[]
    for p in parts:
        name=p['name'].lower()
        if any(x in name for x in COLLIDER_OMIT): continue
        if semantic=='window' and 'daylight' in name:
            # A pane is physically solid even if its visual is an emitter.
            pass
        po=p['point']
        a,b=po.min(0),po.max(0)
        dim=np.maximum(b-a,.01)
        result.append({'type':'box','center':((a+b)/2).tolist(),'size':dim.tolist(),'part':p['name']})
    if not result: raise ValueError('Asset has no collision support.')
    return result


def add_phys_material(stage,path,profile):
    m=UsdShade.Material.Define(stage,path)
    props=UsdPhysics.MaterialAPI.Apply(m.GetPrim())
    props.CreateStaticFrictionAttr(profile['static_friction'])
    props.CreateDynamicFrictionAttr(profile['dynamic_friction'])
    props.CreateRestitutionAttr(profile['restitution'])
    props.CreateDensityAttr(profile['density_kg_m3'])
    return m


def physics(stage,root,entry,scale):
    profile=entry['physics']
    root.CreateAttribute('twin:physicsMode',Sdf.ValueTypeNames.Token,custom=True).Set(profile['mode'])
    root.CreateAttribute('twin:physicsEvidence',Sdf.ValueTypeNames.String,custom=True).Set(profile['evidence'])
    p=root.GetPath()
    m=add_phys_material(stage,p.AppendChild('PhysicalMaterial'),profile)
    UsdGeom.Scope.Define(stage,p.AppendChild('Colliders'))
    ratios=np.array(scale)
    dynamic=profile['mode']=='dynamic'
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(True)
        mass=profile['mass_kg']
        if mass is None:
            mass=max(.1,profile['density_kg_m3']*sum(np.prod(c['size']) for c in entry['colliders'])*.38)
        mass=max(.03,float(mass)*float(np.prod(ratios)))
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
    elif profile['mass_kg'] is not None:
        root.CreateAttribute('twin:estimatedMassKg',Sdf.ValueTypeNames.Float,custom=True).Set(float(profile['mass_kg']))
    for i,collider in enumerate(entry['colliders']):
        path=p.AppendPath(f'Colliders/Collider_{i:03d}')
        dim=np.asarray(collider['size'])*ratios
        at=np.asarray(collider['center'])*ratios
        if collider['type']=='sphere' and max(dim)/min(dim)<1.05:
            s=UsdGeom.Sphere.Define(stage,path)
            s.CreateRadiusAttr(float(np.mean(dim)/2))
            s.AddTranslateOp().Set(vec3(at))
        else:
            s=UsdGeom.Cube.Define(stage,path)
            s.CreateSizeAttr(1.)
            s.AddTranslateOp().Set(vec3(at))
            s.AddScaleOp().Set(vec3(dim))
        s.CreatePurposeAttr(UsdGeom.Tokens.guide)
        prim=s.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(m,materialPurpose='physics')


def export_asset(staging,key,entry,folder):
    folder.mkdir(parents=True,exist_ok=True)
    materials=read_json(staging/'MaterialDefinitions.json')
    visual_layer=Sdf.Layer.FindOrOpen(str(staging/'AssetPrototypes.usda'))
    visual_stage=Usd.Stage.Open(visual_layer)
    source=f'/Prototypes/{key}'
    geom=parts_from(visual_stage,key)
    # Use each bound visual surface's individual PBR file rather than the entire scene library.
    looks=visual_stage.GetPrimAtPath(f'/Prototypes/{key}/Looks')
    used={p.GetName() for p in looks.GetChildren() if p.IsA(UsdShade.Material)}
    used_defs={name:materials[name] for name in used}
    src_mats=Sdf.Layer.FindOrOpen(str(staging/'Materials.usda'))
    mat_layer=Sdf.Layer.CreateNew(str(folder/'Materials.usda'))
    mkdir_prim(mat_layer,'/Materials','Scope')
    for n,desc in used_defs.items():
        Sdf.CopySpec(src_mats,Sdf.Path('/Materials/'+n),mat_layer,Sdf.Path('/Materials/'+n))
        for f in desc['files']:
            (folder/'Textures').mkdir(exist_ok=True)
            shutil.copyfile(staging/'Textures'/f,folder/'Textures'/f)
    mat_layer.Save()
    new_visual=Sdf.Layer.CreateNew(str(folder/'Geometry.usda'))
    mkdir_prim(new_visual,'/Prototypes','Scope')
    Sdf.CopySpec(visual_layer,Sdf.Path(source),new_visual,Sdf.Path(source))
    new_visual.subLayerPaths=['./Materials.usda']
    new_visual.Save()
    entry=dict(entry)
    entry['prototype_key']=key
    entry['geometry_sha256']=signature(geom)
    entry['geometry_topology_sha256']=geometry_topology_signature(geom)
    entry['material_definitions']=used_defs
    entry['physics']=physics_info(key,entry['semantic'],entry)
    entry['colliders']=collider_recipe(geom,entry['physics'],entry['semantic'])
    stage=Usd.Stage.CreateNew(str(folder/'Asset.usda'))
    stage_meta(stage)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/Asset').GetPrim())
    root=stage.GetDefaultPrim()
    root.CreateAttribute('twin:assetId',Sdf.ValueTypeNames.String,custom=True).Set(entry['asset_id'])
    visual=UsdGeom.Xform.Define(stage,'/Asset/Visual').GetPrim()
    visual.GetReferences().AddReference('./Geometry.usda',source)
    visual.SetInstanceable(True)
    physics(stage,root,entry,[1,1,1])
    stage.GetRootLayer().Save()
    shutil.copyfile(staging/'PrototypeBlends'/(key+'.blend'),folder/'Prototype.blend')
    create_thumbnail(geom,used_defs,folder/'Thumbnail.jpg')
    if stage.GetCompositionErrors() or not visual.IsInstance():
        raise ValueError(f'Broken visual composition for {key}.')
    put_json(folder/'Metadata.json',entry)
    return entry


def xform_matrix(prim,m):
    op=UsdGeom.Xformable(prim).AddTransformOp()
    op.Set(Gf.Matrix4d(*np.asarray(m).T.ravel().tolist()))


def scene(staging,manifest,selected,bank):
    root_path=staging/'Scene.usda'
    stage=Usd.Stage.CreateNew(str(root_path))
    stage_meta(stage)
    before=Usd.Stage.Open(str(staging/'IndoorTwin.usda'))
    stage.SetStartTimeCode(before.GetStartTimeCode())
    stage.SetEndTimeCode(before.GetEndTimeCode())
    stage.SetTimeCodesPerSecond(before.GetTimeCodesPerSecond())
    stage.SetFramesPerSecond(before.GetFramesPerSecond())
    world=UsdGeom.Xform.Define(stage,'/World').GetPrim()
    stage.SetDefaultPrim(world)
    ps=UsdPhysics.Scene.Define(stage,'/World/Physics')
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
    ps.CreateGravityMagnitudeAttr(9.81)
    for category in {x['category'] for x in manifest['instances']}:
        UsdGeom.Scope.Define(stage,f'/World/{category}')
    Sdf.CopySpec(before.GetRootLayer(),Sdf.Path('/World/Cameras'),stage.GetRootLayer(),Sdf.Path('/World/Cameras'))
    for light in before.GetPrimAtPath('/World/Lighting').GetChildren():
        if light.GetTypeName()=='RectLight':
            Sdf.CopySpec(before.GetRootLayer(),light.GetPath(),stage.GetRootLayer(),light.GetPath())
    assets=staging/'Assets'
    assets.mkdir(exist_ok=True)
    for entry in selected.values():
        catalog_path=bank.directory(entry['asset_id'])
        dest=assets/entry['asset_id']
        if not dest.exists():
            shutil.copytree(catalog_path,dest,ignore=shutil.ignore_patterns('Prototype.blend'))
    for item in manifest['instances']:
        key=item['asset_id']
        entry=selected[key]
        stem=Tf.MakeValidIdentifier(item['instance_id'])
        p=f'/World/{item["category"]}/{stem}'
        suffix=1
        while stage.GetPrimAtPath(p):
            p=f'/World/{item["category"]}/{stem}_{suffix:03d}'
            suffix+=1
        item['usd_prim_path']=p
        o=UsdGeom.Xform.Define(stage,p).GetPrim()
        o.CreateAttribute('twin:assetId',Sdf.ValueTypeNames.String,custom=True).Set(entry['asset_id'])
        o.CreateAttribute('twin:instanceId',Sdf.ValueTypeNames.String,custom=True).Set(item['instance_id'])
        scale=np.array(item['scale'])
        m=np.array(item['matrix_world_row_major'])
        m[:3,:3]/=scale
        xform_matrix(o,m)
        visual=UsdGeom.Xform.Define(stage,p+'/Visual')
        visual.GetPrim().GetReferences().AddReference(f'./Assets/{entry["asset_id"]}/Asset.usda','/Asset/Visual')
        visual.GetPrim().SetInstanceable(True)
        visual.AddScaleOp().Set(vec3(scale))
        physics(stage,o,entry,scale)
        item['asset_id']=entry['asset_id']
        item['prototype_key']=key
        item['physical_profile']=entry['physics']
        item['usd_reference']=f'Assets/{entry["asset_id"]}/Asset.usda'
        item['source']=f'catalog/{entry["asset_id"]}'
    put_json(staging/'SimulationManifest.json',manifest)
    stage.GetRootLayer().Save()
    return validate(stage,manifest)


def validate(stage,manifest):
    stats=Counter()
    if stage.GetCompositionErrors():
        raise ValueError(str(stage.GetCompositionErrors()))
    for i in manifest['instances']:
        p=stage.GetPrimAtPath(i.get('usd_prim_path',f'/World/{i["category"]}/{i["instance_id"]}'))
        v=p.GetChild('Visual')
        if not v.IsInstance() or not v.GetPrototype().GetChildren():
            raise ValueError(f'Missing composed visual for {p.GetPath()}')
        stats['instances']+=1
        stats['dynamic_roots' if p.HasAPI(UsdPhysics.RigidBodyAPI) else 'static_roots']+=1
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            assert UsdPhysics.MassAPI(p).GetMassAttr().Get()>0
        for c in p.GetChild('Colliders').GetChildren():
            assert c.HasAPI(UsdPhysics.CollisionAPI)
            mat=UsdShade.MaterialBindingAPI(c).ComputeBoundMaterial(materialPurpose='physics')[0]
            assert mat and mat.GetPrim().HasAPI(UsdPhysics.MaterialAPI)
            stats['colliders']+=1
        assert p.GetChild('Colliders').GetChildren()
    for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh):
            mat=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            assert mat and mat.GetPrim().IsValid(),prim.GetPath()
            stats['material_bound_meshes']+=1
        if prim.IsA(UsdShade.Shader):
            attr=prim.GetAttribute('inputs:file')
            if attr:
                val=attr.Get()
                assert val.resolvedPath and Path(val.resolvedPath).is_file(),attr.GetPath()
                stats['resolved_texture_references']+=1
    stats['camera_samples']=len(stage.GetPrimAtPath('/World/Cameras/TrajectoryCam').GetAttribute('xformOp:transform').GetTimeSamples())
    assert stats['camera_samples']>1
    return dict(stats,up_axis='Z',meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
                physics_gravity=[0,0,-9.81],isaac_sim_runtime_executed=False)
