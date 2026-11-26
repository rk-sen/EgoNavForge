"""Build this surveyed indoor scene inside Blender with Blender Python."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from usd_writer import write_usd

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--preview', action='store_true')
parser.add_argument('--survey',type=Path,required=True)
parser.add_argument('--dispatch',type=Path,required=True)
parser.add_argument('--samples',type=int,default=56)
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
args.input = args.input.resolve()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
(OUT/'Textures').mkdir(exist_ok=True)
survey = json.loads(args.survey.read_text(encoding='utf8'))
dispatch = json.loads(args.dispatch.read_text(encoding='utf8'))
target = int(survey['target_index'])
needed=set(dispatch)
native_build={key for key,row in dispatch.items() if row['branch']=='BUILD' and not row.get('template')}
native_build.update(row['template'] for row in dispatch.values() if row['branch']=='BUILD' and row.get('template'))

def source(pattern):
    matches = sorted(args.input.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f'Expected one {pattern} in {args.input}: {matches}')
    return matches[0]

video_path, pose_path, intr_path = source('*.mp4'), source('*trajectory*.json'), source('*pinhole*.json')
cloud_path=next(args.input.glob('*.ply'),None)
intr = json.loads(intr_path.read_text())
track = json.loads(pose_path.read_text())
K = np.array(intr['K_pinhole_pipeline'])
W, H = intr['pipeline_wh']
clock_path = args.input/'video_clock.csv'
with clock_path.open() as f:
    pts = [float(row[0]) for row in csv.reader(f) if row and row[0]]
if len(pts) < 2 or any(np.diff(pts) <= 0):
    raise ValueError('The video clock must contain increasing decoded-frame timestamps.')
T = np.array(survey['source_to_room_rotation'])
origin = np.array(survey['room_origin_in_source_m'])
LEVEL = Matrix(np.column_stack((T, -T@origin)).tolist()+[[0,0,0,1]])
assert abs(np.linalg.det(T) - 1) < 1e-6
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.name = 'Indoor Twin | metric Z-up'
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1
scene.render.fps = 25
scene.render.resolution_x, scene.render.resolution_y = W,H
# Blender clamps each component to >=1, so preserve ratios on either side of unity.
pixel_ratio=K[0,0]/K[1,1]
scene.render.pixel_aspect_x=max(1,1/pixel_ratio)
scene.render.pixel_aspect_y=max(1,pixel_ratio)
scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_percentage = 100
scene.render.engine = 'CYCLES'
scene.cycles.samples = args.samples
scene.cycles.adaptive_threshold = .055
scene.cycles.sample_clamp_indirect = 2.2
scene.cycles.use_denoising = False
scene.view_settings.view_transform = 'AgX'
scene.view_settings.look = 'AgX - Medium High Contrast'
scene.view_settings.exposure = -.53
scene.world = bpy.data.worlds.new('Indoor ambient')
scene.world.use_nodes = True
scene.world.node_tree.nodes.get('Background').inputs[0].default_value = (.61,.68,.84,1)
scene.world.node_tree.nodes.get('Background').inputs[1].default_value = .095

def group(name):
    col = bpy.data.collections.new(name)
    scene.collection.children.link(col)
    return col

categories = {name: group(name) for name in ('Architecture','Furniture','Equipment','Lighting','Cameras','Reference')}
cats = {
    'Architecture':'e31b7d92-8326-44bd-9326-3419cefc6350',
    'Furniture':'6d3027fe-7fec-47de-bbab-eac56d06b533',
    'Equipment':'1e2d6eb8-a3b3-4ab4-a2cb-ad9213bbcdee',
    'Lighting':'ebbca08e-b451-4adb-957e-99c95d69338c',
}
(OUT/'blender_assets.cats.txt').write_text('VERSION 1\n\n' + ''.join(
    f'{uid}:{label}:{label}\n' for label,uid in cats.items()))

MATS = {}
def texture(key, style, col):
    size = 512 if style == 'floor' else 256
    v,u = np.mgrid[0:size,0:size].astype(np.float32)/size
    rng = np.random.default_rng(62)
    grit = rng.normal(0,1,u.shape).astype(np.float32)
    line = np.sin(2*np.pi*(4*u+.16*np.sin(2*np.pi*v)))
    shade = 1+.055*grit
    relief = .5+.10*grit
    if style == 'floor':
        edge = np.minimum.reduce([u,v,1-u,1-v])
        grout = np.clip(edge/.0065,0,1)
        shade = .19+.81*grout + .04*grit+.03*np.sin(6*np.pi*(u+v))
        relief = .19+.34*grout+.027*grit
        # Small pairs of pale fastener points near each raised-floor corner.
        for sx,sy in ((.027,.025),(.97,.025),(.027,.974),(.97,.974)):
            mark = (u-sx)**2+(v-sy)**2 < .004**2
            shade[mark] = 1.8
    elif style == 'wood':
        phase = 2*np.pi*(11*v+.38*np.sin(2*np.pi*u)+.13*np.cos(4*np.pi*u))
        shade = .93+.16*np.sin(phase)+.095*np.sin(phase*2)+.037*grit
        relief = .5+.11*np.sin(phase)+.035*grit
    elif style in {'cloth','leather','rubber','blind'}:
        weave = np.sin(2*np.pi*28*u)*np.sin(2*np.pi*27*v)
        shade = .98+.054*weave+.023*grit+.045*line
        relief = .5+.11*weave+.052*grit
        if style == 'rubber':
            shade += .12*np.cos(2*np.pi*15*v)**14
            relief += .11*np.cos(2*np.pi*15*v)**14
    base = np.asarray(col,dtype=np.float32)
    pixels = np.clip(shade[:,:,None]*base[None,None,:],0,1)
    du = (np.roll(relief,-1,axis=1)-np.roll(relief,1,axis=1)) * (.95 if style=='floor' else .56)
    dv = (np.roll(relief,-1,axis=0)-np.roll(relief,1,axis=0)) * (.95 if style=='floor' else .56)
    normal = np.stack((-du,-dv,np.ones_like(v)),axis=-1)
    normal = .5+.5*(normal/np.linalg.norm(normal,axis=-1,keepdims=True))
    result = []
    for suffix,rgb,cs in [('base',pixels,'sRGB'),('normal',normal,'Non-Color')]:
        name = f'{key}_{suffix}.png'
        rgba = np.concatenate((rgb,np.ones_like(v)[:,:,None]),axis=-1)
        img = bpy.data.images.new(name,size,size,alpha=True)
        img.colorspace_settings.name = cs
        img.pixels.foreach_set(np.ascontiguousarray(rgba,dtype=np.float32).ravel())
        img.filepath_raw = str(OUT/'Textures'/name)
        img.file_format='PNG'
        img.save()
        img.pack()
        img.filepath = '//Textures/'+name
        result.append(img)
    return result

def material(key, color, rough=.62, metal=0, style=None, emission=0):
    col = tuple(float(x) for x in color)
    mat = bpy.data.materials.new(key)
    mat.diffuse_color = (*col,1)
    mat.use_nodes=True
    tree=mat.node_tree
    bs=tree.nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value = (*col,1)
    bs.inputs['Roughness'].default_value = rough
    bs.inputs['Metallic'].default_value = metal
    bs.inputs['Emission Color'].default_value = (*col,1)
    bs.inputs['Emission Strength'].default_value = emission
    if emission and hasattr(mat.cycles, 'emission_sampling'):
        mat.cycles.emission_sampling = 'NONE'
    files = []
    if style:
        a,n = texture(key,style,col)
        for im, pos in ((a,(-620,130)),(n,(-620,-160))):
            node = tree.nodes.new('ShaderNodeTexImage')
            node.image = im
            node.location = pos
            if im == a:
                tree.links.new(node.outputs['Color'],bs.inputs['Base Color'])
            else:
                bump=tree.nodes.new('ShaderNodeNormalMap')
                bump.inputs['Strength'].default_value=.18
                bump.location=(-280,-120)
                tree.links.new(node.outputs['Color'],bump.inputs['Color'])
                tree.links.new(bump.outputs['Normal'],bs.inputs['Normal'])
            files.append(im.name)
    MATS[key] = dict(color=col,roughness=rough,metallic=metal,emission=emission,files=files)
    return mat

material('Floor_graphite',(.164,.17,.178),.26,style='floor')
material('Paint_ivory',(.72,.69,.60),.77,style='paint')
material('Ceiling_concrete',(.56,.58,.60),.81,style='paint')
material('Trim_dark',(.026,.029,.034),.62,.12)
material('Aluminium',(.48,.49,.49),.28,.78)
material('Barrier_indigo',(.050,.036,.259),.85,style='cloth')
material('Foot_red',(.51,.021,.031),.64)
material('Terracotta',(.43,.225,.142),.61,style='leather')
material('Cushion_ocher',(.55,.30,.13),.84,style='cloth')
material('Cushion_gray',(.30,.29,.25),.89,style='cloth')
material('Beanbag_charcoal',(.20,.192,.20),.83,style='cloth')
material('Desk_laminate',(.62,.62,.58),.46,style='paint')
material('Walnut',(.25,.144,.081),.46,style='wood')
material('Rubber_green',(.045,.46,.255),.42,style='rubber')
material('Blind_slate',(.44,.45,.47),.8,style='blind')
material('Chair_dark',(.082,.084,.09),.53,style='cloth')
material('Jacket_gray',(.47,.47,.45),.89,style='cloth')
material('Screen',(.052,.078,.10),.3,emission=.13)
material('Cardboard',(.45,.35,.239),.81,style='paint')
material('Daylight',(.43,.68,1),.28,emission=5.0)
material('LED_white',(.92,.96,1),.47,emission=3.9)
material('Skin_proxy',(.47,.29,.20),.74)
material('Marker_green',(.14,.66,.28),.46,emission=1.3)

ASSETS, INSTANCES = {}, []
current = None
def asset(key, cat, note):
    global current
    current = bpy.data.collections.new(key)
    current.asset_mark()
    current.asset_data.catalog_id = cats[cat]
    current.asset_data.description = note
    current.asset_data.tags.new(cat)
    current['asset_id'] = key
    ASSETS[key] = (current, cat, note)
    return current

def put(obj, name, mat):
    for col in tuple(obj.users_collection):
        col.objects.unlink(obj)
    current.objects.link(obj)
    obj.name = name
    obj.data.materials.append(bpy.data.materials[mat])
    return obj

def mesh(name, verts, faces, mat, tex_scale=1, uv_func=None, smooth=False):
    data = bpy.data.meshes.new(name)
    data.from_pydata(verts,[],faces)
    data.update()
    uv = data.uv_layers.new(name='UVMap')
    for poly in data.polygons:
        norm=poly.normal
        axis=max(range(3),key=lambda a:abs(norm[a]))
        a,b = ((1,2),(0,2),(0,1))[axis]
        poly.use_smooth=smooth
        for lk in poly.loop_indices:
            n=data.loops[lk].vertex_index
            pt=verts[n]
            uv.data[lk].uv=uv_func(n) if uv_func else (pt[a]/tex_scale,pt[b]/tex_scale)
    obj=bpy.data.objects.new(name,data)
    return put(obj,name,mat)

def box(name, loc, size, mat, bevel=.012, rot=None, tex_scale=1):
    x,y,z=[v*.5 for v in size]
    verts=[(a*x,b*y,c*z) for a,b,c in
        [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]]
    obj=mesh(name,verts,[(3,2,1,0),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)],mat,tex_scale)
    obj.location=loc
    if rot:
        obj.rotation_euler=rot
    if bevel:
        mod=obj.modifiers.new('Rounded corners','BEVEL')
        mod.width=min(bevel, min(size)*.39)
        mod.segments=3
        mod=obj.modifiers.new('Weighted normals','WEIGHTED_NORMAL')
        mod.keep_sharp=True
    return obj

def sphere(name, loc, scale, mat, style='sphere'):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=36,ring_count=24,location=loc)
    obj=put(bpy.context.object,name,mat)
    for vert in obj.data.vertices:
        x,y,z=vert.co
        if style=='bag':
            # A lowered seat at the front and a higher, asymmetric back.
            fac=1+.07*math.sin(7*x+4*y)*math.cos(5*z)+.032*math.sin(11*y)
            vert.co=(x*scale[0]*fac,y*scale[1]*fac,
                     max(-.28,(z*.81+.22*x-.30*math.exp(-((x+.25)**2+y*y)*4)*max(z,0))*scale[2]*fac))
        elif style=='pillow':
            vert.co=tuple(math.copysign(abs(t)**.62,t)*s for t,s in zip((x,y,z),scale))
        else:
            vert.co=(x*scale[0],y*scale[1],z*scale[2])
    for p in obj.data.polygons: p.use_smooth=True
    return obj

def rod(name, a, b, r, mat, r2=None):
    d=Vector(b)-Vector(a)
    bpy.ops.mesh.primitive_cone_add(vertices=16,radius1=r,radius2=r if r2 is None else r2,
                                   depth=d.length,location=(Vector(a)+Vector(b))*.5)
    ob=put(bpy.context.object,name,mat)
    ob.rotation_euler=d.to_track_quat('Z','Y').to_euler()
    for p in ob.data.polygons:
        p.use_smooth=len(p.vertices)==4
    return ob

def pose_asset(name, key, xyz, yaw=0, scale=(1,1,1), evidence='cloud_and_video', tilt=(0,0)):
    prototype, cat, _ = ASSETS[key]
    ob=bpy.data.objects.new(name,None)
    categories[cat].objects.link(ob)
    ob.instance_type='COLLECTION'
    ob.instance_collection=prototype
    ob.empty_display_type='PLAIN_AXES'
    ob.empty_display_size=.10
    ob.location=xyz
    ob.rotation_euler=[math.radians(v) for v in (*tilt,yaw)]
    ob.scale=scale
    ob['asset_id']=key
    ob['asset_category']=cat
    ob['instance_id']=name
    ob['placement_evidence']=evidence
    INSTANCES.append(ob)
    return ob

def light(name, loc, target, watts, width, length, color=(1,1,1)):
    data=bpy.data.lights.new(name,'AREA')
    data.energy=watts
    data.shape='RECTANGLE'
    data.size=width
    data.size_y=length
    data.color=color
    ob=bpy.data.objects.new(name,data)
    categories['Lighting'].objects.link(ob)
    ob.location=loc
    ob.rotation_euler=(Vector(target)-ob.location).to_track_quat('-Z','Y').to_euler()
    return ob

if 'WallSegment' in native_build:
    asset('WallSegment','Architecture','Unit wall segment with painted finish. Scale about its base.')
    box('PaintedWall',(0,0,.5),(1,1,1),'Paint_ivory',0)
if 'CorridorWall' in native_build:
    asset('CorridorWall','Architecture','Unglazed painted wall used for the dark, unlit interior return.')
    box('DarkCorridorWall',(0,0,.5),(1,1,1),'Trim_dark',0)
if 'Column' in native_build:
    asset('Column','Architecture','Painted rectangular column with skirting.')
    box('ColumnBody',(0,0,.5),(1,1,1),'Paint_ivory',0)
    box('Skirting',(0,0,.013),(1.02,1.02,.026),'Trim_dark',.004)
if 'CeilingSlab' in native_build:
    asset('CeilingSlab','Architecture','Painted concrete ceiling slab with a downward-facing interior.')
    box('CeilingSlab',(0,0,.5),(1,1,1),'Ceiling_concrete',0)
if 'CeilingBeam' in native_build:
    asset('CeilingBeam','Architecture','Rectangular exposed ceiling beam.')
    box('Beam',(0,0,.5),(1,1,1),'Ceiling_concrete',.012)
if 'LabRaisedFloor' in native_build:
    asset('LabRaisedFloor','Architecture','Graphite raised floor with repeating 0.82 m tile texture.')
    floor=box('Floor',(0,0,-.07),(18.3,21.8,.14),'Floor_graphite',0,tex_scale=.82)
    # Offset the floor UVs so a joint crossing coincides with the window-area video landmark.
    for loop in floor.data.uv_layers.active.uv:
        loop.vector.x += (1.25-.16)/.82
        loop.vector.y += (-6.64-.63)/.82
if 'WindowBay' in native_build:
    # Real sealed window: aluminium frame, glazing, one vertical mullion and
    # one transom.  Local bbox 1.49 x 0.20 x 2.33, origin at opening centre.
    asset('WindowBay','Architecture','Sealed aluminium-framed window with glazing, mullion and transom.')
    glass=bpy.data.materials.new('Window_glass')
    glass.use_nodes=True
    b=glass.node_tree.nodes.get('Principled BSDF')
    b.inputs['Base Color'].default_value=(.63,.78,.88,1)
    b.inputs['Roughness'].default_value=.08
    b.inputs['Transmission Weight'].default_value=.72
    b.inputs['Alpha'].default_value=.55
    glass.blend_method='BLEND'
    MATS['Window_glass']=dict(color=[.63,.78,.88],roughness=.08,metallic=0,emission=0,files=[])
    box('WindowGlass',(0,0,0),(1.36,.018,2.18),'Window_glass',.004)
    for x in (-.715,.715):
        box(f'FrameSide_{x:+.2f}',(x,0,0),(.06,.20,2.33),'Aluminium',.008)
    for z in (-1.135,1.135):
        box(f'FrameRail_{z:+.2f}',(0,0,z),(1.49,.20,.06),'Aluminium',.008)
    box('Mullion',(0,0,0),(.045,.16,2.20),'Aluminium',.006)
    box('Transom',(0,0,.55),(1.40,.16,.045),'Aluminium',.006)
    box('Sill',(0,-.07,-1.18),(1.52,.28,.07),'Aluminium',.008)
if 'WoodPartition' in native_build:
    # Observed office cubicle divider: timber posts, rails and inset acoustic
    # panels.  Deliberately unrelated to the lab portable fabric barrier.
    asset('WoodPartition','Furniture','Timber office partition with posts, rails and inset panels.')
    for x in (-.80,.80):
        box(f'Post_{x:+.2f}',(x,0,.76),(.09,.15,1.52),'Walnut',.008)
    box('TopRail',(0,0,1.49),(1.72,.15,.06),'Walnut',.006)
    box('BottomRail',(0,0,.045),(1.72,.15,.09),'Walnut',.006)
    for i,x in enumerate((-.53,0,.53)):
        box(f'Panel_{i}',(x,0,.78),(.47,.055,1.36),'Walnut',.004)
        box(f'PanelInset_{i}',(x,-.035,.78),(.40,.012,1.22),'Cushion_gray' if i%2 else 'Cushion_ocher',.003)
if 'DoorPortal' in native_build:
    asset('DoorPortal','Architecture','Door frame with open passage and separate timber jambs.')
    for y in (-.68,.68):
        box('Jamb',(0,y,1.28),(.19,.084,2.56),'Walnut',.009)
    box('Header',(0,0,2.61),(.19,1.44,.11),'Walnut')
    box('Threshold',(0,0,.012),(.22,1.39,.024),'Trim_dark',.006)

if 'FabricBarrier' in native_build:
    asset('FabricBarrier','Furniture','Portable indigo fabric barrier with tubular frame and red feet.')
    width,height=1.62,1.13
    vs=[]
    for j in range(16):
        v=j/15
        for i in range(33):
            u=i/32
            x=(u-.5)*width
            z=.065+v*height-.044*math.sin(math.pi*u)**2*v**3
            y=(.019*math.sin(12*u+4*v)+.012*math.sin(31*u-8*v))*math.sin(math.pi*u)*math.sin(math.pi*v)
            vs.append((x,y,z))
    faces=[(32*j+j+i,33*j+i+1,33*(j+1)+i+1,33*(j+1)+i) for j in range(15) for i in range(32)]
    ob=mesh('WrinkledFabric',vs,faces,'Barrier_indigo',uv_func=lambda i:(i%33/32,i//33/15),smooth=True)
    ob.modifiers.new('Fabric thickness','SOLIDIFY').thickness=.007
    for x in (-width*.5,width*.5):
        rod('Upright',(x,0,.027),(x,0,1.205),.016,'Aluminium')
        box('RedSupportFoot',(x,0,.022),(.16,.32,.044),'Foot_red',.018)
    rod('BottomRail',(-.81,.013,.06),(.81,.013,.06),.016,'Aluminium')

if 'ExerciseBall' in native_build:
    asset('ExerciseBall','Furniture','Green textured exercise ball with equatorial ribs.')
    sphere('GreenShell',(0,0,.342),(.342,.342,.342),'Rubber_green')
if 'Beanbag' in native_build:
    asset('Beanbag','Furniture','Asymmetric dark fabric beanbag with a recessed sitting surface.')
    sphere('LowerShell',(0,0,.28),(.57,.49,.44),'Beanbag_charcoal','bag')
    sphere('RoundedBase',(0,0,.15),(.54,.475,.15),'Beanbag_charcoal','pillow')
    seam=[(.555*math.cos(a),.465*math.sin(a),.125+.025*math.sin(a)) for a in np.linspace(0,2*math.pi,29)]
    for a,b in zip(seam,seam[1:]): rod('Hem',a,b,.0055,'Chair_dark')

if 'TerracottaSofa' in native_build:
    asset('TerracottaSofa','Furniture','Terracotta two-seat upholstered sofa. Front faces local -X.')
    box('RecessedPlinth',(.015,0,.055),(.88,1.91,.11),'Trim_dark',.027)
    box('UpholsteredBase',(0,0,.205),(.98,1.97,.31),'Terracotta',.075)
    box('SingleSeat',(-.04,0,.41),(.76,1.63,.15),'Terracotta',.065)
    box('Backrest',(.386,0,.59),(.185,1.79,.43),'Terracotta',.067,rot=(0,-.11,0))
    for side in (-1,1):
        box('RoundedArm',(0,side*.929,.455),(1.02,.165,.72),'Terracotta',.057)
        rod('ArmPiping',(-.438,side*.938,.16),(-.438,side*.938,.73),.006,'Cushion_ocher')
if 'LooseThrowPillow' in native_build:
    asset('LooseThrowPillow','Furniture','Two-tone throw pillow with contrasting fabric on its underside.')
    sphere('Body',(0,0,0),(.165,.235,.073),'Cushion_ocher','pillow')
    sphere('Underside',(-.003,0,-.027),(.16,.229,.047),'Cushion_gray','pillow')
if 'TrefoilWoodBench' in native_build:
    asset('TrefoilWoodBench','Furniture','Long walnut bench with splayed legs and joined horizontal stretchers.')
    box('Slab',(0,0,.659),(.445,1.98,.081),'Walnut',.018,tex_scale=.75)
    for sy in (-1,1):
        for sx in (-1,1):
            rod('SplayedLeg',(sx*.205,sy*.802,.018),(sx*.133,sy*.735,.627),.029,'Walnut',.038)
        rod('TransverseStretcher',(-.177,sy*.77,.263),(.177,sy*.77,.263),.021,'Walnut')
    rod('LongStretcher',(0,-.77,.263),(0,.77,.263),.023,'Walnut')

if 'OfficeDesk' in native_build:
    asset('OfficeDesk','Furniture','Work surface, metal legs, modesty panel and lower cross rail.')
    box('Worktop',(0,0,.965),(1.74,.75,.061),'Desk_laminate',.014)
    box('ModestyPanel',(0,.24,.63),(1.54,.041,.37),'Desk_laminate')
    for x in (-.78,.78):
        for y in (-.28,.28):
            box('DeskLeg',(x,y,.48),(.053,.052,.95),'Aluminium')
        box('EndRail',(x,0,.22),(.041,.54,.041),'Aluminium')
if 'FlatMonitor' in native_build:
    asset('FlatMonitor','Equipment','Dark monitor with a bezel, screen and stand. Screen faces local -Y.')
    box('Bezel',(0,0,.285),(.64,.064,.383),'Trim_dark',.018)
    box('LCD',(0,-.035,.292),(.597,.008,.338),'Screen',.008)
    box('Stem',(0,.026,.113),(.068,.045,.185),'Trim_dark')
    box('Foot',(0,-.005,.016),(.265,.17,.032),'Trim_dark',.01)
if 'ComputerTower' in native_build:
    asset('ComputerTower','Equipment','Black workstation tower with front grille and status light.')
    box('Enclosure',(0,0,.335),(.34,.57,.67),'Trim_dark',.014)
    for n in range(9):
        box('Vent',(0,-.289,.14+n*.025),(.24,.007,.009),'Chair_dark',.002)
    box('PowerIndicator',(.084,-.291,.545),(.015,.007,.014),'Marker_green',.002)
if 'RollingChair' in native_build:
    asset('RollingChair','Furniture','Task chair with casters, pedestal, seat and fabric back. Front faces -Y.')
    sphere('Seat',(0,-.035,.49),(.269,.25,.075),'Chair_dark','pillow')
    box('Back',(0,.205,.755),(.49,.097,.51),'Chair_dark',.084,rot=(-.12,0,0))
    rod('Stem',(0,0,.095),(0,0,.48),.042,'Aluminium')
    for i in range(5):
        a=2*math.pi*i/5
        x,y=.285*math.cos(a),.285*math.sin(a)
        rod('StarBase',(0,0,.15),(x,y,.070),.032,'Trim_dark')
        sphere('Caster',(x,y,.048),(.041,.039,.043),'Trim_dark')
    for x in (-.285,.285):
        rod('ArmSupport',(x,.022,.49),(x,.022,.66),.018,'Trim_dark')
        box('ArmPad',(x,-.046,.675),(.057,.239,.039),'Chair_dark',.019)
if 'DrapedJacket' in native_build:
    asset('DrapedJacket','Furniture','Scanned appearance proxy for the gray garment draped on a chair back.')
    nv,nu=19,22
    verts=[]
    for j in range(nv):
        v=j/(nv-1)
        for i in range(nu):
            u=i/(nu-1)
            x=(u-.5)*(.52+.13*v)
            y=.022*math.sin(37*u+8*v)+.015*math.sin(20*u-5*v)+.095*math.sin(math.pi*v)
            z=(1-v)*.51-.069*math.cos(2*math.pi*u)*v**2
            verts.append((x,y,z))
    faces=[(j*nu+i,j*nu+i+1,(j+1)*nu+i+1,(j+1)*nu+i) for j in range(nv-1) for i in range(nu-1)]
    ob=mesh('Cloth',verts,faces,'Jacket_gray',uv_func=lambda i:(i%nu/(nu-1),i//nu/(nv-1)),smooth=True)
    ob.modifiers.new('Cloth thickness','SOLIDIFY').thickness=.012

if 'OfficeShelf' in native_build:
    asset('OfficeShelf','Furniture','Open steel storage rack with several wood shelves.')
    for x in (-.67,.67):
        for y in (-.235,.235):
            box('SteelPost',(x,y,1.02),(.048,.046,2.04),'Trim_dark')
    for z in (.16,.56,1.0,1.45,1.90):
        box('Shelf',(0,0,z),(1.40,.51,.047),'Walnut',.007)
if 'StorageBox' in native_build:
    asset('StorageBox','Equipment','Plain cardboard storage carton.')
    box('Box',(0,0,.15),(.40,.33,.30),'Cardboard',.007)
    box('Tape',(0,0,.302),(.073,.33,.004),'Desk_laminate',0)
if 'MobileEquipmentRig' in native_build:
    asset('MobileEquipmentRig','Equipment','Low mobile lab equipment stand reconstructed as a visual proxy.')
    box('SteelTray',(0,0,.095),(.49,.43,.079),'Aluminium',.027)
    box('Equipment',(0,.015,.25),(.25,.24,.25),'Trim_dark',.025)
    rod('HandleA',(-.19,.13,.09),(-.19,.13,.48),.017,'Aluminium')
    rod('HandleB',(.19,.13,.09),(.19,.13,.48),.017,'Aluminium')
    rod('Handle',(-.19,.13,.48),(.19,.13,.48),.017,'Aluminium')
    for x,y in ((-.19,-.16),(.19,-.16),(-.19,.16),(.19,.16)):
        sphere('Wheel',(x,y,.045),(.058,.046,.055),'Trim_dark')
if 'StandingPersonProxy' in native_build:
    asset('StandingPersonProxy','Equipment','Static low-detail mannequin for the observed person. No animated pose or identity.')
    sphere('Head',(0,0,1.68),(.104,.095,.135),'Skin_proxy')
    sphere('Hair',(0,.031,1.72),(.108,.1,.12),'Chair_dark')
    sphere('Torso',(0,0,1.27),(.229,.117,.288),'Chair_dark','pillow')
    for x in (-.108,.108):
        rod('TrouserLeg',(x,.018,.10),(x,0,.94),.084,'Chair_dark')
        box('Shoe',(x,-.055,.047),(.135,.25,.094),'Trim_dark',.024)
    for s in (-1,1):
        rod('Sleeve',(.218*s,.0,1.42),(.275*s,-.04,1.07),.068,'Chair_dark',.082)
        rod('Forearm',(.275*s,-.04,1.07),(.249*s,-.18,.998),.048,'Chair_dark',.067)
if 'StripLuminaire' in native_build:
    asset('StripLuminaire','Lighting','Linear ceiling luminaire with luminous diffuser.')
    box('Housing',(0,0,.031),(1,.105,.062),'Aluminium',.009)
    box('LEDStrip',(0,0,-.008),(.985,.078,.019),'LED_white',.005)


for key,entry in dispatch.items():
    if entry['branch'] == 'BUILD':
        if key in ASSETS:
            continue
        if entry.get('template'):
            native=ASSETS[entry['template']][0]
            asset(key,entry['category'],entry['description'])
            props=entry['material']
            finish=material('finish_'+key,props['color'],props['roughness'],props['metallic'],props['texture'])
            for template in native.objects:
                obj=template.copy()
                obj.name=template.name
                current.objects.link(obj)
                for slot in obj.material_slots:
                    if slot.material and slot.material.name.split('.')[0] in entry['replace_materials']:
                        slot.link='OBJECT'
                        slot.material=finish
            continue
        recipe=entry['recipe']
        c=entry['category']
        asset(key,c,entry['description'])
        props=entry['material']
        palette=material('finish_'+key,props['color'],props['roughness'],props['metallic'],props['texture'])
        role={'body':palette.name,'frame':'Aluminium','accent':'Walnut','dark':'Trim_dark'}
        for p in recipe:
            center=p['center']
            size=p['size']
            if p['primitive']=='box':
                ob=box(p['name'],center,size,role[p['material_role']],.018)
            elif p['primitive']=='sphere':
                ob=sphere(p['name'],center,[v/2 for v in size],role[p['material_role']])
            else:
                bpy.ops.mesh.primitive_cylinder_add(vertices=28,radius=.5,depth=1,location=center)
                ob=put(bpy.context.object,p['name'],role[p['material_role']])
                for v in ob.data.vertices:
                    v.co=tuple(v.co[i]*size[i] for i in range(3))
            ob.rotation_euler=[math.radians(v) for v in p['rotation_deg']]
        continue
    source_key=entry['source_key']
    existing=set(bpy.data.objects)
    with bpy.data.libraries.load(entry['blend_path'],link=False) as (orig,dest):
        if source_key not in orig.collections:
            raise ValueError(f'Catalog prototype missing {source_key}.')
        dest.collections=[source_key]
    current=dest.collections[0]
    current.name=key
    current['asset_id']=key
    ASSETS[key]=(current,entry['category'],entry['description'])
    if entry['branch']=='MATERIAL_VARIANT':
        props=entry['material']
        finish=material('finish_'+key,props['color'],props['roughness'],props['metallic'],props['texture'])
        for obj in current.objects:
            for slot in obj.material_slots:
                if slot.material and slot.material.name.split('.')[0] in entry['replace_materials']:
                    slot.link='OBJECT'
                    slot.material=finish
    # Restore canonical material IDs for USD. Catalog material descriptions include custom finishes.
    MATS.update(entry['material_definitions'])
    for ob in current.objects:
        if ob in existing:
            raise RuntimeError('Unexpected shared imported object.')
        for slot in ob.material_slots:
            if slot.material:
                name=slot.material.name.split('.')[0]
                if name!=slot.material.name and bpy.data.materials.get(name):
                    slot.link='OBJECT'
                    slot.material=bpy.data.materials[name]

ASSETS={k:v for k,v in ASSETS.items() if k in dispatch}
prototype_dir=OUT/'PrototypeBlends'
prototype_dir.mkdir(exist_ok=True)
(OUT/'MaterialDefinitions.json').write_text(json.dumps(MATS,indent=2))
for key,(collection,cat,note) in ASSETS.items():
    if dispatch[key]['branch'] != 'REUSE':
        bpy.data.libraries.write(str(prototype_dir/(key+'.blend')),{collection},
                                 fake_user=True,path_remap='NONE',compress=True)
for item in survey['instances']:
    pose_asset(*item)
for item in survey['barrier_runs']:
    tag,start,end,num,h,confidence=item
    a,b=Vector(start),Vector(end)
    delta=(b-a)/num
    for n in range(num):
        p=a+(n+.5)*delta
        pose_asset(f'{tag}_{n+1:02}', 'FabricBarrier', p,
                   math.degrees(math.atan2(delta.y,delta.x)), (delta.length/1.64,1,h/1.20), confidence)
for item in survey['window_centres']:
    n,x,y,z,s = item
    pose_asset(n,'WindowBay',(x,y,z),0,s,'cloud_and_video')
    light(n+'_Daylight',(x,y-.14,z+.05),(x,y-2,z-.4),195,1.28*s[0],1.2*s[2],(.67,.81,1))
for item in survey['light_strips']:
    name,loc,length,yaw,watts=item
    pose_asset(name,'StripLuminaire',loc,yaw,(length,1,1),'cloud_and_video')
    area=light(name+'_Area',(loc[0],loc[1],loc[2]-.065),(loc[0],loc[1],0),watts,length,.18,(.95,.98,1))
    area.rotation_euler.z=math.radians(yaw)

def camera(name):
    dat=bpy.data.cameras.new(name)
    dat.sensor_fit='HORIZONTAL'
    dat.sensor_width=36
    dat.lens=36*K[0,0]/W
    dat.shift_x=(W/2-K[0,2])/W
    dat.shift_y=(K[1,2]-H/2)/W*pixel_ratio
    dat.clip_start=.025
    dat.clip_end=100
    ob=bpy.data.objects.new(name,dat)
    categories['Cameras'].objects.link(ob)
    return ob

pos_entries={s['row']:s for s in track['subsample_xyz']}
pose_entries={s['pose_row']:s for s in track['key_frames']}
def time_at(row): return (pose_entries[row] if row in pose_entries else pos_entries[row])['time_s']
def scene_frame(t): return 1+t*scene.render.fps

anchors={}
for p in track['key_frames']:
    row=p['pose_row']
    raw=Matrix(p['blender_matrix_world'])
    if row in anchors:
        if max(abs(t-s) for a,b in zip(raw,anchors[row][0]) for t,s in zip(a,b))>1e-5:
            raise ValueError('Conflicting camera matrices on a repeated pose row.')
        continue
    anchors[row]=(raw,LEVEL@raw)
positions={s['row']:Vector(T@(np.array(s['xyz'])-origin)) for s in track['subsample_xyz']}
for row,(_,world) in anchors.items():
    positions[row]=world.translation
pr=sorted(positions)
ar=sorted(anchors)
rots={r:anchors[r][1].to_quaternion().normalized() for r in ar}

# Optional video-registered orientation knots use image/point-cloud tie points, with source positions fixed.
def fit_rotation(position, pairs):
    a=[]; b=[]
    for uv,xyz in pairs:
        x,y=np.linalg.solve(K,np.array([*uv,1]))[:2]
        ray=np.array([x,-y,-1.])
        v=np.asarray(xyz)-np.asarray(position)
        a.append(ray/np.linalg.norm(ray))
        b.append(v/np.linalg.norm(v))
    U,_,Vh=np.linalg.svd(np.array(b).T@np.array(a))
    D=np.diag([1,1,np.linalg.det(U@Vh)])
    return Matrix((U@D@Vh).tolist()).to_quaternion()

def position_at(t):
    ts=np.array([time_at(r) for r in pr])
    vs=np.array([positions[r] for r in pr])
    return Vector([float(np.interp(t,ts,vs[:,axis])) for axis in range(3)])

def fit_yaw(position, pairs, prior):
    offsets=[]
    for uv,xyz in pairs:
        x,y=np.linalg.solve(K,np.array([*uv,1]))[:2]
        ray=prior @ Vector((x,-y,-1))
        target=Vector(xyz)-position
        offsets.append(math.atan2(target.y,target.x)-math.atan2(ray.y,ray.x))
    offset=math.atan2(sum(math.sin(v) for v in offsets),sum(math.cos(v) for v in offsets))
    return Quaternion((0,0,1),offset) @ prior

orientation_times={time_at(r):rot for r,rot in rots.items()}
orientation_tie_report=[]
for knot in survey.get('orientation_ties',[]):
    index=knot['video_index']
    t=pts[index]
    k=int(np.searchsorted([time_at(r) for r in ar],t,side='right'))
    assert 0<k<len(ar) and t not in orientation_times
    left,right=ar[k-1:k+1]
    prior=rots[left].slerp(rots[right],(t-time_at(left))/(time_at(right)-time_at(left)))
    position=position_at(t)
    fitted=fit_yaw(position,knot['pairs'],prior) if knot['mode']=='yaw' else fit_rotation(position,knot['pairs'])
    orientation_times[t]=fitted
    residuals=[]
    for uv,xyz in knot['pairs']:
        local=np.array(fitted.inverted() @ (Vector(xyz)-position))*[1,-1,-1]
        pix=K@local
        residuals.append(float(np.linalg.norm(pix[:2]/pix[2]-uv)))
    orientation_tie_report.append(dict(**knot,video_pts_seconds=t,
          rotation_quaternion_wxyz=list(fitted),fit_landmark_residuals_px=residuals))
ordered=sorted(orientation_times)
cam=camera('TrajectoryCam')
cam.rotation_mode='QUATERNION'
cam['source_pose_matrix_count']=len(anchors)
cam['source_position_sample_count']=len(positions)
cam['video_mapping']='Normalized timestamped trajectory mapped to decoded-video PTS.'
scene.camera=cam
scene.frame_start=1
scene.frame_end=1+round(pts[-1]*scene.render.fps)
camera_keys=[]
prev=None
source_frames=[scene_frame(time_at(r)) for r in ar]
samples=sorted({*range(scene.frame_start,scene.frame_end+1),*[round(f,7) for f in source_frames]})
for f in samples:
    t=(f-1)/scene.render.fps
    j=int(np.searchsorted(ordered,t,side='right'))
    i=max(0,j-1)
    j=min(j,len(ordered)-1)
    left,right=ordered[i],ordered[j]
    alpha=(t-left)/(right-left) if right>left else 0
    loc=position_at(t)
    a,b=orientation_times[left],orientation_times[right]
    q=a.slerp(b,alpha)
    if prev is not None and q.dot(prev)<0:
        q.negate()
    prev=q.copy()
    cam.location=loc
    cam.rotation_quaternion=q
    cam.keyframe_insert(data_path='location',frame=f,group='Camera')
    cam.keyframe_insert(data_path='rotation_quaternion',frame=f,group='Camera')
    camera_keys.append((f,loc.copy(),q.copy()))
for curve in cam.animation_data.action.fcurves:
    for k in curve.keyframe_points:
        k.interpolation='LINEAR'

frame_info=[]
for row,(raw,world) in anchors.items():
    t=time_at(row)
    note=f"P{row:03} / video {pose_entries[row]['video_index']:03}"
    scene.timeline_markers.new(note,frame=round(scene_frame(t)))
    frame_info.append({'pose_row':row,'decoded_video_frame':pose_entries[row]['video_index'],
                       'video_pts_seconds':t,'timeline_frame':scene_frame(t),
                       'pi3_matrix_world':[list(r) for r in raw],
                       'room_matrix_world':[list(r) for r in world]})

cam110=camera('TwinCam')
anchor_row=survey.get('anchor_row',ar[len(ar)//2])
cam110.matrix_world=anchors[anchor_row][1]
cam110['pose_row']=anchor_row
cam110['decoded_video_frame']=pose_entries[anchor_row]['video_index']
cam110['role']='Exact source anchor after the same rigid leveling applied to the point cloud.'
cam110['source_matrix_world']=sum((list(row) for row in anchors[anchor_row][0]),[])

def load_cloud(path):
    with path.open('rb') as fh:
        hdr=[]
        while True:
            raw=fh.readline()
            if not raw:
                raise ValueError('Missing PLY header terminator.')
            hdr.append(raw.decode('ascii').strip())
            if hdr[-1]=='end_header':
                break
        expected=['property double '+a for a in ('x','y','z','nx','ny','nz')]+['property uchar '+a for a in ('red','green','blue')]
        if hdr[1]!='format binary_little_endian 1.0' or [x for x in hdr if x.startswith('property ')]!=expected:
            raise ValueError('PLY schema differs from the supplied source schema.')
        count=int(next(h.split()[-1] for h in hdr if h.startswith('element vertex ')))
        cloud=np.fromfile(fh,count=count,dtype=[('p','<f8',(3,)),('n','<f8',(3,)),('c','u1',(3,))])
        if len(cloud)!=count: raise ValueError('Incomplete PLY data.')
    return cloud

room=np.zeros((0,3))
if cloud_path:
    cloud=load_cloud(cloud_path)
    room=(cloud['p']-origin)@T.T
    pointmesh=bpy.data.meshes.new('Pi3_ReferenceVertices')
    pointmesh.from_pydata(room.tolist(),[],[])
    attr=pointmesh.color_attributes.new(name='Pi3_RGB',type='BYTE_COLOR',domain='POINT')
    colors=np.column_stack((cloud['c']/255,np.ones(len(cloud)))).astype(np.float32)
    attr.data.foreach_set('color_srgb', colors.ravel())
    ref=bpy.data.objects.new('Pi3_ColoredPointCloud',pointmesh)
    categories['Reference'].objects.link(ref)
    ref.hide_render=True
    categories['Reference'].hide_viewport=True

    # Give the loose reference vertices visible radii in material/rendered viewport modes.
    point_material=bpy.data.materials.new('Pi3 reference color')
    point_material.use_nodes=True
    point_shader=point_material.node_tree
    point_shader.nodes.remove(point_shader.nodes.get('Principled BSDF'))
    attr_node=point_shader.nodes.new('ShaderNodeVertexColor')
    attr_node.layer_name='Pi3_RGB'
    emit_node=point_shader.nodes.new('ShaderNodeEmission')
    point_shader.links.new(attr_node.outputs['Color'],emit_node.inputs['Color'])
    point_shader.links.new(emit_node.outputs[0],point_shader.nodes.get('Material Output').inputs['Surface'])
    points_tree=bpy.data.node_groups.new('Pi3 colored reference points','GeometryNodeTree')
    points_tree.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    points_tree.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    inp=points_tree.nodes.new('NodeGroupInput')
    outp=points_tree.nodes.new('NodeGroupOutput')
    to_points=points_tree.nodes.new('GeometryNodeMeshToPoints')
    to_points.inputs['Radius'].default_value=.021
    set_material=points_tree.nodes.new('GeometryNodeSetMaterial')
    set_material.inputs['Material'].default_value=point_material
    points_tree.links.new(inp.outputs['Geometry'],to_points.inputs['Mesh'])
    points_tree.links.new(to_points.outputs['Points'],set_material.inputs['Geometry'])
    points_tree.links.new(set_material.outputs['Geometry'],outp.inputs['Geometry'])
    ref.modifiers.new('Plot RGB samples','NODES').node_group=points_tree

    # Duplicate source space as an independent audit scene with the literal supplied matrix.
    raw_scene=bpy.data.scenes.new('Pi3 audit | original world')
    raw_scene.unit_settings.system='METRIC'
    raw_scene.render.resolution_x=W
    raw_scene.render.resolution_y=H
    raw_scene.render.pixel_aspect_y=scene.render.pixel_aspect_y
    raw_scene.render.pixel_aspect_x=scene.render.pixel_aspect_x
    raw_reference=bpy.data.objects.new('Pi3_PointCloud_Unleveled',pointmesh)
    raw_reference.matrix_world=LEVEL.inverted()
    raw_scene.collection.objects.link(raw_reference)
    raw_reference.modifiers.new('Plot RGB samples','NODES').node_group=points_tree
    raw_cam=bpy.data.objects.new('TwinCam_Pi3_original',cam110.data)
    raw_scene.collection.objects.link(raw_cam)
    raw_cam.matrix_world=anchors[anchor_row][0]
    raw_scene.camera=raw_cam

# Optional time-synchronised reference footage in a separate scene, available through the Sequencer.
(OUT/'Reference').mkdir(exist_ok=True)
clip_copy=OUT/'Reference'/'source_video.mp4'
clip_copy.write_bytes(video_path.read_bytes())
video_scene=bpy.data.scenes.new('Video comparison | source')
video_scene.render.fps=25
video_scene.render.resolution_x=W
video_scene.render.resolution_y=H
video_scene.render.resolution_percentage=100
video_scene.frame_end=scene.frame_end
ed=video_scene.sequence_editor_create()
strips=ed.strips if hasattr(ed,'strips') else ed.sequences
strip=strips.new_movie('Source footage',str(clip_copy),1,1)
strip.filepath='//Reference/source_video.mp4'

# Overview camera and viewport starting position.
overview=camera('OverviewCam')
overview.location=(-13.2,-15.2,15.8)
overview.rotation_euler=(Vector((.2,-6.05,.6))-overview.location).to_track_quat('-Z','Y').to_euler()
overview.data.lens=23
overview.hide_render=True
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=='VIEW_3D':
            area.spaces.active.region_3d.view_perspective='CAMERA'
            area.spaces.active.clip_end=200

scene.frame_set(round(scene_frame(pts[target])))
bpy.context.view_layer.update()
video_scene.frame_set(round(scene_frame(pts[target])))
asset_list=[]
for key,(collection,cat,note) in ASSETS.items():
    asset_list.append(dict(asset_id=key,category=cat,asset_class=key,source='AssetLibrary.blend',description=note,
                           part_count=len(collection.objects)))
assets_by_name={a['asset_id']:a for a in asset_list}

# Geometry-level duplicate suppression.  Duplicates that survived the language
# planner still have identical prototype keys and nearly identical transforms.
def prune_duplicate_instances():
    kept=[]
    for ob in list(INSTANCES):
        duplicate=None
        for other in kept:
            if ob['asset_id']!=other['asset_id']:
                continue
            if (ob.location-other.location).length < .38:
                duplicate=other
                break
        if duplicate is None:
            kept.append(ob)
        else:
            bpy.data.objects.remove(ob,do_unlink=True)
    INSTANCES[:]=kept
prune_duplicate_instances()

def instance_aabb(ob):
    collection=ob.instance_collection
    pts=[]
    if collection:
        for child in collection.objects:
            if child.type!='MESH':
                continue
            for corner in child.bound_box:
                pts.append(ob.matrix_world @ child.matrix_world @ Vector(corner))
    if not pts:
        return None
    pts=np.array([[p.x,p.y,p.z] for p in pts])
    return pts.min(axis=0),pts.max(axis=0)

def box_volume(lo,hi):
    d=np.maximum(0,hi-lo)
    return float(np.prod(d))

records=[]; boxes=[]
for ob in INSTANCES:
    bb=instance_aabb(ob)
    if bb is None: continue
    lo,hi=bb
    rec=dict(instance_id=ob.name,asset_id=ob['asset_id'],category=ob['asset_category'],
             min_m=lo.round(3).tolist(),max_m=hi.round(3).tolist(),
             volume_m3=round(box_volume(lo,hi),4))
    records.append(rec); boxes.append((ob,lo,hi))
overlaps=[]
for i,(a,alo,ahi) in enumerate(boxes):
    if a['asset_category'] in {'Architecture','Lighting','Cameras'}: continue
    for b,blo,bhi in boxes[i+1:]:
        if b['asset_category'] in {'Architecture','Lighting','Cameras'}: continue
        inter=np.maximum(0,np.minimum(ahi,bhi)-np.maximum(alo,blo))
        vol=float(np.prod(inter))
        if vol>.012:
            overlaps.append(dict(a=a.name,b=b.name,overlap_m3=round(vol,4),
                                 overlap_fraction=round(vol/max(.001,min(box_volume(alo,ahi),box_volume(blo,bhi))),3)))
ducks=[]
for i,(a,alo,ahi) in enumerate(boxes):
    for b,blo,bhi in boxes[i+1:]:
        if a['asset_id']!=b['asset_id']: continue
        c1=(alo+ahi)/2; c2=(blo+bhi)/2
        if np.linalg.norm(c1-c2)<.38 and np.prod(ahi-alo)>.01:
            ducks.append([a.name,b.name])
window_bad=[]
for ob in INSTANCES:
    if ob['asset_id']!='WindowBay': continue
    if ob.scale[0]<.34 or ob.scale[2]<.45:
        window_bad.append(dict(instance=ob.name,scale=list(ob.scale)))
validation=dict(status='review' if overlaps or ducks or window_bad else 'pass',
                counts=dict(instances=len(INSTANCES),overlap_pairs=len(overlaps),duplicate_pairs=len(ducks),
                            invalid_windows=len(window_bad)),
                overlaps=overlaps[:120],duplicates=ducks[:120],invalid_windows=window_bad,
                instances=records)
(OUT/'AssemblyValidation.json').write_text(json.dumps(validation,indent=2))

manifest=[]
for ob in INSTANCES:
    manifest.append(dict(instance_id=ob.name,asset_id=ob['asset_id'],category=ob['asset_category'],
          asset_class=ob['asset_id'],source='AssetLibrary.blend',usd_reference=f'AssetPrototypes.usda</Prototypes/{ob["asset_id"]}>',
          placement_evidence=ob['placement_evidence'],translation_m=list(ob.location),
          rotation_quaternion_wxyz=list(ob.matrix_world.to_quaternion()),scale=list(ob.scale),
          matrix_world_row_major=[list(row) for row in ob.matrix_world]))
(OUT/'AssetManifest.json').write_text(json.dumps({'units':'m','up_axis':'Z','assets':asset_list,'instances':manifest},indent=2))
with (OUT/'VideoFrameMap.csv').open('w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(['decoded_video_index','pts_seconds','blender_frame','translation_source','rotation_source'])
    pos_frames={r['video_index'] for r in pos_entries.values()}
    full_frames={r['video_index'] for r in pose_entries.values()}
    tie_frames={t['video_index']:t['mode'] for t in orientation_tie_report}
    for video_index,t in enumerate(pts):
        position_type='provided' if video_index in pos_frames else ('terminal_hold' if t>time_at(pr[-1]) else 'interpolated')
        rotation_type=('provided_matrix' if video_index in full_frames else
                       ('registered_'+tie_frames[video_index] if video_index in tie_frames else
                        ('terminal_hold' if t>time_at(ar[-1]) else 'slerp')))
        writer.writerow([video_index,t,round(scene_frame(t)),position_type,rotation_type])
with (OUT/'AssetInstances.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.writer(f)
    writer.writerow(['instance_id','asset_id','category','asset_class','library_source','x_m','y_m','z_m',
                     'qw','qx','qy','qz','sx','sy','sz','placement_evidence'])
    for row in manifest:
        writer.writerow([row[k] for k in ('instance_id','asset_id','category','asset_class','source')]+row['translation_m']
                        +row['rotation_quaternion_wxyz']+row['scale']+[row['placement_evidence']])

def projection_stats():
    from bpy_extras.object_utils import world_to_camera_view
    obs=np.array([[.1,.15,-2],[-.7,.35,-3],[.8,-.31,-3],[0,0,-2],[1,.2,-3]])
    samples=[]
    for p in obs:
        pix=(K@(p*[1,-1,-1]))
        expected=pix[:2]/pix[2]
        got=world_to_camera_view(scene,cam110,cam110.matrix_world@Vector(p))
        samples.append(np.linalg.norm(np.array([W*got.x,H*(1-got.y)])-expected))
    return float(max(samples))

qa_floor=(abs(room[:,2])<.075)&(room[:,0]>-4)&(room[:,0]<3)&(room[:,1]>-4)&(room[:,1]<3)
# The band contains fitted-floor samples and is not an independent survey accuracy measurement.
err=projection_stats()
assert err<.003, err
recovered=LEVEL.inverted()@cam110.matrix_world
matrix_error=max(abs(t-s) for a,b in zip(recovered,anchors[anchor_row][0]) for t,s in zip(a,b))
assert matrix_error<1e-5
frames_error={}
for row,(_,world) in anchors.items():
    f=scene_frame(time_at(row))
    scene.frame_set(math.floor(f),subframe=f%1)
    bpy.context.view_layer.update()
    angle=math.degrees(cam.rotation_quaternion.rotation_difference(rots[row]).angle)
    frames_error[row]=dict(translation_m=(cam.location-world.translation).length,
                           angular_deg=min(angle,abs(360-angle)))
worst_t=max(frames_error.values(), key=lambda p: p['translation_m'])
worst_a=max(frames_error.values(), key=lambda p: p['angular_deg'])
print(f"ANCHOR playback max_t={worst_t['translation_m']:.6f}m max_a={worst_a['angular_deg']:.4f}deg", flush=True)
# Evaluated fcurves in Blender 4.5 sit about 0.06 mm off a fractional frame.
assert worst_t['translation_m'] < 2e-4
assert worst_a['angular_deg'] < .05
scene.frame_set(round(scene_frame(pts[target])))

log=dict(inputs=[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in (video_path,cloud_path,pose_path,intr_path) if p is not None],
         point_count=len(room), source_to_room=[list(r) for r in LEVEL],
         raw_z_up_mapping='Rigid: p_room = R @ (p_pi3 - origin). Applied to camera and cloud together.',
         frame_mapping='Normalized camera times and MP4 presentation timestamps.',
         video_frames=len(pts), fps=scene.render.fps, terminal_pts_s=pts[-1],
         supplied_declared_pose_count=track['n_poses'], unique_full_matrix_count=len(anchors),
         position_sample_count=len(positions), observed_video_orientation_ties=orientation_tie_report,
         focal_mm=cam.data.lens, intrinsic_matrix=K.tolist(), resolution=[W,H],
         blender_shift=[cam.data.shift_x,cam.data.shift_y], pixel_aspect_y=scene.render.pixel_aspect_y,
         pixel_aspect_x=scene.render.pixel_aspect_x,
         shift_validation_max_px=err, twin_cam_source_matrix_round_trip_max_abs=matrix_error,
         anchor_playback_errors=frames_error, floor_band_point_count=int(qa_floor.sum()),
         floor_band_abs_height_percentiles_m=np.percentile(abs(room[qa_floor,2]),[50,90,95]).tolist() if qa_floor.any() else [],
         anchor_frames=frame_info, target_video_index=target, target_timeline_frame=round(scene_frame(pts[target])),
         path_length_m=float(sum((b[1]-a[1]).length for a,b in zip(camera_keys,camera_keys[1:]))),
         supplied_path_length_m=track['path_length_m'],survey_notes=survey['notes'],
         usd_export_method='USDA serialized from Blender evaluated geometry and camera data')
(OUT/'AlignmentLog.json').write_text(json.dumps(log,indent=2))

bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'IndoorTwin.blend'),compress=True,relative_remap=False)
bpy.data.libraries.write(str(OUT/'AssetLibrary.blend'),{a[0] for a in ASSETS.values()},
                         path_remap='NONE',fake_user=True,compress=True)
write_usd(OUT,scene,ASSETS,INSTANCES,MATS,cam,cam110,camera_keys,list(categories['Lighting'].objects))
import shutil
shutil.copyfile(OUT/'IndoorTwin.usda',OUT/'Scene.usda')

if args.preview:
    preview_dir=OUT/'Previews'
    preview_dir.mkdir(exist_ok=True)
    scene.render.resolution_percentage=78
    for index in survey['preview_indices']:
        scene.frame_set(round(scene_frame(pts[index])))
        scene.render.filepath=str(preview_dir/f'video_{index:03}.png')
        bpy.ops.render.render(write_still=True)
    # Birds-eye documentation camera. Hide the roof only for this view.
    scene.camera=overview
    for obj in INSTANCES:
        if obj['asset_id'] in {'CeilingSlab','CeilingBeam','StripLuminaire'} or obj.name == 'WestWallInferred':
            obj.hide_render=True
    scene.render.filepath=str(preview_dir/'overview.png')
    bpy.ops.render.render(write_still=True)
