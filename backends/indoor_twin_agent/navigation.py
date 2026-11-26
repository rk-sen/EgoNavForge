"""Audit rigid USD bounds, room enclosure and conservative navigation connectivity."""
from pathlib import Path
import math
import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
from pxr import Usd,UsdGeom,UsdPhysics
from evidence import put_json,read_json
from layout import path_samples


def physical_bounds(scene_path):
    stage=Usd.Stage.Open(str(scene_path))
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.guide])
    shapes=[]
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI): continue
        bbox=cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if bbox.IsEmpty(): continue
        root=prim.GetParent().GetParent()
        name=root.GetAttribute('twin:instanceId').Get()
        shapes.append((np.asarray(bbox.GetMin()),np.asarray(bbox.GetMax()),name,str(root.GetPath())))
    if not shapes: raise ValueError('USD does not contain resolvable collisions.')
    return shapes


def aabb_hits(points,shapes,eps=0):
    out=np.zeros(len(points),dtype=int)
    for lo,hi,*_ in shapes:
        out+=np.all((points>=lo-eps)&(points<=hi+eps),axis=1)
    return out


def graph_links(layout,labels,x,y,radius):
    result=[]
    def lookup(q):
        i=int(np.argmin(abs(x-q[0])));j=int(np.argmin(abs(y-q[1])))
        return int(labels[j,i])
    for wall in layout['walls']:
        for a in wall['apertures']:
            if a['type']!='connected_door': continue
            axis=wall['axis']
            w=a['span_m'][1]-a['span_m'][0]
            c=np.zeros(2);c[axis]=wall['offset_m'];c[1-axis]=np.mean(a['span_m'])
            shift=np.zeros(2);shift[axis]=radius+.24
            left,right=lookup(c-shift),lookup(c+shift)
            validated=left>0 and left==right and len(a['room_ids'])==2
            result.append(dict(wall_id=wall['id'],room_ids=a['room_ids'],width_m=w,center_m=c.tolist(),
                               connected_in_grid=validated))
    return result


def audit(job,ev,radius=.27):
    job=Path(job)
    layout=read_json(job/'Layout.json')
    manifest=read_json(job/'Scene'/'SimulationManifest.json')
    shapes=physical_bounds(job/'Scene'/'Scene.usda')
    instances={i['instance_id']:i for i in manifest['instances']}
    context={i['id'] for i in layout['inferred_context']}
    lo,hi=np.asarray(layout['bounds_m'])
    stride=float(max(.08,math.sqrt(np.prod(hi-lo)/90000)))
    x=np.arange(lo[0]+stride/2,hi[0],stride)
    y=np.arange(lo[1]+stride/2,hi[1],stride)
    X,Y=np.meshgrid(x,y)
    walk=(X>lo[0]+radius)&(X<hi[0]-radius)&(Y>lo[1]+radius)&(Y<hi[1]-radius)
    baseline=walk.copy()
    added=np.zeros_like(walk)
    body_floor=.05;body_height=1.25
    for a,b,n,_ in shapes:
        if b[2]<body_floor or a[2]>body_height:
            continue
        # Exact Euclidean dilation of an axis-aligned world AABB, deliberately conservative for rotated props.
        dx=np.maximum(np.maximum(a[0]-X,0),X-b[0])
        dy=np.maximum(np.maximum(a[1]-Y,0),Y-b[1])
        blocker=dx*dx+dy*dy<=radius*radius
        if n not in context: baseline&=~blocker
        else: added|=blocker
        walk&=~blocker
    labels,n=label(walk)
    raw,n_raw=label(baseline)
    main=np.bincount(labels.ravel())
    top=int(np.max(main[1:])) if len(main)>1 else 0
    baseline_counts=np.bincount(raw.ravel())
    base_top=int(np.max(baseline_counts[1:])) if len(baseline_counts)>1 else 0
    clearance=distance_transform_edt(walk,sampling=stride)
    rooms=[]
    for row in layout['rooms']:
        a=np.array(row['low_m']);b=np.array(row['high_m'])
        keep=(X>=a[0])&(X<=b[0])&(Y>=a[1])&(Y<=b[1])&walk
        if not keep.any():
            rooms.append(dict(id=row['id'],walkable_cells=0,waypoint_m=None))
            continue
        ij=np.unravel_index(np.argmax(clearance*keep),keep.shape)
        rooms.append(dict(id=row['id'],walkable_cells=int(keep.sum()),waypoint_m=[float(X[ij]),float(Y[ij]),0.],
                          component=int(labels[ij]),margin_after_robot_radius_m=float(clearance[ij])))
    links=graph_links(layout,labels,x,y,radius)
    trajectory=path_samples(ev,step=.045)
    arch=[s for s in shapes if instances.get(s[2],{}).get('category')=='Architecture']
    camera_arch=aabb_hits(trajectory,arch)
    camera_context=aabb_hits(trajectory,[s for s in shapes if s[2] in context])
    camera_all=aabb_hits(trajectory,shapes)
    raw_prepare=cKDTree(np.column_stack([X[baseline],Y[baseline]])) if baseline.any() else None
    prep=cKDTree(np.column_stack([X[walk],Y[walk]])) if walk.any() else None
    def track_portion(kdt):
        if kdt is None: return 0.
        distance,_=kdt.query(trajectory[:,:2])
        return float(np.mean(distance<stride*1.5))
    # Probe the exterior physical shell every ~6 cm, all the way through the occupied height.
    closes=[]
    outer=[s for s in arch if s[2] and s[2].startswith('Exterior_')]
    for w in layout['walls']:
        if not w['exterior']: continue
        axis=w['axis']
        t=np.arange(*w['span_m'],.065)
        z=np.arange(.09,layout['wall_height_m']-.04,.095)
        tt,zz=np.meshgrid(t,z)
        positions=np.zeros((tt.size,3))
        positions[:,axis]=w['offset_m']
        positions[:,1-axis]=tt.ravel()
        positions[:,2]=zz.ravel()
        own=[s for s in outer if s[2].startswith(w['id']+'_')]
        hit=aabb_hits(positions,own,.013)
        closes.append(dict(wall=w['id'],probe_count=len(positions),outside_leaks=int((hit==0).sum())))
    # Check the floor and roof slabs separately at an interior raster.
    XY=np.array(np.meshgrid(np.linspace(lo[0]+.075,hi[0]-.075,31),
                            np.linspace(lo[1]+.075,hi[1]-.075,31))).reshape(2,-1).T
    slabs=[]
    for kind,height in [('Floor',-.041),('Ceiling',layout['wall_height_m']+.04)]:
        sample=np.column_stack([XY,np.full(len(XY),height)])
        own=[s for s in arch if s[2] and s[2].startswith(kind)]
        slab_hit=aabb_hits(sample,own,.007)
        slabs.append(dict(slab=kind,probe_count=len(sample),misses=int((slab_hit==0).sum())))
    graph=[]
    use_path=cKDTree(np.column_stack([X[walk],Y[walk]])) if walk.any() else None
    for row in manifest['instances']:
        key=row['instance_id']
        c=row['category']
        prov=row.get('placement_evidence','')
        if c not in {'Furniture','Equipment'} or key in context or prov in {'visual_recipe_estimate','tabletop_support_estimate'}:
            continue
        at=np.asarray(row['translation_m'])
        if use_path is None: continue
        distance,ix=use_path.query(at[:2])
        if distance>.90: continue
        cell=np.column_stack([X[walk],Y[walk]])[ix]
        graph.append(dict(instance_id=key,asset_id=row['asset_id'],world_position_m=at.tolist(),
                          candidate_goal_m=[float(cell[0]),float(cell[1]),0.],
                          provenance=prov,landmark_status='candidate_requires_language_validation'))
    diagnostic=dict(robot_radius_m=radius,body_height_m=body_height,grid_resolution_m=stride,
        grid_cells=int(walk.size),walkable_area_m2=float(walk.sum()*stride**2),
        baseline_without_context_area_m2=float(baseline.sum()*stride**2),
        largest_component_area_m2=float(top*stride**2),baseline_largest_component_area_m2=float(base_top*stride**2),
        components=int(n),components_without_context=int(n_raw),route_handheld_floor_projection_free_ratio=track_portion(prep),
        route_without_context_free_ratio=track_portion(raw_prepare),
        exterior_collision_samples=sum(w['probe_count'] for w in closes),
        exterior_collision_misses=sum(w['outside_leaks'] for w in closes),
        exterior_probes_by_wall=closes,
        slab_collision_samples=sum(x['probe_count'] for x in slabs),
        slab_collision_misses=sum(x['misses'] for x in slabs),slab_probes=slabs,
        sampled_camera_centers=len(trajectory),camera_center_inside_structure=int((camera_arch>0).sum()),
        camera_center_inside_any_collider=int((camera_all>0).sum()),camera_center_inside_context=int((camera_context>0).sum()),
        context_objects=len(context),rooms=rooms,portals=links,candidate_landmarks=graph)
    put_json(job/'Navigation.json',diagnostic)
    np.savez_compressed(job/'NavigationGrid.npz',x_m=x,y_m=y,walkable=walk,component=labels,
                        walkable_without_context=baseline,clearance_m=clearance)
    show=np.zeros((*walk.shape,3),np.uint8)
    show[:]=[42,58,68]
    show[baseline]=[163,184,176]
    show[walk]=[77,162,137]
    show[baseline&~walk]=[221,174,93]
    pic=Image.fromarray(np.flip(show,axis=0)).resize((740,max(208,round(740*len(y)/len(x)))),Image.Resampling.NEAREST)
    canvas=Image.new('RGB',(pic.width,pic.height+74),'#14242a')
    canvas.paste(pic,(0,47))
    d=ImageDraw.Draw(canvas)
    d.text((13,16),'NAVIGATION AUDIT   |   teal: free   gold: occupied by inferred context',fill='#eaf5ee')
    d.text((13,pic.height+53),'Red: recorded camera XY (handheld). White: room waypoint.',fill='#eaf5ee')
    def pixel(p):
        u=(np.asarray(p)[:2]-lo)/(hi-lo)
        return float(u[0])*pic.width,47+(1-float(u[1]))*pic.height
    d.line([pixel(p) for p in trajectory],fill='#f77b72',width=3)
    for room in rooms:
        if room['waypoint_m'] is None: continue
        u,v=pixel(room['waypoint_m'])
        d.ellipse((u-5,v-5,u+5,v+5),fill='white')
        d.text((u+5,v+5),room['id'],fill='white')
    canvas.save(job/'NavigationMap.jpg',quality=94)
    return diagnostic
