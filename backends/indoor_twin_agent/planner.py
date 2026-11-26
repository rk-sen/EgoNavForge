"""Convert multimodal observations into a metric, editable scene plan."""
import hashlib
import json
import math
import re
import numpy as np
from PIL import Image
from evidence import put_json, ray_pose

CATS={
 'table':'table','wooden_table':'table','bench':'bench','coffee_table':'table','desk':'desk',
 'office_desk':'desk','workstation':'desk','sofa':'sofa','couch':'sofa','sectional':'sofa',
 'bean_bag':'beanbag','beanbag':'beanbag','exercise_ball':'ball','ball':'ball',
 'office_chair':'chair','chair':'chair','bookshelf':'shelf','storage_shelf':'shelf','shelving':'shelf',
 'partition':'partition','partition_wall':'partition','barrier':'barrier','padded_barrier':'barrier','window':'window','doorway':'door',
 'door':'door','monitor':'monitor','computer_monitor':'monitor','computer_tower':'computer',
 'ceiling_light':'light','light':'light','person':'person'}
MEAS={
 'table':[1.40,.58,.65],'bench':[1.72,.44,.58],'desk':[1.46,.68,.84],'sofa':[1.94,.83,.79],
 'beanbag':[.87,.80,.51],'ball':[.58,.58,.58],'chair':[.55,.60,1.0],'shelf':[.82,.37,1.65],
 'barrier':[1.65,.10,1.08],'partition':[1.72,.15,1.52],'window':[1.42,.13,1.82],'door':[.93,.15,2.14],
 'monitor':[.56,.12,.43],'person':[.47,.28,1.67],'computer':[.24,.50,.53]}
STOCK={
 'WallSegment':('wall','Architecture',[1,1,1]),'CorridorWall':('wall','Architecture',[1,1,1]),
 'Column':('column','Architecture',[1,1,1]),'CeilingSlab':('ceiling','Architecture',[1,1,1]),
 'CeilingBeam':('beam','Architecture',[1,1,1]),'LabRaisedFloor':('floor','Architecture',[18.3,21.8,.14]),
 'WindowBay':('window','Architecture',[1.49,.20,2.33]),'DoorPortal':('door','Architecture',[.19,1.44,2.66]),
 'FabricBarrier':('barrier','Furniture',[1.62,.32,1.2]),'ExerciseBall':('ball','Furniture',[.684]*3),
 'Beanbag':('beanbag','Furniture',[1.15,.95,.64]),'TerracottaSofa':('sofa','Furniture',[1.02,1.97,.82]),
 'LooseThrowPillow':('pillow','Furniture',[.33,.47,.146]),'TrefoilWoodBench':('bench','Furniture',[.445,1.98,.70]),
 'OfficeDesk':('desk','Furniture',[1.74,.75,1]),'FlatMonitor':('monitor','Equipment',[.64,.19,.47]),
 'ComputerTower':('computer','Equipment',[.34,.57,.67]),'RollingChair':('chair','Furniture',[.58,.57,1.01]),
 'DrapedJacket':('garment','Furniture',[.63,.14,.58]),'OfficeShelf':('shelf','Furniture',[1.4,.51,2.04]),
 'StorageBox':('carton','Equipment',[.40,.33,.30]),'MobileEquipmentRig':('cart','Equipment',[.50,.45,.50]),
 'StandingPersonProxy':('person','Equipment',[.53,.28,1.83]),'WoodPartition':('partition','Furniture',[1.72,.15,1.52]),'StripLuminaire':('light','Lighting',[1,.105,.079])}
TEMPLATES={
 'sofa':('TerracottaSofa','Terracotta'),'ball':('ExerciseBall','Rubber_green'),
 'beanbag':('Beanbag','Beanbag_charcoal'),'bench':('TrefoilWoodBench','Walnut'),
 'table':('TrefoilWoodBench','Walnut'),
 'chair':('RollingChair','Chair_dark'),'shelf':('OfficeShelf','Walnut'),
 'barrier':('FabricBarrier','Barrier_indigo'),'partition':('WoodPartition','Walnut'),'monitor':('FlatMonitor','Screen'),
 'computer':('ComputerTower','Trim_dark'),'person':('StandingPersonProxy','Chair_dark')}
BASE_FINISH={
 'TerracottaSofa':([.43,.225,.142],'leather'),'ExerciseBall':([.045,.46,.255],'rubber'),
 'Beanbag':([.20,.192,.20],'cloth'),'TrefoilWoodBench':([.25,.144,.081],'wood'),
 'OfficeDesk':([.62,.62,.58],'paint'),'RollingChair':([.107,.11,.12],'cloth'),
 'OfficeShelf':([.25,.144,.081],'wood'),'FabricBarrier':([.05,.036,.259],'cloth'),
 'FlatMonitor':([.052,.078,.10],'paint'),'ComputerTower':([.026,.029,.034],'paint'),
 'StandingPersonProxy':([.082,.084,.09],'cloth'),'WoodPartition':([.25,.144,.081],'wood')}
STOCK_DESC={
 'TerracottaSofa':'low upholstered sofa with two rounded full arms, single seat cushion, backrest and plinth',
 'TrefoilWoodBench':'elongated walnut bench, 4 splayed legs, two transverse bars and a long stretcher',
 'RollingChair':'office chair, five casters, padded seat and back, two armrests',
 'FabricBarrier':'portable fabric partition, tubular rails and two red feet',
 'OfficeDesk':'rectangular worktop, four square legs and modesty panel',
 'ExerciseBall':'single spherical rubber exercise ball with equatorial ribs'}
TABLETOP={'monitor','laptop','keyboard','book','bottle','mug','phone','small_device','toy','globe','notebook'}
DEFAULT_MAT=dict(color=[.50,.47,.40],texture='paint',roughness=.65,metallic=0.)


def ident(value):
    s=re.sub('[^a-zA-Z0-9_]+','_',str(value)).strip('_')[:72]
    return s if s and not s[0].isdigit() else 'Obj_'+s


def label(x):
    k=ident(str(x).lower())
    return CATS.get(k,k)


def semantic(obj):
    k=label(obj.get('category','object'))
    evidence=(obj.get('id','')+' '+obj.get('description','')).lower()
    if k in {'person','human','people','standing_person','man','woman'}:return 'person'
    if 'teddy' in evidence or 'stuffed bear' in evidence: return 'teddy_bear'
    if k in {'cardboard_box','storage_box','carton'}: return 'box'
    if k in {'trashcan','waste_bin','trash_bin','wastebasket'}: return 'bin'
    if k in {'window_blinds','curtain_panel','window_curtain'}: return 'curtain'
    if k in {'potted_plant','houseplant'}: return 'plant'
    if k in {'wall_art','picture_frame','artwork'}: return 'art'
    if k in {'light_fixture','fluorescent_light'}: return 'light'
    if k in {'basketball','football','soccer_ball'}: return 'sport_ball'
    if k in {'furniture','equipment','prop','decorative_object'}:
        stem=re.split(r'\b(near|behind|under|next to|on top of|on the|on a|with|before|against)\b',
                      obj.get('description','').lower(),maxsplit=1)[0]
        label_text=obj.get('id','').replace('_',' ').lower()+' '+stem
        aliases=[('exercise ball','ball'),('stability ball','ball'),('bean bag','beanbag'),
                 ('padded barrier','barrier'),('partition','barrier'),('sofa','sofa'),
                 ('shelving','shelf'),('shelf','shelf'),('potted plant','plant'),('bench','bench'),
                 ('monitor','monitor'),('cabinet','cabinet'),('desk','desk')]
        for phrase,name in aliases:
            if phrase in label_text: return name
    if 'laptop' in evidence: return 'laptop'
    if 'keyboard' in evidence: return 'keyboard'
    if 'globe' in evidence: return 'globe'
    if k in {'display','screen'} and 'monitor' in evidence: return 'monitor'
    if 'bean bag' in evidence or 'beanbag' in evidence:
        return 'beanbag'
    if 'exercise ball' in evidence or 'inflatable ball' in evidence:
        return 'ball'
    if k in {'partition_wall','padded_barrier'}:
        return 'partition' if k=='partition_wall' else 'barrier'
    if k=='structural' and 'column' in evidence:
        return 'column'
    return k


def choices(frames,n):
    if len(frames)<=n: return frames
    return [frames[i] for i in np.round(np.linspace(0,len(frames)-1,n)).astype(int)]


def observe(api,e,job,seed=None):
    frames=sorted(e['photos'])
    results=[]
    for start in range(0,len(frames),1):
        group=frames[start:start+1]
        data={'width_px':e['W'],'height_px':e['H'],'video_duration_s':float(e['pts'][-1]),
              'calibration':e['K'].tolist(),'world':'meters, right-handed Z up',
              'frames':[{ 'index':i,'time_s':float(e['pts'][i]),
                          'camera_center_m':ray_pose(e['track'],e['pts'][i],e['R'],e['origin'])[0].round(4).tolist()} for i in group],
              'point_cloud_available':e['xyz'] is not None}
        result=api.ask('observe',data,[(f'Original video decoded frame {i}',e['photos'][i]) for i in group],6200)
        results.append(result)
    put_json(job/'Observations.json',results)
    return [obj for record in results for obj in record.get('objects',[])]


def infer_finish(obj):
    m=obj.get('material') or {}
    c=np.asarray(m.get('color',DEFAULT_MAT['color']),float)
    if c.shape!=(3,) or not np.isfinite(c).all(): c=DEFAULT_MAT['color']
    texture=m.get('texture','paint')
    return dict(color=np.clip(c,0,1).tolist(),texture=texture if texture in {'paint','wood','cloth','leather','rubber','metal'} else 'paint',
                roughness=float(np.clip(m.get('roughness',.65),.05,1)),
                metallic=float(np.clip(m.get('metallic',.0),0,1)))


def infer_dimensions(obj,k):
    d=np.asarray(obj.get('dimensions_m',[]),float)
    defaults={'laptop':[.31,.26,.22],'keyboard':[.43,.15,.05],'book':[.21,.28,.07],
              'bottle':[.085,.085,.24],'mug':[.11,.09,.12],'phone':[.10,.17,.045],'toy':[.18,.16,.28],
              'globe':[.24,.24,.26],'plant':[.36,.36,.80],'notebook':[.21,.28,.043]}
    prior=np.array(MEAS.get(k,defaults.get(k,[.52,.48,.61])))
    if d.shape!=(3,) or not np.isfinite(d).all(): return prior
    # Preserve the low thickness of supported objects such as keyboards.
    return np.clip(np.where((d>=.023)&(d<12),d,prior),.028,6)


def lift_ground(obj,ev,k,dims):
    centers=[]
    inv=np.linalg.inv(ev['K'])
    p=ev['xyz']
    supported=k in TABLETOP or obj.get('support')=='tabletop'
    for observation in obj.get('views',[]):
        idx=observation.get('index',-1)
        if idx not in ev['photos']: continue
        pos,rot=ray_pose(ev['track'],ev['pts'][idx],ev['R'],ev['origin'])
        gt=observation.get('ground')
        bbox=observation.get('bbox')
        if p is not None and bbox is not None and len(bbox)==4:
            # Vote on cloud points with image support and upper/lower vertical bounds.
            crop=np.asarray(bbox,float)
            if np.isfinite(crop).all():
                rel=(p-pos)@rot * [1,-1,-1]
                pixels=rel@ev['K'].T
                fx=pixels[:,0]/np.maximum(.03,pixels[:,2])/ev['W']
                fy=pixels[:,1]/np.maximum(.03,pixels[:,2])/ev['H']
                mx,my=(crop[:2]+crop[2:])*.5
                bx,by=(crop[2:]-crop[:2])*.33
                zmin=.6 if k=='window' else max(.045,dims[2]*.20)
                zmax=3.9 if k=='window' else dims[2]*1.45
                if supported:
                    zmin,zmax=.49,1.75
                hits=(rel[:,2]>.2)&(rel[:,2]<12)&(abs(fx-mx)<bx)&(abs(fy-my)<by)&(p[:,2]>zmin)&(p[:,2]<zmax)
                if hits.sum()>7:
                    z=p[hits]
                    sample=z[rel[hits,2]<np.quantile(rel[hits,2],.60)]
                    base=np.median(sample,axis=0)
                    base[2]=(max(.55,float(base[2]-dims[2]/2)) if k in {'window','monitor'} or supported else 0)
                    centers.append((2,base,idx))
        if not observation.get('ground_visible',False) or gt is None or len(gt)!=2: continue
        xy=np.array(gt,float)
        if not np.isfinite(xy).all() or np.any((xy<0)|(xy>1)): continue
        ray=rot@((inv@np.r_[xy*[ev['W'],ev['H']],1])*[1,-1,-1])
        if ray[2]>-.035: continue
        support_z=.76 if supported else 0.
        scale=(support_z-pos[2])/ray[2]
        if .12<scale<12:
            floor=pos+ray*scale
            floor[2]=support_z
            horizontal=ray.copy(); horizontal[2]=0
            floor+=horizontal/max(.001,np.linalg.norm(horizontal))*min(dims[:2])*.20
            centers.append((1,floor,idx))
    if not centers: return None
    all_p=np.array([r[1] for r in centers])
    dd=np.linalg.norm(all_p[:,:,None]-all_p.T[None,:,:],axis=1)
    weights=np.array([r[0] for r in centers])
    votes=((dd<max(.47,min(dims[:2])*.82))*weights).sum(axis=1)
    best=int(np.argmax(votes))
    keep=dd[best]<max(.47,min(dims[:2])*.82)
    xyz=np.average(all_p[keep],axis=0,weights=weights[keep])
    views=sorted({r[2] for r,v in zip(centers,keep) if v})
    pos,rot=ray_pose(ev['track'],ev['pts'][views[0]],ev['R'],ev['origin'])
    ray=pos-xyz
    if k in {'sofa','table','bench','desk','chair','shelf','barrier','window','door'}:
        # Building axis snap is a proposal, independent from the library asset itself.
        yaw=round((math.degrees(math.atan2(ray[1],ray[0]))+180)/90)*90
    else: yaw=0
    return xyz,float(yaw%360),views


def compact_recipe(x,dims):
    if not isinstance(x,list) or not 1<=len(x)<=14:
        raise ValueError('Procedural recipe must have 1–14 parts.')
    result=[]
    for j,p in enumerate(x):
        v={k:np.asarray(p[k],float) for k in ('center','size','rotation_deg')}
        if any(a.shape!=(3,) or not np.isfinite(a).all() for a in v.values()):
            raise ValueError('Invalid procedural geometry vector.')
        if p['primitive'] not in {'box','sphere','cylinder'} or p['material_role'] not in {'body','frame','accent','dark'}:
            raise ValueError('Unknown procedural part operation.')
        if np.any(v['size']<=0) or np.any(v['size']>dims*2.0+.2) or np.any(abs(v['center'])>dims*1.7+.2):
            raise ValueError('Procedural element outside its local-size envelope.')
        result.append(dict(name=f'{ident(p["name"])}_{j:02}',primitive=p['primitive'],material_role=p['material_role'],
                           **{k:v[k].tolist() for k in v}))
    return result


def query(key,category=None,geometry=None,description=None,material=None,recipe=None,parts=None,motion=None,scene_type='generic'):
    geom=geometry or STOCK[key][2]
    semantic,grouper,_=STOCK.get(key,(category, 'Furniture',geom))
    return dict(query_id=key,semantic=category or semantic,category=grouper,dimensions_m=list(geom),
                description=description or STOCK_DESC.get(key,key),material=material,recipe=recipe,
                shape_parts=parts or [],motion=motion or 'static',scene_type=scene_type,
                spatial_role='wall' if key in {'WindowBay','DoorPortal','WallSegment'} else ('tabletop' if key in {'FlatMonitor','ComputerTower','StorageBox','LooseThrowPillow'} else 'floor'),
                instance_unit='single',
                procedural_hash=hashlib.sha256(json.dumps([key,recipe],sort_keys=True).encode()).hexdigest())


def seeded_plan(e,seed,observations):
    survey=seed.copy()
    scene_type=survey.get('scene_type','generic')
    # The supplied surveyed poses are controlled experimental inputs; Kimi observations
    # are independently reported and used to attach image evidence for retrieval.
    keys=set(i[1] for i in survey['instances'])
    keys|={'FabricBarrier','WindowBay','StripLuminaire'}
    qs={}
    for key in sorted(keys):
        q=query(key)
        candidates=[x for x in observations if semantic(x)==q['semantic']]
        if key=='TrefoilWoodBench' and not candidates:
            candidates=[x for x in observations if semantic(x)=='table']
        q['views']=(max(candidates,key=lambda x:x.get('confidence',0)).get('views',[]) if candidates else [])
        q['scene_type']=scene_type
        q['prefer_public']=True
        qs[key]=q
    return survey,qs


def color_support(ev, query, z_bounds):
    if ev['xyz'] is None:
        return None
    target=np.array(query['color'],float)
    target=target-target.mean()
    if np.linalg.norm(target)<.14:
        return None
    rgb=ev['colors'].astype(float)/255
    chroma=rgb-rgb.mean(axis=1,keepdims=True)
    norm=np.linalg.norm(chroma,axis=1)
    angle=(chroma@target)/np.maximum(.0001,norm*np.linalg.norm(target))
    xyz=ev['xyz']
    valid=(angle>.83)&(norm>.095)&(xyz[:,2]>z_bounds[0])&(xyz[:,2]<z_bounds[1])
    return xyz[valid]


def colored_partitions(ev,observations):
    cues=[x for x in observations if semantic(x) in {'barrier','partition_wall'}]
    if not cues:
        return []
    best=max(cues,key=lambda x:x.get('confidence',0))
    points=color_support(ev,infer_finish(best),(.075,1.8))
    if points is None or len(points)<60:
        return []
    rng=np.random.default_rng(17)
    runs=[]
    for _ in range(9):
        if len(points)<48: break
        count=0
        support=None
        for _ in range(150):
            i,j=rng.choice(len(points),2,replace=False)
            delta=points[j,:2]-points[i,:2]
            length=np.linalg.norm(delta)
            if length<1.1: continue
            normal=np.array([-delta[1],delta[0]])/length
            inliers=abs((points[:,:2]-points[i,:2])@normal)<.17
            if int(inliers.sum())>count:
                support=inliers
                count=int(inliers.sum())
        if count<42: break
        cluster=points[support]
        points=points[~support]
        center=np.median(cluster[:,:2],axis=0)
        _,_,axis=np.linalg.svd(cluster[:,:2]-center,full_matrices=False)
        d=axis[0]
        x=(cluster[:,:2]-center)@d
        order=np.argsort(x)
        sort=x[order]
        splits=np.where(np.diff(sort)>.63)[0]+1
        for group in np.split(order,splits):
            if len(group)<33: continue
            lo,hi=np.quantile(x[group],[.025,.975])
            if hi-lo<.90: continue
            height=float(np.clip(np.quantile(cluster[group,2],.93)+.08,.57,1.65))
            first=[*(center+d*(lo-.055)),0.]
            last=[*(center+d*(hi+.055)),0.]
            num=max(1,round((hi-lo)/1.62))
            a,b=np.array(first[:2]),np.array(last[:2])
            duplicate=False
            for _,start,stop,*_ in runs:
                c,e=np.array(start[:2]),np.array(stop[:2])
                t=e-c
                length=np.linalg.norm(t)
                t/=length
                direction=(b-a)/np.linalg.norm(b-a)
                uv=sorted([(a-c)@t,(b-c)@t])
                overlap=max(0,min(length,uv[1])-max(0,uv[0]))
                lateral=abs(((a+b)*.5-c)@np.array([-t[1],t[0]]))
                if abs(t@direction)>.957 and lateral<.46 and overlap>.43*min(length,uv[1]-uv[0]):
                    duplicate=True
                    break
            if not duplicate:
                runs.append([f'CloudBarrier_{len(runs):02}',first,last,num,height,'observed_chroma_linear_support'])
    return runs


def refine_ball(ev,ob,estimate,diameter):
    pts=color_support(ev,infer_finish(ob),(.025,diameter*1.45))
    if pts is None or len(pts)<16:
        return estimate,diameter
    radius=max(.45,diameter*.85)
    short=pts[np.linalg.norm(pts[:,:2]-estimate[:2],axis=1)<2.5]
    if len(short)<16: return estimate,diameter
    check=short[::max(1,len(short)//100)]
    votes=np.sum(np.linalg.norm(check[:,None,:2]-short[None,:,:2],axis=2)<radius,axis=1)
    center=check[int(np.argmax(votes)),:2]
    support=short[np.linalg.norm(short[:,:2]-center,axis=1)<radius]
    if len(support)<18:
        return estimate,diameter
    lo,hi=np.percentile(support,[5,95],axis=0)
    size=float(np.clip(max(hi[2]+.04,np.max(hi[:2]-lo[:2])),.26,diameter*1.25))
    return np.array([*((lo+hi)[:2]/2),0.]),size


def contradictory_observations(first,second):
    rows={v.get('index'):v.get('bbox') for v in first.get('views',[])}
    for v in second.get('views',[]):
        a=rows.get(v.get('index'))
        b=v.get('bbox')
        if a is None or b is None or len(a)!=4 or len(b)!=4:
            continue
        dims=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
        if np.prod(dims)<.08*min(np.prod(np.subtract(a[2:],a[:2])),np.prod(np.subtract(b[2:],b[:2]))):
            return True
    return False


def general_plan(api,ev,objects,job,scene_type='generic'):
    from support import lift_tabletop,bind_to_furniture
    from layout import (semantic_layout, partitions, render_architecture, path_samples,
                        enrich, save_layout)
    found=[]
    runs=colored_partitions(ev,objects)
    for ob in sorted(objects,key=lambda x:x.get('confidence',0),reverse=True):
        k=semantic(ob)
        if k in {'light','ceiling','wall','floor'}: continue
        d=infer_dimensions(ob,k)
        lifted=lift_ground(ob,ev,k,d)
        if lifted is None and k=='person':
            from placement import lift_occluded_person
            lifted=lift_occluded_person(ev,ob)
        if lifted is None and (k in TABLETOP or ob.get('support')=='tabletop'):
            lifted=lift_tabletop(ev,ob)
        if lifted is None: continue
        xyz,yaw,views=lifted
        if k=='ball':
            xyz,diameter=refine_ball(ev,ob,xyz,float(np.median(d)))
            d=np.repeat(diameter,3)
        match=next((r for r in found if r['semantic']==k and np.linalg.norm(xyz-r['xyz'])<
                    max(.48,min(d[:2])*1.05) and not contradictory_observations(r['object'],ob)),None)
        if match is not None:
            frames={t.get('index') for t in match['object'].get('views',[])}
            match['object']={**match['object'],'views':[*match['object'].get('views',[]),
                           *[x for x in ob.get('views',[]) if x.get('index') not in frames]]}
            continue
        found.append(dict(semantic=k,dimensions=d,xyz=xyz,yaw=yaw,views=views,object=ob))
    path=path_samples(ev)
    envelope, semantics=semantic_layout(api,ev,job,path,found)
    lo,hi,h,*_=envelope
    mid=(lo+hi)/2
    extent=hi-lo
    walls,rooms,partition_decisions=partitions(envelope,semantics,path,found)
    qs={}
    instances=[]
    windows=[]
    lamps=[]
    notes=['Walls enclose measured or inferred room bounds. Exterior openings are physically sealed.',
           'Inferred context is placed separately from observed landmarks after clearance checks.']
    def use(n,k,p,yaw=0,scale=(1,1,1),e='geometry_estimate'):
        ks=k['query_id'] if isinstance(k,dict) else k
        qs.setdefault(ks,k if isinstance(k,dict) else query(k))
        instances.append([ident(n),ks,list(p),yaw,list(scale),e])
    room_records,wall_records=render_architecture(envelope,walls,rooms,use,semantics)
    support_resolution=[]
    for index,r in enumerate(found): r['name']=f'{r["semantic"]}_{index:03}'
    placement_observations={}
    furniture=[r for r in found if r['semantic'] in {'desk','table'}]
    for r in found:
        item=r['object']
        r['supported']=(r['semantic'] in TABLETOP or item.get('support')=='tabletop')
        if not r['supported']: continue
        binding=bind_to_furniture(ev,r,furniture)
        if binding is not None:
            r['support_instance'],r['xyz'],evidence=binding
            r['support_aligned']=True
        else:
            evidence='unresolved'
        support_resolution.append(dict(model_id=item.get('id'),support_aligned=r.get('support_aligned',False),
                                       category=r['semantic'],source=evidence))
    if support_resolution:
        put_json(job/'SupportResolution.json',support_resolution)
    unknown=[]
    for i,r in enumerate(found):
        k=r['semantic']
        ob=r['object']
        r['name']=f'{k}_{i:03}'
        if k in {'door','window'} or (k=='barrier' and runs): continue
        if k=='partition' and any(w.id.startswith('Partition_') for w in walls): continue
        if r['supported'] and not r.get('support_aligned',False):
            continue
        if np.any(r['xyz'][:2]<lo-.25) or np.any(r['xyz'][:2]>hi+.25):
            continue
        r['xyz'][:2]=np.clip(r['xyz'][:2],lo+.10,hi-.10)
        evidence='tabletop_support_estimate' if r['supported'] else (
            'image_and_cloud' if ev['xyz'] is not None else 'image_ground_ray')
        finish=infer_finish(ob)
        placement_observations[r['name']]=dict(object=ob,size_m=r['dimensions'].tolist(),
                                              support_instance=r.get('support_instance'))
        if k=='column':
            use(r['name'],'Column',r['xyz'],scale=(r['dimensions'][0],r['dimensions'][1],h),e='video_column_proxy')
            continue
        if k in TEMPLATES:
            style,role=TEMPLATES[k]
            stock_dims=STOCK[style][2]
            native_xy=[r['dimensions'][0],r['dimensions'][1]]
            if (stock_dims[1]>stock_dims[0]) != (native_xy[1]>native_xy[0]):
                native_xy.reverse()
            dims=(*native_xy,float(r['dimensions'][2]))
            q=query(style)
            q.update(semantic=k,views=ob.get('views',[]),requested_dimensions_m=list(dims),
                     observation_description=ob.get('description',''),prefer_public=True)
            if k=='person':
                q['public_required']=True
                q['description']='A standing human as a single prebuilt pose.'
            if k=='table':
                q.update(query_id=f'TableMatch_{r["name"]}',template=style,material=finish,replace_materials=[role])
            base_color,base_texture=BASE_FINISH[style]
            if np.linalg.norm(np.asarray(finish['color'])-base_color)>.26 or (
                k in {'sofa','bench','table','desk','chair','shelf'} and base_texture!=finish['texture']):
                rounded=[int(v*7) for v in finish['color']]
                suffix='_'.join(map(str,rounded))+'_'+finish['texture']
                q.update(query_id='Finish_'+style+'_'+suffix,template=style,material=finish,replace_materials=[role])
            # A stock recipe describes shape. Appearance is a separate variable.
            use(r['name'],q,r['xyz'],r['yaw'],np.asarray(dims)/stock_dims,evidence)
        else:
            unknown.append(r)
    for x in unknown:
        ob=x['object']
        d=x['dimensions']
        q=query('New_'+ident(x['name']),x['semantic'],d.tolist(),ob.get('description',''),infer_finish(ob),
                None,ob.get('shape_parts',[]),ob.get('motion'))
        q.update(views=ob.get('views',[]),defer_recipe=True,prefer_public=True,
                 observation_description=ob.get('description',''),requested_dimensions_m=d.tolist())
        use(x['name'],q,x['xyz'],x['yaw'],e='tabletop_support_estimate' if x['supported'] else 'visual_recipe_estimate')
    context=enrich(envelope,semantics,walls,room_records,path,ev,instances,qs,use)
    save_layout(job,envelope,wall_records,room_records,context,path,semantics,partition_decisions)
    for r,(room_l,room_h) in enumerate(rooms):
        room_mid=(room_l+room_h)/2
        room_size=room_h-room_l
        number=int(np.clip(round(room_size[1]/2.6),1,6))
        for i,y in enumerate(np.linspace(room_l[1]+.4,room_h[1]-.4,number)):
            watts=float(np.clip(np.prod(room_size)*18/number,72,220))
            lamps.append([f'Strip_{r:02}_{i:02}',[room_mid[0],float(y),h-.08],
                          float(max(.7,room_size[0]*.70)),0,watts])
    qs['StripLuminaire']=query('StripLuminaire')
    for _q in qs.values(): _q['scene_type']=scene_type
    if runs:
        qs['FabricBarrier']=query('FabricBarrier')
        notes.append('Portable partition chains use line support from RGB points with visual color evidence.')
    frames=choices(list(ev['photos']),4)
    return dict(source_to_room_rotation=ev['R'].tolist(),room_origin_in_source_m=ev['origin'].tolist(),
                scene_type=scene_type,instances=instances,barrier_runs=runs,window_centres=windows,light_strips=lamps,
                notes=notes,target_index=frames[len(frames)//2],preview_indices=frames,
                placement_observations=placement_observations,room_bounds=[lo.tolist(),hi.tolist()]),qs


def query_crops(queries,ev,job):
    output={}
    for key,q in queries.items():
        for view in q.get('views',[]):
            index=view.get('index')
            if index not in ev['photos'] or 'bbox' not in view: continue
            im=Image.open(ev['photos'][index]).convert('RGB')
            x0,y0,x1,y1=np.asarray(view['bbox'])*[im.width,im.height,im.width,im.height]
            if x1-x0<12 or y1-y0<12: continue
            crop=im.crop((max(0,x0),max(0,y0),min(im.width,x1),min(im.height,y1)))
            crop.thumbnail((240,240))
            f=job/'Evidence'/f'crop_{key}.jpg'
            crop.save(f,quality=88)
            output[key]=f
            break
    return output
