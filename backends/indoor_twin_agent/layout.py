"""Estimate enclosed rooms, supported portal transitions and conservative furnishing."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw

from evidence import put_json, ray_pose
from planner import query


def path_samples(ev, step=.08):
    p=np.asarray([s['xyz'] for s in ev['track']['subsample_xyz']])
    room=(p-ev['origin'])@ev['R'].T
    result=[room[0]]
    for a,b in zip(room,room[1:]):
        n=max(1,math.ceil(np.linalg.norm(b-a)/step))
        result.extend(a+(b-a)*(k/n) for k in range(1,n+1))
    return np.asarray(result)


def manhattan(points):
    """Use tall structural surfaces to resolve yaw modulo pi/2."""
    if points is None:
        return 0.,0.
    cloud=points[(points[:,2]>.85)&(points[:,2]<4.4)]
    if len(cloud)<450:
        return 0.,0.
    cloud=cloud[::max(1,len(cloud)//18000)]
    xy=cloud[:,:2]
    def strength(deg):
        t=np.deg2rad(deg)
        rot=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        coord=xy@rot
        total=0.
        for x in coord.T:
            # Limit the contribution of small concentrated objects by combining neighboring peaks.
            counts=np.bincount(np.floor((x-x.min())/.045).astype(int))
            s=gaussian_filter1d(counts.astype(float),1.45)
            strongest=[]
            for _ in range(3):
                k=int(np.argmax(s))
                if s[k]<=0: break
                strongest.append(s[k])
                s[max(0,k-8):k+9]=0
            total+=sum(strongest)
        return total
    degs=np.arange(-45.,45.,2.)
    scores=np.asarray([strength(x) for x in degs])
    best=degs[np.argmax(scores)]
    candidates=np.arange(best-2.25,best+2.3,.25)
    measured=np.asarray([strength(x) for x in candidates])
    contrast=float(measured.max()/max(1,np.median(scores)))
    return (float(candidates[np.argmax(measured)]) if contrast>1.15 else 0.),contrast


def align_evidence(ev,job):
    theta,contrast=manhattan(ev['xyz'])
    a=np.deg2rad(theta)
    R=np.array([[np.cos(a),np.sin(a),0],[-np.sin(a),np.cos(a),0],[0,0,1]])
    ev['R']=R@ev['R']
    if ev['xyz'] is not None:
        ev['xyz']=ev['xyz']@R.T
    ev['audit'].update(manhattan_yaw_deg=theta,manhattan_peak_contrast=contrast)
    put_json(job/'InputAudit.json',ev['audit'])


def wall_candidates(xyz,height):
    if xyz is None:
        return []
    points=xyz[(xyz[:,2]>.85)&(xyz[:,2]<height-.10)]
    if len(points)<200:
        return []
    if len(points)>85000:
        points=points[::math.ceil(len(points)/85000)]
    surfaces=[]
    for axis in (0,1):
        p=points[:,axis]
        edges=np.arange(p.min()-.05,p.max()+.09,.055)
        counts=np.histogram(p,edges)[0].astype(float)
        score=gaussian_filter1d(counts,.85)
        seen=[]
        for _ in range(16):
            k=int(np.argmax(score))
            if score[k]<max(14,len(points)*.0013): break
            pos=(edges[k]+edges[k+1])/2
            score[max(0,k-5):k+6]=0
            hit=points[abs(p-pos)<.095]
            tangent=hit[:,1-axis]
            z=hit[:,2]
            if len(hit)<42 or np.ptp(tangent)<.8:
                continue
            first=float(np.quantile(tangent,.04))
            last=float(np.quantile(tangent,.96))
            cells=np.floor((tangent-first)/.24).astype(int)
            levels=np.floor((z-.85)/.24).astype(int)
            unique=np.unique(np.stack([cells,levels],axis=1),axis=0)
            group,num=np.unique(unique[:,0],return_counts=True)
            mid=group[(num>=3)&(group>=0)&(group*.24<last-first)]
            tall_span=float(len(mid)*.24)
            coverage=tall_span/max(.24,last-first)
            if not len(mid) or coverage<.23: continue
            high=np.count_nonzero(z>1.58)
            if high<max(10,len(z)*.10): continue
            if any(abs(pos-old)<.32 for old in seen): continue
            seen.append(pos)
            surfaces.append(dict(id=f'P{axis}_{len(seen):02}',axis=axis,offset_m=float(np.median(hit[:,axis])),
                span_m=[first,last],tall_span_m=tall_span,vertical_coverage=float(coverage),points=len(hit)))
    return surfaces


def estimate_height(q):
    if q is None:
        return 2.75,'standard_height_prior'
    s=q[(q[:,2]>2.05)&(q[:,2]<4.75)]
    if len(s)>65:
        orders=np.floor(s/.12).astype(int)
        cell=np.unique(orders,axis=0)
        z,counts=np.unique(cell[:,2],return_counts=True)
        peak=int(np.argmax(counts))
        n=int(counts[peak])
        if n>=32:
            ident=np.abs(s[:,2]-(z[peak]+.5)*.12)<.105
            return float(np.clip(np.median(s[ident,2]),2.23,4.55)),'observed_upper_horizontal_band'
    return 2.75,'standard_height_prior'


def envelope(ev,path,found):
    xyz=ev['xyz']
    h,mode=estimate_height(xyz)
    candidates=wall_candidates(xyz,h)
    cam=path[:,:2]
    if xyz is None:
        f=np.array([r['xyz'][:2] for r in found]).reshape(-1,2)
        combo=np.concatenate([cam,f])
        qlo=combo.min(0)-.70
        qhi=combo.max(0)+.70
    else:
        fit=xyz[(xyz[:,2]>-.15)&(xyz[:,2]<h+.25)]
        if len(fit)<30: fit=xyz
        qlo,qhi=np.percentile(fit[:,:2],[2,98],axis=0)
    e=[]
    chosen=[]
    for axis in (0,1):
        for sign in (-1,1):
            end=float(cam[:,axis].min() if sign<0 else cam[:,axis].max())
            rough=float(qlo[axis] if sign<0 else qhi[axis])
            candidates_on_side=[c for c in candidates if c['axis']==axis and sign*(c['offset_m']-end)>=.19
                                and abs(c['offset_m']-rough)<max(1.30,.24*(qhi[axis]-qlo[axis]))]
            best=max(candidates_on_side,
                     key=lambda c:c['tall_span_m']*c['vertical_coverage']/(1+.40*abs(c['offset_m']-rough)),
                     default=None)
            if best and best['tall_span_m']>=.75:
                v=best['offset_m']
                chosen.append(dict(plane_id=best['id'],type='supported_wall_plane',offset_m=v))
            else:
                v=(min(end-.36,rough-.14) if sign<0 else max(end+.36,rough+.14))
                chosen.append(dict(plane_id=None,type='inferred_enclosure',offset_m=v))
            e.append(v)
    lo=np.array(e)[[0,2]]
    hi=np.array(e)[[1,3]]
    mid=(lo+hi)/2
    lo=np.minimum(lo,mid-.86)
    hi=np.maximum(hi,mid+.86)
    return lo,hi,h,mode,chosen,candidates


def map_evidence(ev,path,layout,path_out):
    lo,hi,planes=layout
    W,H=800,650
    mx=np.maximum(hi-lo,.1)
    sc=float(min((W-112)/mx[0],(H-122)/mx[1]))
    def map_xy(p):
        x,y=(np.asarray(p)[:2]-lo)*sc
        return 57+float(x),H-47-float(y)
    img=Image.new('RGB',(W,H),'#14232c')
    d=ImageDraw.Draw(img)
    q=ev['xyz']
    if q is not None:
        qc=q[(q[:,2]>.24)&(q[:,2]<3.9)]
        for a in qc[::max(1,len(qc)//9500)]:
            x,y=map_xy(a)
            d.point((x,y),fill='#6c8387')
    a=map_xy(lo);b=map_xy(hi)
    d.rectangle((a[0],b[1],b[0],a[1]),outline='#f0cf80',width=3)
    for w in planes:
        axis=w['axis'];t=w['span_m']
        p=[[w['offset_m'],t[i]] if axis==0 else [t[i],w['offset_m']] for i in (0,1)]
        uv=[map_xy(x) for x in p]
        d.line(uv,fill='#457e84',width=2)
        ux,uy=map_xy(np.mean(p,axis=0))
        d.text((ux+2,uy+2),w['id'],fill='#9cd9d1')
    d.line([map_xy(x) for x in path[::max(1,len(path)//500)]],fill='#ef6663',width=3)
    d.text((24,19),'Ground plane | gold = closed bounds | red = camera trace | cyan = candidate planes',fill='#faf1df')
    img.save(path_out,quality=93)


def semantic_layout(api,ev,job,path,found):
    lo,hi,h,height_source,ext,planes=envelope(ev,path,found)
    path_out=job/'Evidence'/'FloorplanEvidence.jpg'
    map_evidence(ev,path,(lo,hi,planes),path_out)
    photos=sorted(ev['photos'])
    indices=[photos[i] for i in sorted(set(np.round(np.linspace(0,len(photos)-1,min(5,len(photos)))).astype(int)))]
    pref=[(f'Original decoded frame {i}',ev['photos'][i]) for i in indices]
    ctx=dict(bounds_m=[lo.tolist(),hi.tolist()],ceiling_height_m=h,candidate_wall_planes=planes,
             views=[{'index':i,'camera_m':ray_pose(ev['track'],ev['pts'][i],ev['R'],ev['origin'])[0].round(4).tolist()} for i in indices],
             recovered_objects=[dict(category=r['semantic'],xyz_m=np.asarray(r['xyz']).round(3).tolist()) for r in found])
    if api is None:
        semantic=dict(partitions=[],surface_finishes={'floor_style':'floor','wall_style':'paint',
                    'floor_rgb':[.25,.25,.24],'wall_rgb':[.74,.72,.68]},notes=['offline layout: measured envelope only'])
    else:
        semantic=api.ask('layout',ctx,[('metric top-down evidence',path_out),*pref],3500)
    return (lo,hi,h,height_source,ext,planes),semantic


@dataclass
class Wall:
    id: str
    axis: int
    offset: float
    lo: float
    hi: float
    support: str
    portals: list


def crossing_positions(path,axis,offset):
    c=[]
    for a,b in zip(path,path[1:]):
        d=b[axis]-a[axis]
        if abs(d)<1e-8:
            continue
        step=(offset-a[axis])/d
        if 0<=step<1:
            c.append(float(a[1-axis]+(b[1-axis]-a[1-axis])*step))
    return c


def partitions(env,model,path,objects):
    lo,hi,height,height_source,external,plane_candidates=env
    walls=[]
    h=float(height)
    for j in range(4):
        axis=j//2
        offset=[lo[0],hi[0],lo[1],hi[1]][j]
        walls.append(Wall(f'Exterior_{j}',axis,float(offset),float(lo[1-axis]),float(hi[1-axis]),
                          external[j]['type'],[]))
    rooms=[(lo.copy(),hi.copy())]
    reasons=[]
    by_id={p['id']:p for p in plane_candidates}
    for proposal in model.get('partitions',[])[:4]:
        c=by_id.get(proposal.get('plane_id'))
        if c is None:
            continue
        axis=c['axis'];offset=c['offset_m'];t=1-axis
        for i,(rlo,rhi) in enumerate(rooms):
            if not rlo[axis]+1.24<offset<rhi[axis]-1.24:
                continue
            coverage=min(rhi[t],c['span_m'][1])-max(rlo[t],c['span_m'][0])
            if coverage<.63*(rhi[t]-rlo[t]) or c['vertical_coverage']<.53:
                continue
            left=(path[:,axis]<offset-.32)&np.all((path[:,:2]>=rlo)&(path[:,:2]<=rhi),axis=1)
            right=(path[:,axis]>offset+.32)&np.all((path[:,:2]>=rlo)&(path[:,:2]<=rhi),axis=1)
            crosses=[p for p in crossing_positions(path,axis,offset) if rlo[t]<p<rhi[t]]
            door=[r for r in objects if r['semantic']=='door' and abs(r['xyz'][axis]-offset)<.80
                  and rlo[t]<r['xyz'][t]<rhi[t]]
            if not left.any() or not right.any() or not crosses or not door:
                reasons.append(dict(plane_id=c['id'],status='unresolved_connection',reason='full-height evidence, verified door and trace on both sides required'))
                continue
            middle=float(np.median(crosses))
            if np.ptp(crosses)>.85 or not any(abs(d['xyz'][t]-middle)<.88 for d in door):
                reasons.append(dict(plane_id=c['id'],status='rejected_ambiguous_crossings'))
                continue
            width=max(.82,min(1.26,np.ptp(crosses)+.64))
            a,b=middle-width/2,middle+width/2
            if a<rlo[t]+.14 or b>rhi[t]-.14: continue
            walls.append(Wall(f'Partition_{len(walls)-4}',axis,offset,float(rlo[t]),float(rhi[t]),
                              'measured_plane_plus_visible_door_and_crossing',[(a,b,0.,2.12,'connected_door')]))
            rlo2=rlo.copy();rlo2[axis]=offset
            rhi1=rhi.copy();rhi1[axis]=offset
            rooms[i:i+1]=[(rlo,rhi1),(rlo2,rhi)]
            reasons.append(dict(plane_id=c['id'],status='confirmed_portal'))
            break
    for r in objects:
        k=r['semantic']
        if k not in {'door','window'}:
            continue
        w=min(walls[:4],key=lambda w:abs(w.offset-r['xyz'][w.axis]))
        d=abs(w.offset-r['xyz'][w.axis])
        if d>(.64 if k=='window' else .72):
            continue
        y=float(r['xyz'][1-w.axis])
        wide=float(np.clip(max(r['dimensions'][:2]),.62,2.65 if k=='window' else 1.35))
        low=float(np.clip(r['xyz'][2],.55,h-1.12)) if k=='window' else 0.
        top=min(h-.16,low+float(r['dimensions'][2]))
        a,b=max(w.lo+.105,y-wide/2),min(w.hi-.105,y+wide/2)
        if b-a<.55 or top-low<.59: continue
        if any(a<p[1]+.13 and b>p[0]-.13 for p in w.portals): continue
        w.portals.append((a,b,low,top,'sealed_window' if k=='window' else 'closed_boundary_door'))
    return walls,rooms,reasons


def xyz_on(w,t,z):
    return (w.offset,t,z) if w.axis==0 else (t,w.offset,z)


def wall_size(w,t,z,thick=.14):
    return (thick,t,z) if w.axis==0 else (t,thick,z)


def decoration_query(asset,typ,body,parts):
    def m():
        return dict(color=body,texture='wood' if typ=='cabinet' else 'paint',roughness=.61,metallic=0.)
    key=dict(cabinet='VLNStorageCabinet',plant='VLNPlanter',art='VLNWallArtwork',
             bin='VLNWasteBin',book='VLNBooks')[typ]
    recipe=[]
    for shape,n,p,s,slot in parts:
        recipe.append(dict(name=n,primitive=shape,center=p,size=s,material_role=slot,rotation_deg=[0,0,0]))
    bound=np.max([np.abs(np.array(p['center']))+np.array(p['size'])/2 for p in recipe],axis=0)
    return query(key,typ,[float(2*bound[0]),float(2*bound[1]),float(bound[2])],asset,m(),recipe,[x['name'] for x in recipe])


DECOR={
 'cabinet':decoration_query('A two-door low storage cabinet','cabinet',[.41,.29,.19],[
    ('box','Carcass',[0,0,.35],[.35,.94,.70],'body'),('box','Door_L',[-.183,-.226,.365],[.026,.447,.61],'body'),
    ('box','Door_R',[-.183,.226,.365],[.026,.447,.61],'body'),
    ('box','Handle_L',[-.203,-.063,.44],[.03,.024,.12],'frame'),
    ('box','Handle_R',[-.203,.063,.44],[.03,.024,.12],'frame')]),
 'plant':decoration_query('Small plant in a ceramic floor pot','plant',[.19,.31,.19],[
    ('cylinder','Pot',[0,0,.115],[.27,.27,.23],'dark'),
    ('cylinder','Stem',[0,0,.321],[.035,.035,.29],'accent'),
    ('sphere','Leaves_A',[-.06,.03,.43],[.22,.28,.20],'body'),('sphere','Leaves_B',[.08,-.07,.48],[.28,.24,.25],'body'),
    ('sphere','Leaves_C',[.04,.04,.60],[.21,.20,.23],'body')]),
 'art':decoration_query('Wall-mounted print with a timber frame','art',[.42,.51,.57],[
    ('box','OuterFrame',[0,0,.176],[.028,.53,.352],'dark'),
    ('box','Print',[-.018,0,.176],[.01,.48,.304],'body'),
    ('box','Inset',[-.025,.065,.173],[.004,.19,.21],'accent')]),
 'bin':decoration_query('Small open-top bin approximation','bin',[.32,.31,.27],[
    ('cylinder','Tub',[0,0,.17],[.24,.24,.34],'body'),
    ('cylinder','TopRim',[0,0,.335],[.256,.256,.022],'dark')]),
 'book':decoration_query('A short unlabelled stack of books','book',[.35,.19,.105],[
    ('box','Volume_A',[0,0,.017],[.22,.29,.032],'body'),
    ('box','Volume_B',[.014,0,.049],[.19,.26,.034],'accent'),
    ('box','Volume_C',[.012,.015,.077],[.17,.245,.021],'dark')])}


def render_architecture(env,walls,rooms,use,model=None):
    lo,hi,h,_,_,_=env
    mid=(lo+hi)/2; sizes=hi-lo
    finish=(model or {}).get('surface_finishes',{})
    def template(key,role,col,style):
        name=key+'_'+style+'_'+'_'.join(str(int(f*10)) for f in col)
        q=query(key)
        q.update(query_id=name,template=key,replace_materials=[role],material={
                'color':list(col),'texture':style,'roughness':.54,'metallic':0.})
        return q
    def color(v,fallback):
        c=np.asarray(v if v is not None else fallback,dtype=float)
        return np.clip(c,.025,.98) if c.shape==(3,) and np.isfinite(c).all() else np.asarray(fallback)
    floor_style=str(finish.get('floor_style','floor'))
    if floor_style not in {'floor','wood','paint','cloth'}: floor_style='floor'
    floor_col=color(finish.get('floor_rgb'),[.25,.25,.24])
    wall_style='wood' if finish.get('wall_style')=='wood' else 'paint'
    wall_col=color(finish.get('wall_rgb'),[.74,.72,.68])
    floor_key=template('LabRaisedFloor','Floor_graphite',floor_col,floor_style)
    wall_key=template('WallSegment','Paint_ivory',wall_col,wall_style)
    use('Floor',floor_key,(*mid,0),scale=(sizes[0]/18.3,sizes[1]/21.8,1))
    use('Ceiling','CeilingSlab',(*mid,h),scale=(*sizes,.16))
    leaf=decoration_query('Sealed painted wood door leaf','cabinet',[.53,.40,.27],[
      ('box','Leaf',[0,0,.5],[.072,1.,1.],'body')])
    leaf.update(query_id='VLNClosedDoorLeaf',semantic='closed_door_leaf',category='Architecture',dimensions_m=[.072,1.,1.])
    room_data=[dict(id=f'Room{i:02}',low_m=a.tolist(),high_m=b.tolist()) for i,(a,b) in enumerate(rooms)]
    ledger=[]
    for w in walls:
        apertures=[]
        last=w.lo-.071
        def piece(a,b,z0,z1,key):
            if b-a<.008 or z1-z0<.008:
                return
            use(f'{w.id}_{key}',wall_key,xyz_on(w,(a+b)/2,z0),
                scale=wall_size(w,b-a,z1-z0),e=w.support)
        for i,(a,b,z0,z1,kind) in enumerate(sorted(w.portals)):
            piece(last,a,0,h,f'Jamb_{i}')
            piece(a,b,0,z0,f'Sill_{i}')
            piece(a,b,z1,h,f'Lintel_{i}')
            mid_t=(a+b)/2
            xyz=xyz_on(w,mid_t,(z0+z1)/2)
            if kind=='sealed_window':
                # Determine facing from the outside wall sign. The opening stays physical through glazing.
                yaw=90 if w.axis==0 and w.offset>mid[0] else -90 if w.axis==0 else 0 if w.offset>mid[1] else 180
                use(f'{w.id}_Window_{i}','WindowBay',xyz,yaw=yaw,
                    scale=((b-a)/1.49,1,(z1-z0)/2.33),e='visible_window_physically_sealed')
            else:
                yaw=0 if w.axis==0 else 90
                use(f'{w.id}_Frame_{i}','DoorPortal',xyz_on(w,mid_t,0),yaw=yaw,
                    scale=(1,(b-a)/1.44,z1/2.66),e=kind)
                if kind=='closed_boundary_door':
                    use(f'{w.id}_Leaf_{i}',leaf,xyz_on(w,mid_t,0),yaw=yaw,scale=(1,b-a,z1),e=kind)
            adjacent=[]
            if kind=='connected_door':
                for r in room_data:
                    rrlo=np.array(r['low_m']);rrhi=np.array(r['high_m'])
                    if min(abs(w.offset-rrlo[w.axis]),abs(w.offset-rrhi[w.axis]))<.04 and rrlo[1-w.axis]<mid_t<rrhi[1-w.axis]:
                        adjacent.append(r['id'])
            apertures.append(dict(type=kind,span_m=[a,b],height_m=[z0,z1],room_ids=adjacent))
            last=b
        piece(last,w.hi+.071,0,h,'End')
        ledger.append(dict(id=w.id,axis=w.axis,offset_m=w.offset,span_m=[w.lo,w.hi],support=w.support,
                           apertures=apertures,exterior=w.id.startswith('Exterior')))
    return room_data,ledger


def use_type(r):
    k=str(r).lower()
    return k if k in {'office','lounge','bedroom','kitchen','hall','storage','generic'} else 'generic'


def enrich(env,model,walls,room_data,path,ev,instances,queries,use):
    """Populate distant margins while protecting observed assets and the camera sweep."""
    lo,hi,h,*_=env
    path_index=cKDTree(path[:,:2])
    records=[]
    occupied=[]
    for n,key,p,yaw,s,e,*_ in instances:
        if queries[key]['category'] in {'Architecture','Lighting'}: continue
        d=np.array(queries[key]['dimensions_m'])*np.asarray(s)
        a=math.radians(yaw)
        rot=np.abs(np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]]))
        occupied.append((np.asarray(p)[:2],rot@d[:2]/2,n))
    cloud=ev['xyz']
    zbuffers=[]
    if cloud is not None:
        for i in sorted(ev['photos']):
            p,R=ray_pose(ev['track'],ev['pts'][i],ev['R'],ev['origin'])
            xyz=(cloud-p)@R*[1,-1,-1]
            keep=xyz[:,2]>.12
            xyz=xyz[keep]
            pp=xyz@ev['K'].T
            ux=np.floor(96*pp[:,0]/pp[:,2]/ev['W']).astype(int)
            vy=np.floor(72*pp[:,1]/pp[:,2]/ev['H']).astype(int)
            keep=(ux>=0)&(ux<96)&(vy>=0)&(vy<72)
            buf=np.full((72,96),np.inf)
            np.minimum.at(buf,(vy[keep],ux[keep]),pp[keep,2])
            zbuffers.append((p,R,buf))
    def clear(p,kind,d,yaw):
        a=math.radians(yaw)
        rot=np.abs(np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]]))
        half=rot@np.array(d[:2])/2
        path_distance=float(path_index.query(p[:2])[0])
        if path_distance<.62+np.linalg.norm(half): return False,'path_clearance',0
        if any(np.all(abs(p[:2]-at)<half+other+.10) for at,other,_ in occupied):
            return False,'existing_object_overlap',0
        # Reject candidates obviously exposed near a camera with measured free line of sight.
        far_views=0
        for at,R,zbuf in zbuffers:
            local=(p+np.array([0,0,d[2]*.58])-at)@R*[1,-1,-1]
            if not .2<local[2]<9: continue
            v=ev['K']@local
            px,py=np.floor(v[:2]/v[2]*[96/ev['W'],72/ev['H']]).astype(int)
            if not 0<=px<96 or not 0<=py<72: continue
            depth=zbuf[max(0,py-1):py+2,max(0,px-1):px+2].min()
            if local[2]<2.45 and np.isfinite(depth) and local[2]+.42<depth:
                return False,'observed_free_near_field',0
            if np.isfinite(depth) and depth>local[2]-.24 and local[2]>2.45:
                far_views+=1
        return True,'weak_far_view' if far_views else 'occluded_perimeter_context',far_views
    type_model=use_type(model.get('dominant_use','generic'))
    allowed={'office':['cabinet','plant','bin','art'],'lounge':['plant','cabinet','art'],
             'bedroom':['cabinet','plant','art'],'kitchen':['bin','cabinet'],'hall':['art','plant'],
             'storage':['cabinet','bin'],'generic':['cabinet','plant','art','bin']}
    chosen=[x for x in model.get('completion_palette',[]) if x in allowed[type_model]]
    palette=list(dict.fromkeys(chosen+allowed[type_model]))
    for room in room_data:
        rlo=np.array(room['low_m']);rhi=np.array(room['high_m'])
        rarea=np.prod(rhi-rlo)
        max_items=int(np.clip(1+rarea/4.2,2,8))
        candidates=[]
        for w in walls:
            axis=w.axis;t=1-axis
            if w.offset<rlo[axis]-.07 or w.offset>rhi[axis]+.07: continue
            room_center=(rlo+rhi)/2
            sign=1 if room_center[axis]>w.offset else -1
            if abs(w.offset-room_center[axis])<.12: continue
            start=max(w.lo,rlo[t]);end=min(w.hi,rhi[t])
            if end-start<1.25: continue
            for x in np.linspace(start+.61,end-.61,max(1,int((end-start)/1.34))):
                if any(a-.48<x<b+.48 for a,b,*_ in w.portals): continue
                distance=float(path_index.query(xyz_on(w,x,0)[:2])[0])
                candidates.append((distance,w,float(x),sign))
        candidates.sort(key=lambda x:x[0],reverse=True)
        made=0
        for distance,w,p,sign in candidates:
            if made>=max_items: break
            t=1-w.axis
            k=palette[made%len(palette)]
            q=DECOR[k]
            d=q['dimensions_m']
            direction=np.eye(2)[w.axis]*sign
            yaw=math.degrees(math.atan2(-direction[1],-direction[0]))
            origin=np.array(xyz_on(w,p,1.36 if k=='art' else 0.))
            origin[w.axis]+=sign*(d[0]/2+.09)
            if not np.all(origin[:2]>rlo+.02) or not np.all(origin[:2]<rhi-.02): continue
            valid,reason,fv=clear(origin,k,d,yaw)
            if not valid: continue
            # Require enough separation from measured surfaces for floor-standing fills.
            if cloud is not None and k!='art':
                diff=np.abs(cloud-origin)
                hit=(diff[:,0]<min(.16,d[0]/3))&(diff[:,1]<min(.16,d[1]/3))&(diff[:,2]>.13)&(diff[:,2]<d[2]*.82)
                if hit.sum()>10: continue
            name=f'Context_{room["id"]}_{made:02}_{k}'
            use(name,q,origin,yaw=yaw,e='inferred_context')
            a=math.radians(yaw)
            half=np.abs(np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]]))@np.asarray(d[:2])/2
            if k!='art': occupied.append((origin[:2],half,name))
            records.append(dict(id=name,category=k,room_id=room['id'],provenance='inferred_context',
                                distance_to_recorded_path_m=distance,rationale=reason,far_visible_views=fv))
            made+=1
    return records


def save_layout(job,env,walls,rooms,context,path,model,reasons):
    lo,hi,h,height_source,ext,candidates=env
    xmin,ymin=lo
    xmax,ymax=hi
    coverage=bool(len(walls)>=4 and all(not [a for a in w['apertures'] if a['type']=='connected_door']
                                        for w in walls if w['exterior']))
    crossing_ok=all(len(a['room_ids'])==2 for w in walls for a in w['apertures'] if a['type']=='connected_door')
    put_json(job/'Layout.json',dict(bounds_m=[lo.tolist(),hi.tolist()],wall_height_m=h,
             height_source=height_source,exterior_enclosed_by_construction=coverage,
             every_connected_portal_links_two_rooms=crossing_ok,rooms=rooms,walls=walls,
             inferred_context=context,partition_decisions=reasons,layout_semantics=model))
