"""Robust calibrated object-envelope fitting and support constraints."""
from itertools import product
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from evidence import put_json,ray_pose

BOX=np.array(list(product((-.5,.5),(-.5,.5),(0.,1.))))


def yaw_mat(angle):
    a=math.radians(angle)
    return np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])


def poses_for_views(ev):
    if '_known_views' not in ev:
        ev['_known_views']={i:ray_pose(ev['track'],ev['pts'][i],ev['R'],ev['origin']) for i in ev['photos']}
    return ev['_known_views']


def projections(ev):
    if '_cloud_projections' not in ev:
        ev['_cloud_projections']={}
        if ev['xyz'] is not None:
            for i,(pos,R) in poses_for_views(ev).items():
                rel=(ev['xyz']-pos)@R*[1,-1,-1]
                raw=rel@ev['K'].T
                uv=raw[:,:2]/np.maximum(.075,raw[:,2,None])
                ev['_cloud_projections'][i]=(uv/[ev['W'],ev['H']],rel[:,2])
    return ev['_cloud_projections']


def observations(ev,ob):
    view_poses=poses_for_views(ev)
    found={}
    for v in ob.get('views',[]):
        i=v.get('index')
        if i not in view_poses: continue
        b=np.array(v.get('bbox',[]),dtype=float)
        if b.shape!=(4,) or not np.isfinite(b).all() or np.any(b[2:]-b[:2]<.016):continue
        b=np.clip(b,0,1)
        if np.any(b[2:]-b[:2]<.015):continue
        partial=bool(v.get('partial',False) or np.min(b[:2])<.015 or np.max(b[2:])>.985)
        found[i]=(i,b,partial,*view_poses[i])
    return list(found.values())


def lift_occluded_person(ev,ob):
    rec=observations(ev,ob)
    if not rec:
        return None
    inv=np.linalg.inv(ev['K'])
    estimates=[]
    for i,b,partial,pos,R in rec:
        uv=(b[:2]+b[2:])/2
        ray=R@((inv@np.r_[uv*[ev['W'],ev['H']],1.])*[1,-1,-1])
        # A visible head/shoulder fragment supplies a ray to a narrow height band.
        # Full-body observations and visible feet are handled by the normal lift.
        if not partial or ray[2]<.035 or pos[2]>1.46:
            continue
        for h in [1.49,1.63,1.76]:
            length=(h-pos[2])/ray[2]
            if .28<length<9:
                p=pos+length*ray
                p[2]=0.
                anchors=[k['time_s'] for k in ev['track']['key_frames']]
                reliability=1./(.20+min(abs(ev['pts'][i]-t) for t in anchors))
                estimates.append((p,i,pos,reliability,h))
    if not estimates:return None
    xyz=np.array([x[0] for x in estimates])
    matrix=np.linalg.norm(xyz[:,None]-xyz[None,:],axis=2)
    scores=[]
    for i in range(len(xyz)):
        per_view={}
        for j in np.where(matrix[i]<.73)[0]:
            _,frame,_,reliability,head=estimates[j]
            signal=reliability*(1-.50*abs(head-1.63))
            if frame not in per_view or per_view[frame][0]<signal:
                per_view[frame]=(signal,j)
        scores.append((sum(x[0] for x in per_view.values()),per_view))
    score,used=max(scores,key=lambda x:x[0])
    ids=[v[1] for v in used.values()]
    if len(ids)<2:
        best=max(estimates,key=lambda p:p[3]-abs(p[4]-1.63))
        if best[3]<2.5:return None
    weights=np.array([v[0] for v in used.values()])
    if len(ids)==1:
        p,_,pos,_,_=best
        observers=np.stack([pos])
    else:
        p=np.average(xyz[ids],axis=0,weights=weights)
        observers=np.stack([estimates[x][2] for x in ids])
    toward=np.mean(observers[:,:2]-p[:2],axis=0)
    yaw=(math.degrees(math.atan2(toward[1],toward[0]))+90)%360
    return p,yaw,sorted({estimates[x][1] for x in ids})


def points_for_box(ev,record,center,extent,support):
    i,b,_,pos,_=record
    uv,depth=projections(ev).get(i,(None,None))
    if depth is None:return np.zeros((0,3))
    xyz=ev['xyz']
    span=b[2:]-b[:2]
    lo,hi=b[:2]+span*.15,b[2:]-span*.15
    mask=np.all((uv>=lo)&(uv<=hi),axis=1)&(depth>.18)&(depth<9)
    zlo=center[2]+(.026 if support else .04)
    zhi=center[2]+max(.13,extent[2]*1.33)
    mask&=(xyz[:,2]>zlo)&(xyz[:,2]<zhi)
    diff=xyz[:,:2]-center[:2]
    radius=float(np.clip(max(extent[:2])*.75+.28,.37,1.7))
    mask&=(np.sum(diff*diff,axis=1)<radius*radius)
    select=xyz[mask]
    if len(select)>240:
        select=select[::math.ceil(len(select)/240)]
    return select


def uv_box(center,yaw,dims,pose,ev):
    p=BOX*dims
    p[:,:2]=p[:,:2]@yaw_mat(yaw).T
    v=(p+center-pose[-2])@pose[-1]*[1,-1,-1]
    if np.any(v[:,2]<.083): return None
    w=v@ev['K'].T
    x=w[:,:2]/w[:,2,None]/[ev['W'],ev['H']]
    return np.r_[x.min(0),x.max(0)]


def residual_box(out,b,partial):
    if out is None:return [2.,2.,2.,2.]
    z=np.maximum(b[2:]-b[:2],.09)
    if partial:
        # A visible truncated region is a containment constraint.
        return [*np.maximum(0,(out[:2]-b[:2])/z),*np.maximum(0,(b[2:]-out[2:])/z)]
    a=np.clip(out,-.3,1.3)
    return [*((a[:2]+a[2:]-b[:2]-b[2:])/(2*z)),*((a[2:]-a[:2]-z)/z*.43)]


def iou(a,b):
    if a is None:return 0.
    d=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
    v=float(np.prod(d))
    return v/max(1e-6,float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-v))


def fit_one(ev,ob,pos,yaw,dims,lo,hi,can_raise):
    records=observations(ev,ob)
    if not records:
        return pos,yaw,dims,{'status':'no_valid_observations'}
    rigorous=[r for r in records if not r[2]]
    # Evaluate a held-out view if the remaining fit still has two full boxes.
    hold=rigorous[-1:] if len(rigorous)>=3 else []
    used=[r for r in records if not hold or r[0]!=hold[0][0]]
    near=[points_for_box(ev,r,pos,dims,can_raise) for r in used]
    support=np.concatenate(near) if near else np.zeros((0,3))
    if len(support)>340:
        support=support[::math.ceil(len(support)/340)]
    vote_views=sum(len(r)>=8 for r in near)
    with_depth=len(support)>=15
    if not with_depth and len(rigorous)<2:
        return pos,yaw,dims,{'status':'insufficient_metric_evidence','points':len(support)}
    lower=np.r_[np.maximum(lo+.035-pos[:2],-.52),(-.07 if can_raise else -1e-6),-105,[-.32]*3]
    upper=np.r_[np.minimum(hi-.035-pos[:2],.52),(.07 if can_raise else 1e-6),105,[.32]*3]
    if np.any(lower>=upper) or np.any(lower[:2]>0) or np.any(upper[:2]<0):
        return pos,yaw,dims,{'status':'outside_fit_domain'}
    span=np.maximum(.13,dims)
    def decode(theta):
        return pos+theta[:3],yaw+theta[3],dims*np.exp(theta[4:])
    def err(theta):
        p,a,d=decode(theta)
        out=[]
        for row in used:
            res=residual_box(uv_box(p,a,d,row,ev),row[1],row[2])
            weight=(.40 if row[2] else 1.)/math.sqrt(len(used))
            out.extend(np.array(res)*weight)
        if with_depth:
            loc=support-p
            xy=loc[:,:2]@yaw_mat(a)
            xy_err=np.maximum(0,abs(xy)-d[:2]/2)
            z_err=np.maximum(0,np.maximum(-loc[:,2],loc[:,2]-d[2]))
            d_err=np.sqrt(np.sum(xy_err*xy_err,axis=1)+z_err*z_err)
            # Reject background points conservatively instead of inflating dimensions to fit them.
            out.extend(np.minimum(d_err/np.mean(span),.35)*(.95/math.sqrt(len(support))))
        out.extend(theta[:3]*[.18,.18,.55])
        out.extend(theta[4:]*.31)
        return np.asarray(out)
    zero=np.zeros(7)
    e0=np.sum(err(zero)**2)
    options=[]
    for a in (0.,-69.,69.):
        candidate=zero.copy();candidate[3]=a
        solution=least_squares(err,candidate,bounds=(lower,upper),loss='soft_l1',f_scale=.16,max_nfev=32)
        options.append(solution.x)
    fit=min(options,key=lambda x:np.sum(err(x)**2))
    e1=np.sum(err(fit)**2)
    p,a,d=decode(fit)
    check=hold or rigorous or used
    j0=float(np.mean([iou(uv_box(pos,yaw,dims,r,ev),r[1]) for r in check]))
    j1=float(np.mean([iou(uv_box(p,a,d,r,ev),r[1]) for r in check]))
    accept=e1<e0*.96 and j1>=j0-(.06 if hold else .025)
    stats={'status':'refined' if accept else 'kept_initial','metric_points':len(support),'metric_view_count':vote_views,
           'view_count':len(records),'holdout_count':len(hold),'initial_envelope_iou':j0,
           'candidate_envelope_iou':j1,'selected_envelope_iou':j1 if accept else j0,
           'fit_objective_before':float(e0),'fit_objective_candidate':float(e1),
           'translation_delta_m':float(np.linalg.norm(fit[:3])) if accept else 0.0}
    return (p,a,d,stats) if accept else (pos,yaw,dims,stats)


def normalize_sizes(planned,nominal):
    direct=np.linalg.norm(np.log(planned/nominal))
    swap=np.linalg.norm(np.log(planned[[1,0,2]]/nominal))
    return (planned[[1,0,2]],90.) if swap+.12<direct else (planned,0.)


def refine_scene(plan,queries,dispatch,bank,ev,job):
    evidence=plan.get('placement_observations',{})
    bounds=plan.get('room_bounds')
    if not bounds:return
    lo,hi=np.array(bounds)
    entries={r['asset_id']:r for r in bank.entries()}
    data=[]
    by_id={r[0]:r for r in plan['instances']}
    originals={k:np.array(v[2]) for k,v in by_id.items()}
    widths={}
    # Refit supports first; tabletop objects inherit a valid support plane afterwards.
    iterable=sorted(plan['instances'],key=lambda i:i[5]=='tabletop_support_estimate')
    for item in iterable:
        name,key,loc,yaw,s,e,*_=item
        if name not in evidence:continue
        source=dispatch[key]
        q=queries[key]
        actual=entries.get(source.get('source_asset_id',''),{})
        native=np.array(actual.get('dimensions_m',q.get('native_dimensions_m',q['dimensions_m'])),float)
        planned=np.asarray(q['dimensions_m'],float)*np.array(s)
        if evidence[name].get('size_m') is not None and not actual:
            initial=np.asarray(evidence[name]['size_m'],float)
            planned,_=normalize_sizes(initial,native)
        pos=np.array(loc,float)
        desired,turn=normalize_sizes(planned,native) if actual else (planned,0.)
        if q['semantic']=='person':
            target=float(np.clip(desired[2],1.4,2.02))
            desired=native*(target/native[2])
        pos,angle,desired,stat=fit_one(ev,evidence[name]['object'],pos,yaw+turn,desired,lo,hi,
                                     e=='tabletop_support_estimate')
        if q['semantic']=='person':
            desired=native*(float(np.clip(desired[2],1.39,2.05))/native[2])
        if e=='tabletop_support_estimate':
            parent=evidence[name].get('support_instance')
            if parent in widths:
                row=by_id[parent]
                pd=widths[parent]
                p=np.array(row[2])
                theta=yaw_mat(row[3])
                delta=pos[:2]-p[:2]
                # Stable support contact inside the tabletop footprint.
                gap=np.maximum(.015,pd[:2]*.5-np.sort(desired[:2])*0.27-.015)
                xy=np.clip(delta@theta,-gap,gap)@theta.T
                pos[:2]=p[:2]+xy
                pos[2]=p[2]+pd[2]-.006
                stat['support_instance']=parent
                stat['support_plane_m']=float(pos[2])
        scale=np.clip(desired/native,.36,2.75)
        item[2]=pos.tolist()
        item[3]=float(angle)
        item[4]=scale.tolist()
        widths[name]=native*scale
        data.append(dict(instance=name,semantic=q['semantic'],asset_source=actual.get('provider'),**stat))
    report={'instances':data,'fitted':sum(x['status']=='refined' for x in data),
            'heldout':sum(bool(x.get('holdout_count')) for x in data)}
    put_json(job/'PoseRefinement.json',report)
