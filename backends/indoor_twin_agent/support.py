"""Use co-visible furniture and calibrated rays to anchor tabletop objects."""
import numpy as np
from placement import poses_for_views


def bounds(view):
    b=np.asarray(view.get('bbox',[]),float)
    if b.shape!=(4,) or not np.isfinite(b).all() or np.any(b[2:]<=b[:2]):
        return None
    return b


def ray_to_z(ev,index,xy,height):
    V=poses_for_views(ev)
    if index not in V:return None
    center,rot=V[index]
    u=np.linalg.solve(ev['K'],np.r_[xy*np.array([ev['W'],ev['H']]),1])
    direction=rot@(u*[1,-1,-1])
    if abs(direction[2])<.035:return None
    depth=(height-center[2])/direction[2]
    if not .14<=depth<=10.:return None
    return center+direction*depth


def lift_tabletop(ev,object,height=.81):
    result=[]
    for view in object.get('views',[]):
        b=bounds(view)
        if b is None:continue
        index=view.get('index')
        foot=np.asarray(view.get('ground',[]),float)
        if foot.shape!=(2,) or not view.get('ground_visible',False):
            foot=np.array([(b[0]+b[2])/2,b[3]-.04*(b[3]-b[1])])
        p=ray_to_z(ev,index,foot,height)
        if p is not None:
            result.append((index,p))
    if not result:return None
    return np.median([x[1] for x in result],axis=0),0.,[x[0] for x in result]


def bind_to_furniture(ev,child,furniture):
    scores=[]
    for table in furniture:
        center=table['xyz']
        dims=table['dimensions']
        delta=float(np.linalg.norm(child['xyz'][:2]-center[:2]))
        table_views={v['index']:bounds(v) for v in table['object'].get('views',[])}
        candidates=[]
        for image in child['object'].get('views',[]):
            idx=image['index']
            b=bounds(image)
            t=table_views.get(idx)
            if b is None or t is None:continue
            mid=(b[0]+b[2])/2
            horizontal=t[0]-.07<mid<t[2]+.07
            # Bottom of the tabletop item is approximately at the upper part of the table bbox.
            top=np.clip(b[3],0,1)
            vertical=t[1]-.15<top<t[1]+.60*(t[3]-t[1])+.07
            if not horizontal or not vertical:continue
            p=ray_to_z(ev,idx,np.array([mid,top]),center[2]+dims[2])
            if p is not None and np.linalg.norm(p[:2]-center[:2])<max(.63,max(dims[:2])*.95):
                candidates.append(p)
        gate=max(.42,min(1.10,float(max(dims[:2]))*.60))
        if candidates:
            est=np.median(candidates,axis=0)
            scores.append((delta-.65-.15*len(candidates),table,est,'co_visible_desk_surface'))
        elif delta<gate:
            scores.append((delta,table,child['xyz'].copy(),'nearby_surface'))
    if not scores:return None
    _,table,pos,mode=min(scores,key=lambda row:row[0])
    half=max(.23,min(table['dimensions'][:2])*.64)
    pos[:2]=np.clip(pos[:2],table['xyz'][:2]-half,table['xyz'][:2]+half)
    pos[2]=table['xyz'][2]+table['dimensions'][2]-.006
    return table['name'],pos,mode
