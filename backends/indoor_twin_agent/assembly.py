"""Validate and anchor the connected assembly of a novel procedural object."""
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation

CORNERS=np.asarray(list(product((-.5,.5),repeat=3)))


def boxes(parts):
    result=[]
    for p in parts:
        r=Rotation.from_euler('xyz',p['rotation_deg'],degrees=True)
        q=r.apply(CORNERS*np.asarray(p['size']))+p['center']
        result.append((q.min(axis=0),q.max(axis=0)))
    return np.array(result)


def distances(a,b):
    return np.maximum(np.maximum(a[0]-b[1],b[0]-a[1]),0)


def rebuild(parts,dims):
    dims=np.asarray(dims,float)
    volumes=np.asarray([np.prod(x['size']) for x in parts])
    tolerance=max(.007,min(.028,np.max(dims)*.019))
    maximum_repair=max(.025,min(.09,np.max(dims)*.11))
    anchored=[int(np.argmax(volumes))]
    remaining=set(range(len(parts)))-set(anchored)
    snaps=[]
    while remaining:
        ext=boxes(parts)
        choices=[(float(np.linalg.norm(distances(ext[a],ext[b]))),a,b) for a in anchored for b in remaining]
        gap,a,b=min(choices)
        if gap>maximum_repair:
            return None,{'status':'unsound_recipe','unconnected_parts':[parts[x]['name'] for x in sorted(remaining)],
                         'separation_m':gap}
        if gap>tolerance:
            lo,hi=ext[b]
            al,ah=ext[a]
            shift=np.where(lo>ah,ah-lo+tolerance/3,np.where(hi<al,al-hi-tolerance/3,0))
            parts[b]['center']=(np.asarray(parts[b]['center'])+shift).tolist()
            snaps.append({'part':parts[b]['name'],'gap_m':gap,'shift_m':shift.tolist()})
        anchored.append(b)
        remaining.remove(b)
    ext=boxes(parts)
    lo=ext[:,0].min(0);hi=ext[:,1].max(0)
    offset=np.r_[(lo[:2]+hi[:2])/2,lo[2]]
    for p in parts: p['center']=(np.asarray(p['center'])-offset).tolist()
    return parts,{'status':'connected','components':1,'snaps':snaps,'native_dimensions_m':(hi-lo).tolist()}
