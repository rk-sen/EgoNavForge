"""Independently verify scene USD camera projection using its aperture and focal length."""
import argparse
import json
from pathlib import Path
import numpy as np
from pxr import Gf,Usd,UsdGeom


def check(stage_file,intrinsics):
    stage=Usd.Stage.Open(str(stage_file))
    k=np.array(intrinsics['K'])
    w,h=intrinsics['width'],intrinsics['height']
    cam=UsdGeom.Camera.Get(stage,'/World/Cameras/TrajectoryCam')
    mat=cam.GetCamera().frustum.ComputeProjectionMatrix()
    pts=[(-.36,-.4,-2),(0,0,-2),(1,.1,-3),(.089,.22,-2),(.6,-.7,-5)]
    errors=[]
    for x,y,z in pts:
        q=mat.Transform(Gf.Vec3d(x,y,z))
        u,v,_=k@np.array([x,-y,-z])/(-z)
        errors.append(float(np.linalg.norm([u-(1+q[0])*w/2,v-(1-q[1])*h/2])))
    if max(errors)>1e-3:
        raise AssertionError(f'USD camera differs from K by {max(errors):.6f}px')
    return {'usd_intrinsics_max_arithmetic_pixel_error':max(errors),'checks':len(pts)}


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage',type=Path)
    cli.add_argument('intrinsics',type=Path)
    opts=cli.parse_args()
    print(json.dumps(check(opts.stage,json.loads(opts.intrinsics.read_text())),indent=2))
