"""Drop a physics probe at a room waypoint using NVIDIA ovphysx CPU."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import ovphysx
import ovstage
from ovphysx import PhysX
from ovphysx.population import physics_population_descs
from ovphysx.samples.python_samples.schema_support import physx_schema_paths
from ovphysx.types import TensorType


def preview_scene(case,folder):
    nav=json.loads((case/'Navigation.json').read_text())
    rooms=[r for r in nav['rooms'] if r['waypoint_m'] and r['margin_after_robot_radius_m']>.12]
    if not rooms: raise ValueError('No sufficiently clear waypoint for probe.')
    r=max(rooms,key=lambda r:r['margin_after_robot_radius_m'])
    xyz=r['waypoint_m']
    layered=Usd.Stage.CreateNew(str(Path(folder)/'ProbeScene.usda'))
    layered.GetRootLayer().subLayerPaths=[str((case/'Scene'/'Scene.usda').resolve())]
    UsdGeom.SetStageMetersPerUnit(layered,1)
    UsdGeom.SetStageUpAxis(layered,UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(layered,1)
    s=UsdGeom.Sphere.Define(layered,'/World/NavProbe')
    s.CreateRadiusAttr(.17)
    s.AddTranslateOp().Set(Gf.Vec3d(float(xyz[0]),float(xyz[1]),.63))
    UsdPhysics.RigidBodyAPI.Apply(s.GetPrim()).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(s.GetPrim()).CreateMassAttr(1.0)
    UsdPhysics.CollisionAPI.Apply(s.GetPrim()).CreateCollisionEnabledAttr(True)
    layered.GetRootLayer().Save()
    return Path(layered.GetRootLayer().realPath),r['id'],xyz


def run(case):
    ovstage.population.register_usd_schemas(
        [*[str(p) for p in physx_schema_paths()],str(ovphysx.newton_schema_root())])
    PhysX.set_cpu_mode(True)
    with TemporaryDirectory() as tmp:
        layer,room,xyz=preview_scene(case,tmp)
        stage=ovstage.Stage('robot-floor-probe')
        px=PhysX()
        attached=False
        try:
            ovstage.population.open_usd_with_desc(stage,str(layer),ordinal=1,time=ovstage.population.DEFAULT_TIME,
                                                 descs=physics_population_descs(rendering=False))
            stage.advance_write_floor(ordinal=1).wait()
            px.wait_task(px.attach_ovstage(stage,read_ordinal=1))
            attached=True
            for _ in range(120): px.step_sync(1/60)
            b=px.create_tensor_binding(pattern='/World/NavProbe',tensor_type=TensorType.RIGID_BODY_POSE)
            try:
                samples=np.zeros(b.shape,dtype=np.float32)
                b.read(samples)
                p=samples.reshape(-1,7)[0,:3].tolist()
            finally:
                b.destroy()
        finally:
            if attached: px.wait_task(px.detach_ovstage())
            stage.destroy()
            px.destroy()
    err=abs(p[2]-.17)
    if err>.045 or np.linalg.norm(np.array(p[:2])-xyz[:2])>.06:
        raise AssertionError(f'Floor probe failed contact at {p}')
    return dict(sdk='NVIDIA ovphysx CPU',version=ovphysx.__version__,steps=120,dt_seconds=1/60,waypoint_room=room,
                sphere_radius_m=.17,initial_z_m=.63,settled_position_m=p,floor_contact_error_m=err)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path)
    args=parser.parse_args()
    data=run(args.case)
    payload=json.dumps(data,indent=2)+'\n'
    (args.case/'PhysXFloorProbe.json').write_text(payload)
    print(payload)
