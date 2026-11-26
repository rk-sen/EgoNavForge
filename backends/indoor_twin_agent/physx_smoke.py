"""Run CPU PhysX against the composed scene via NVIDIA ovphysx."""
import argparse
import json
from pathlib import Path
import numpy as np
import ovstage
import ovphysx
from ovphysx import PhysX
from ovphysx.samples.python_samples.schema_support import physx_schema_paths
from ovphysx.population import physics_population_descs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('scene',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--steps',type=int,default=120)
    a=p.parse_args()
    if a.steps<1: p.error('--steps must be positive')
    ovstage.population.register_usd_schemas(
        [*[str(p) for p in physx_schema_paths()],str(ovphysx.newton_schema_root())])
    stage=ovstage.Stage('indoor-twin-scene')
    PhysX.set_cpu_mode(True)
    px=PhysX()
    attached=False
    try:
        report=ovstage.population.open_usd_with_desc(
            stage,str(a.scene.resolve()),ordinal=1,time=ovstage.population.DEFAULT_TIME,
            descs=physics_population_descs(rendering=False))
        stage.advance_write_floor(ordinal=1).wait()
        px.wait_task(px.attach_ovstage(stage,read_ordinal=1))
        attached=True
        for _ in range(a.steps):
            px.step_sync(1/60.)
        from ovphysx.types import TensorType
        from pxr import Usd, UsdPhysics
        metadata_stage=Usd.Stage.Open(str(a.scene.resolve()))
        dynamic=[str(p.GetPath()) for p in metadata_stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        positions={}
        for path in dynamic:
            binding=px.create_tensor_binding(pattern=path,tensor_type=TensorType.RIGID_BODY_POSE)
            try:
                values=np.zeros(binding.shape,dtype=np.float32)
                binding.read(values)
                positions[path]=values.reshape(-1,7)[0,:3].tolist()
            finally:
                binding.destroy()
        if not positions or any(not np.isfinite(value).all() for value in positions.values()):
            raise RuntimeError('Missing or nonfinite dynamic body poses.')
        result={'engine':'NVIDIA ovphysx','version':ovphysx.__version__,'scene':a.scene.name,
                'physics_steps':a.steps,'dt_seconds':1/60.,'finite_dynamic_positions':True,
                'dynamic_bodies':len(positions),'positions_m_by_prim':positions}
        if '/World/Furniture/ExerciseBall_01' in positions:
            z=positions['/World/Furniture/ExerciseBall_01'][2]
            result['exercise_ball_root_floor_offset_m']=z
            if abs(z)>.08:
                raise RuntimeError('Exercise ball fell through the floor or left the expected contact height.')
        if a.output:
            a.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        if attached: px.wait_task(px.detach_ovstage())
        stage.destroy()
        px.destroy()


if __name__=='__main__':
    main()
