"""Test controlled BUILD / REUSE parity and material-variant invariants."""
import argparse
import json
from pathlib import Path

from PIL import Image
import numpy as np
from pxr import Usd


def get(path):
    return json.loads(path.read_text(encoding='utf8'))


def compare(root):
    root=Path(root)
    base,repeat,variant=(root/x for x in ('cold','warm','variant'))
    cold=get(base/'RunSummary.json')
    warm=get(repeat/'RunSummary.json')
    assert cold['branch_counts']=={'BUILD':24}
    assert warm['branch_counts']=={'REUSE':24}
    assert cold['physics']==warm['physics']
    cold_obs=get(base/'ModelLogs/001_observe.response.json')['id']
    warm_obs=get(repeat/'ModelLogs/001_observe.response.json')['id']
    assert cold_obs!=warm_obs
    shots=[]
    for p in sorted((base/'Scene/Previews').glob('video_*.png')):
        a=np.asarray(Image.open(p).convert('RGB'))
        b=np.asarray(Image.open(repeat/'Scene/Previews'/p.name).convert('RGB'))
        assert a.shape==b.shape and np.array_equal(a,b),p
        shots.append({'file':p.name,'shape':list(a.shape),'rgb_max_abs_difference':int(np.max(abs(a.astype(int)-b)))})
    assert len(shots)==4
    ic=get(base/'Scene/SimulationManifest.json')['instances']
    iw=get(repeat/'Scene/SimulationManifest.json')['instances']
    assert all(a['asset_id']==b['asset_id'] and a['instance_id']==b['instance_id'] and
               np.array_equal(a['matrix_world_row_major'],b['matrix_world_row_major']) for a,b in zip(ic,iw,strict=True))
    sv=get(variant/'RunSummary.json')
    assert sv['branch_counts']=={'REUSE':23,'MATERIAL_VARIANT':1}
    shape=sv['variants']['ExerciseBall_Blue']
    assert shape['same_mesh_sha256']
    assert shape['source_materials']!=shape['variant_materials']
    iv=get(variant/'Scene/SimulationManifest.json')['instances']
    for a,b in zip(ic,iv,strict=True):
        assert a['instance_id']==b['instance_id'] and a['physical_profile']==b['physical_profile']
        assert np.array_equal(a['matrix_world_row_major'],b['matrix_world_row_major'])
    simulations={}
    for p in (base,repeat,variant):
        stage=Usd.Stage.Open(str(p/'Scene/Scene.usda'))
        assert not stage.GetCompositionErrors()
        report=get(p/'PhysXSmoke.json')
        assert report['physics_steps']==120 and report['dynamic_bodies']==12
        assert abs(report['exercise_ball_root_floor_offset_m'])<.01
        simulations[p.name]=report['exercise_ball_root_floor_offset_m']
    return dict(cold_build=24,warm_reuse=24,material_variants=1,scene_instances=len(ic),
                distinctKimiObservationIDs=[cold_obs,warm_obs],
                controlled_shots=shots,identical_instance_matrices=True,variant=shape,
                physx_ball_root_z_m=simulations)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('runs',type=Path)
    cli.add_argument('--output',type=Path)
    opts=cli.parse_args()
    data=json.dumps(compare(opts.runs),indent=2)
    print(data)
    if opts.output:
        opts.output.write_text(data+'\n',encoding='utf8')
