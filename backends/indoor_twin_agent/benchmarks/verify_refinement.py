"""Read final manifests and verify that library assembly is composed as one instance root."""
from collections import Counter
from pathlib import Path
import argparse,json
import numpy as np
from pxr import Usd,UsdGeom

def load(p):return json.loads(p.read_text(encoding='utf8'))

def verify(case,bank):
    manifest=load(case/'Scene/SimulationManifest.json')
    stage=Usd.Stage.Open(str(case/'Scene/Scene.usda'))
    assert stage and not stage.GetCompositionErrors()
    rows=[]
    for item in manifest['instances']:
        p=stage.GetPrimAtPath(item['usd_prim_path'])
        model=p.GetChild('Visual')
        assert model and model.IsInstance(),p.GetPath()
        proto=model.GetPrototype()
        meshes=[g for g in Usd.PrimRange(proto) if g.IsA(UsdGeom.Mesh)]
        assert meshes,p.GetPath()
        meta=load(bank/item['asset_id']/'Metadata.json')
        if meta.get('public_prefab'):
            assert len(meshes)==meta['mesh_count'],(p.GetPath(),len(meshes),meta['mesh_count'])
        rows.append(dict(instance=item['instance_id'],prototype=meta.get('prototype_key'),
                         semantic=meta['semantic'],provider=meta.get('provider'),
                         mesh_parts=len(meshes),root=str(p.GetPath())))
    poses=load(case/'PoseRefinement.json')['instances']
    stats=[r for r in poses if 'selected_envelope_iou' in r]
    held=[r for r in stats if r.get('holdout_count')]
    selection=lambda st,key:float(np.mean([x[key] for x in st])) if st else None
    build=load(case/'AssemblyValidation.json')
    return dict(instances=len(rows),prefab_instances=sum(bool(r['provider']) for r in rows),
                prefab_providers=dict(Counter(r['provider'] for r in rows if r['provider'])),
                multi_mesh_prefab_instances=sum(bool(r['provider']) and r['mesh_parts']>1 for r in rows),
                people=[r for r in rows if r['semantic']=='person'],
                refined=sum(r['status']=='refined' for r in poses),evaluated=len(stats),
                fit_iou_before=selection(stats,'initial_envelope_iou'),
                fit_iou_selected=selection(stats,'selected_envelope_iou'),
                heldout_count=len(held),heldout_iou_before=selection(held,'initial_envelope_iou'),
                heldout_iou_selected=selection(held,'selected_envelope_iou'),
                accepted_translation_mean_m=selection([r for r in stats if r['status']=='refined'],'translation_delta_m'),
                support_objects=sum(bool(r.get('support_instance')) for r in poses),
                recipes=len(build),repaired=sum(bool(r.get('snaps')) for r in build),
                single_mesh_proxy=sum(bool(r.get('fallback')) for r in build),objects=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('case',type=Path)
    p.add_argument('bank',type=Path)
    a=p.parse_args()
    print(json.dumps(verify(a.case,a.bank),indent=2))
