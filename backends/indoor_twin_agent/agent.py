"""Run the Kimi observation, retrieval, Blender build, review and USD publication loop."""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from catalog import Catalog, copy_reused_textures, dispatch
from evidence import prepare, put_json, ray_pose, read_json
from kimi import Kimi, ORIGIN, MODEL
from planner import general_plan, observe, query_crops, seeded_plan
from layout import align_evidence
try:
    from navigation import audit
except Exception:
    audit=None
from novel_assets import resolve
from placement import refine_scene

ROOT=Path(__file__).resolve().parent


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['video','trajectory','intrinsics','pointcloud','output','catalog','survey']:
        p.add_argument('--'+arg,type=Path,required=arg in {'video','trajectory','intrinsics','output','catalog'})
    p.add_argument('--endpoint',default=ORIGIN)
    p.add_argument('--model',default=MODEL)
    p.add_argument('--blender',default='blender')
    p.add_argument('--frames',type=int,default=10)
    p.add_argument('--samples',type=int,default=52)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--camera-height',type=float)
    p.add_argument('--nav-radius',type=float,default=.27)
    p.add_argument('--resume-observations',action='store_true')
    p.add_argument('--skip-review',action='store_true')
    p.add_argument('--finish-override',type=Path)
    p.add_argument('--scene-type',default='auto')
    p.add_argument('--api-key',default=None)
    p.add_argument('--offline-retrieval',action='store_true')
    a=p.parse_args()
    if a.frames<2 or a.samples<4 or a.threads<1 or not .10<=a.nav_radius<=.70 or a.camera_height is not None and a.camera_height<=0:
        p.error('Invalid frame/sample/thread budget or camera height.')
    for arg in ['video','trajectory','intrinsics','pointcloud','output','catalog','survey','finish_override']:
        path=getattr(a,arg,None)
        if path is not None: setattr(a,arg,path.resolve())
    return a


def blender(a,prepared,folder,survey,decisions):
    folder.mkdir(parents=True,exist_ok=True)
    put_json(folder/'ScenePlan.json',survey)
    put_json(folder/'Dispatch.json',decisions)
    log=folder/'Blender.log'
    cmd=[a.blender,'-b','-t',str(a.threads),'--python-exit-code','4','--python',str(ROOT/'build_twin.py'),
         '--','--input',str(prepared),'--output',str(folder),'--survey',str(folder/'ScenePlan.json'),
         '--dispatch',str(folder/'Dispatch.json'),'--samples',str(a.samples),'--preview']
    with log.open('wb') as f:
        result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode or not (folder/'IndoorTwin.blend').exists():
        raise RuntimeError(f'Blender failed. Examine {log}.')


def reviewer(api,ev,plan,folder):
    usable=[]
    imgs=[]
    for n in plan['preview_indices']:
        f=folder/'Previews'/f'video_{n:03}.png'
        if n not in ev['photos'] or not f.exists(): continue
        usable.append(n)
        compressed=folder.parent/'Evidence'/f'{folder.name}_review_{n:06}.jpg'
        Image.open(f).convert('RGB').save(compressed,quality=86)
        imgs.extend([(f'Original video frame {n}',ev['photos'][n]),(f'Same-camera Blender render {n}',compressed)])
    locations=[{'id':i[0],'category':i[1],'position_m':i[2],'yaw_deg':i[3]} for i in plan['instances']]
    return api.ask('review',{'frame_pairs':usable,'placements':locations},imgs,3900)


def ground_error(p,views,ev):
    err={}
    for view in views:
        i=view.get('index')
        uv=view.get('ground')
        if i not in ev['photos'] or not view.get('ground_visible') or not uv: continue
        pos,R=ray_pose(ev['track'],ev['pts'][i],ev['R'],ev['origin'])
        local=(np.array(p)-pos)@R*[1,-1,-1]
        if local[2]<.1: continue
        pix=ev['K']@local
        error=np.linalg.norm(pix[:2]/pix[2]-np.array(uv)*[ev['W'],ev['H']])
        err[i]=float(error)
    return err


def propose(plan,review,queries,e,frozen):
    p=copy.deepcopy(plan)
    by_id={row[0]:row for row in p['instances']}
    updates=[]
    used=set()
    for item in review.get('patches',[])[:8]:
        key=item.get('id')
        row=by_id.get(key)
        if row is None or key in used: continue
        used.add(key)
        q=queries[row[1]]
        frames=set(item.get('evidence_frames',[]))
        if frozen or row[5] in {'inferred_context','tabletop_support_estimate'} or q['category'] in {'Architecture','Lighting'} or len(frames)<2:
            updates.append({'id':key,'accepted':False,'reason':'locked experiment or insufficient evidence'})
            continue
        delta=np.asarray(item.get('delta_m'),dtype=float)
        yaw=float(item.get('delta_yaw_deg',0))
        if delta.shape!=(3,) or not np.isfinite(delta).all() or not math.isfinite(yaw): continue
        # The validator can prove a supported floor contact update, but lacks shape
        # landmarks for yaw. Yaw remains unchanged until a separate fit exists.
        if np.linalg.norm(delta)>.42 or abs(delta[2])>.015: continue
        old=ground_error(row[2],q.get('views',[]),e)
        point=(np.array(row[2])+delta).tolist()
        new=ground_error(point,q.get('views',[]),e)
        support=set(old)&set(new)&frames
        if len(support)<2 or any(new[n]>old[n]+2 for n in support) or sum(new[n] for n in support)>.9*sum(old[n] for n in support):
            updates.append({'id':key,'accepted':False,'reason':'ground-contact reprojection did not improve'})
            continue
        row[2]=point
        updates.append({'id':key,'accepted':True,'reason':item.get('reason'),'original_px':old,'updated_px':new})
    return p,updates


def override_finish(plan,queries,overrides):
    for i in plan['instances']:
        if i[0] not in overrides: continue
        spec=overrides[i[0]]
        original=queries[i[1]]
        key=spec['query_id']
        edited=dict(original,query_id=key,template=original.get('template') or i[1],
                    material=spec['material'],replace_materials=spec['replace_materials'])
        queries[key]=edited
        i[1]=key
    used={i[1] for i in plan['instances']}
    used|={'FabricBarrier'} if plan['barrier_runs'] else set()
    used|={'StripLuminaire'} if plan['light_strips'] else set()
    used|={'WindowBay'} if plan['window_centres'] else set()
    return {k:v for k,v in queries.items() if k in used}


def run(a):
    start=time.monotonic()
    job=a.output
    scene_type=a.scene_type
    if scene_type=='auto':
        parts=[p.lower() for p in job.parts]
        scene_type=next((x for x in ('lab','household','large') if x in parts), 'office')
    if a.api_key:
        import os
        os.environ['KIMI_API_KEY']=a.api_key
    job.mkdir(parents=True,exist_ok=True)
    scene_dir=job/'Scene'
    if (scene_dir/'Scene.usda').exists():
        raise ValueError('Use a new output directory for a new scene transaction.')
    seed=read_json(a.survey) if a.survey else None
    ev=prepare(a,job,seed)
    if not seed:
        align_evidence(ev,job)
    kimi=Kimi(job/'ModelLogs',a.endpoint,a.model)
    if a.resume_observations:
        if not (job/'Observations.json').exists() or not list((job/'ModelLogs').glob('*observe.response.json')):
            raise ValueError('No real original observation responses exist for this job.')
        observed=[obj for batch in read_json(job/'Observations.json') for obj in batch.get('objects',[])]
    else:
        observed=observe(kimi,ev,job,seed)
    if seed:
        seed=dict(seed,scene_type=scene_type)
        plan,queries=seeded_plan(ev,seed,observed)
    else:
        plan,queries=general_plan(None if a.offline_retrieval else kimi,ev,observed,job,scene_type=scene_type)
    if a.finish_override:
        queries=override_finish(plan,queries,read_json(a.finish_override))
    put_json(job/'AssetQueries.json',queries)
    bank=Catalog(a.catalog)
    crops=query_crops(queries,ev,job)
    choices=dispatch(None if a.offline_retrieval else kimi,queries,crops,bank,job)
    if not a.offline_retrieval:
        resolve(kimi,queries,choices,job)
    else:
        # Deterministic geometry fallback for an uncatalogued object: a single
        # box with the measured bounding size.  It is still a real scene mesh.
        for choice in choices.values():
            if choice['branch']=='BUILD' and not choice.get('template') and not choice.get('recipe'):
                d=[float(x) for x in choice.get('dimensions_m',[.5,.5,.6])]
                choice['recipe']=[dict(name='MeasuredBox',primitive='box',center=[0,0,d[2]/2],
                                       size=d,rotation_deg=[0,0,0],material_role='body')]
                choice['description']=choice.get('description','Measured uncatalogued object as a single box')
    if not seed:
        refine_scene(plan,queries,choices,bank,ev,job)
    copy_reused_textures(choices,bank,scene_dir)
    blender(a,ev['prepared'],scene_dir,plan,choices)
    review=None
    accepted=0
    if not a.skip_review:
        review=reviewer(kimi,ev,plan,scene_dir)
        edited,actions=propose(plan,review,queries,ev,bool(seed) or bool(plan.get('placement_observations')))
        put_json(job/'Review.json',{'review':review,'patch_gate':actions})
        if any(x['accepted'] for x in actions):
            refined=job/'CandidateScene'
            copy_reused_textures(choices,bank,refined)
            blender(a,ev['prepared'],refined,edited,choices)
            confirmed=reviewer(kimi,ev,edited,refined)
            put_json(job/'CandidateReview.json',confirmed)
            if confirmed.get('score',0)>=review.get('score',0):
                scene_dir.rename(job/'InitialScene')
                refined.rename(scene_dir)
                plan=edited
                accepted=sum(x['accepted'] for x in actions)
    manifest=read_json(scene_dir/'AssetManifest.json')
    validation=read_json(scene_dir/'AssemblyValidation.json')
    # Scene.usda is serialized directly from the evaluated Blender scene by
    # build_twin.py.  No secondary catalog transaction can drop textures or
    # rewrite instance transforms after placement.
    navigation=audit(job,ev,a.nav_radius) if (audit is not None and not seed) else None
    variants={}
    retrieval=read_json(job/'Retrieval.json')
    summary=dict(public_prefab_queries=sum(x.get('public_prefab',False) for x in retrieval),
                 branch_counts=dict(Counter(r['branch'] for r in choices.values())),source_survey=bool(seed),
                 independently_observed=True,mode='controlled_pose' if seed else 'unsurveyed',
                 model_call_count=len(list((job/'ModelLogs').glob('*.response.json'))),
                 library_total=len(bank.entries()),observed_instances=len(observed),
                 placement_patches_applied=accepted,variants=variants,physics=validation,
                 navigation_summary={k:v for k,v in navigation.items() if k not in
                    {'candidate_landmarks','exterior_probes_by_wall','rooms','portals'}} if navigation else None,
                 elapsed_seconds=round(time.monotonic()-start,3))
    put_json(job/'RunSummary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    run(arguments())
