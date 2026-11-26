"""Model only objects that have no accepted asset match."""
import copy
import numpy as np
from catalog import canonical_hash,fingerprint
from evidence import put_json,read_json
from assembly import rebuild
from planner import compact_recipe


def resolve(api,queries,choices,job):
    keys=[key for key,d in choices.items() if d['branch']=='BUILD' and d.get('defer_recipe')]
    audits=[]
    for i in range(0,len(keys),6):
        group=keys[i:i+6]
        payload=[dict(id=key,category=queries[key]['semantic'],dimensions_m=queries[key]['dimensions_m'],
                      description=queries[key]['description'],shape_parts=queries[key]['shape_parts']) for key in group]
        result=api.ask('recipes',{'unresolved':payload},tokens=4500)
        got={x['id']:x['parts'] for x in result.get('recipes',[])}
        for key in group:
            q=queries[key]
            e=choices[key]
            size=q['dimensions_m']
            try:
                items=compact_recipe(copy.deepcopy(got[key]),np.array(size))
                fixed,diag=rebuild(items,size)
            except (KeyError,ValueError,TypeError) as exc:
                fixed=None
                diag={'status':'invalid_recipe','reason':str(exc)}
            if not fixed:
                # A single connected proxy is preferable to visibly detached limbs or panels.
                fixed=[dict(name='ConnectedProxy',primitive='box',center=[0,0,size[2]/2],size=size,
                            rotation_deg=[0,0,0],material_role='body')]
                diag['fallback']='single_mesh_proxy'
                diag['native_dimensions_m']=size
            updates={'recipe':fixed,'native_dimensions_m':diag['native_dimensions_m']}
            q.update(updates)
            e.update(updates)
            e['procedural_hash']=fingerprint(q)
            e['asset_id']='AT_'+canonical_hash([e['procedural_hash'],e['material'],e['motion'],None])[:24]
            audits.append({'query_id':key,**diag})
    plan=read_json(job/'Retrieval.json')
    for row in plan:
        row['asset_id']=choices[row['query_id']]['asset_id']
    put_json(job/'Retrieval.json',plan)
    put_json(job/'AssemblyValidation.json',audits)
