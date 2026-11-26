"""Versioned asset bank with transactional publication and explicit retrieval gates."""
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile

import numpy as np
from evidence import put_json, read_json, sha

GEOM_NEAR=.94
MAT_NEAR=.105
# This versions the stock mesh and material recipes. Camera/UI changes do not evict assets.
CODE_FINGERPRINT='8e2d4ecaa79424e77c8f3'


def canonical_hash(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class Catalog:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.db=self.root/'index.sqlite3'
        with closing(sqlite3.connect(self.db)) as conn:
            with conn:
                conn.execute('CREATE TABLE IF NOT EXISTS assets(id TEXT PRIMARY KEY, semantic TEXT NOT NULL, metadata TEXT NOT NULL)')
                conn.execute('CREATE INDEX IF NOT EXISTS semantic_lookup ON assets(semantic)')

    def directory(self,asset_id):
        if not re.fullmatch(r'AT_[a-f0-9]{24}',asset_id):
            raise ValueError('Malformed catalog asset ID.')
        return self.root/asset_id

    def entries(self):
        with closing(sqlite3.connect(self.db)) as conn:
            return [json.loads(row[0]) for row in conn.execute('SELECT metadata FROM assets ORDER BY id')]

    def geometry_matches(self,first,second):
        from physical import stored_topology
        if first['geometry_sha256']==second['geometry_sha256']:
            return True
        def digest(entry):
            return entry.get('geometry_topology_sha256') or stored_topology(
                self.directory(entry['asset_id']),entry['prototype_key'])
        return digest(first)==digest(second)

    def publish(self,staging,key,entry):
        from physical import export_asset, stored_topology
        final=self.directory(entry['asset_id'])
        with tempfile.TemporaryDirectory(prefix='_pending_',dir=self.root) as t:
            temp=Path(t)/'asset'
            prepared=export_asset(staging,key,entry,temp)
            with closing(sqlite3.connect(self.db,timeout=30)) as conn:
                conn.execute('BEGIN IMMEDIATE')
                existing=conn.execute('SELECT metadata FROM assets WHERE id=?',(entry['asset_id'],)).fetchone()
                if existing:
                    previous=json.loads(existing[0])
                    same_mesh=(previous['geometry_sha256']==prepared['geometry_sha256'] or
                               (previous.get('geometry_topology_sha256') or stored_topology(final,previous['prototype_key']))==
                               prepared['geometry_topology_sha256'])
                    if not same_mesh:
                        raise ValueError('Asset ID collision with different mesh contents.')
                    return previous
                if final.exists():
                    previous=read_json(final/'Metadata.json')
                    if (previous['geometry_sha256']!=prepared['geometry_sha256'] and
                          (previous.get('geometry_topology_sha256') or stored_topology(final,previous['prototype_key']))
                          !=prepared['geometry_topology_sha256']):
                        raise ValueError('Orphan catalog asset ID has incompatible geometry.')
                    prepared=previous
                else:
                    temp.rename(final)
                conn.execute('INSERT INTO assets VALUES(?,?,?)',(entry['asset_id'],entry['semantic'],
                            json.dumps(prepared,ensure_ascii=False,sort_keys=True)))
                conn.commit()
            return prepared


def geometrical_distance(query, candidate):
    d=np.sort(np.array(query['dimensions_m'],float))
    c=np.sort(np.array(candidate['dimensions_m'],float))
    return float(np.max(np.abs(np.log(d/c))))


def material_distance(first,second):
    if first is None or second is None:
        return 0. if first is None and second is None else 1.
    color=float(np.linalg.norm(np.array(first['color'])-second['color'])/np.sqrt(3))
    texture=.36 if first['texture']!=second['texture'] else 0.
    rough=abs(first['roughness']-second['roughness'])*.22
    metallic=abs(first['metallic']-second['metallic'])*.25
    return math.sqrt(color*color+texture*texture+rough*rough+metallic*metallic)


def token_overlap(q,c):
    tokens=lambda x:set(re.findall(r'[a-z]{3,}',(' '.join(x['shape_parts'])+' '+x['description']).lower()))
    a,b=tokens(q),tokens(c)
    return len(a&b)/max(1,len(a|b))


def fingerprint(q):
    if q.get('recipe'):
        return canonical_hash([q['semantic'],q['recipe'],q['dimensions_m']])
    return canonical_hash(['native',q.get('template') or q['query_id'],CODE_FINGERPRINT])


def prefab_distance(q,c):
    x=np.asarray(q.get('requested_dimensions_m',q['dimensions_m']))
    d=np.asarray(c['dimensions_m'])
    if q['semantic'] in {'monitor','laptop','television','keyboard','mouse'}:
        # Cropped front views constrain width and height better than stand depth.
        return float(np.linalg.norm(np.log([max(x[:2])/max(d[:2]),x[2]/d[2]]))/math.sqrt(2))
    p=np.log(x/d)
    swap=np.log(x[[1,0,2]]/d)
    return float(min(np.linalg.norm(p),np.linalg.norm(swap))/math.sqrt(3))


def scene_compatible(q,e):
    scene=q.get('scene_type') or 'generic'
    if scene not in {'generic','all'} and e.get('scene_types') and scene not in e['scene_types']:
        return False
    if q.get('semantic') in (e.get('negative_roles') or []):
        return False
    role=q.get('spatial_role')
    if role in {'wall','tabletop'} and e.get('spatial_roles') and role not in e['spatial_roles']:
        # wall/tabletop are hard physical roles; floor is the default for free-standing objects
        return False
    if e.get('instance_unit')=='compound_set' and not q.get('allow_compound'):
        return False
    return True


def find_candidates(q,items):
    rows=[]
    for e in items:
        if e['semantic']!=q['semantic']: continue
        if not scene_compatible(q,e): continue
        if e.get('public_prefab'):
            if q.get('prefer_public') and (q.get('public_required') or prefab_distance(q,e)<.74):
                rows.append(e)
        elif e['procedural_hash']==q['procedural_hash'] or geometrical_distance(q,e)<.23:
            rows.append(e)
    return sorted(rows,key=lambda x:(not x.get('public_prefab',False),q.get('procedural_hash')!=x.get('procedural_hash'),
                                     prefab_distance(q,x) if x.get('public_prefab') else geometrical_distance(q,x),
                                     material_distance(q['material'],x['material'])))[:4]


def describe(e):
    keys=('asset_id','query_id','semantic','dimensions_m','description','material','shape_parts',
          'procedural_hash','geometry_sha256','public_prefab','provider','observation_description',
          'requested_dimensions_m','scene_types','spatial_roles','negative_roles','instance_unit','material_family')
    return {k:e[k] for k in keys if k in e}


def dispatch(api,queries,crops,bank,job):
    items=bank.entries()
    for q in queries.values():
        q['procedural_hash']=fingerprint(q)
    candidates={key:find_candidates(q,items) for key,q in queries.items()}
    by_id={x['asset_id']:x for x in items}
    raw={}
    # No call is required when the complete request has no geometric candidates.
    eligible=[key for key,rows in candidates.items() if rows]
    for offset in range(0,0 if api is None else len(eligible),8):
        keys=eligible[offset:offset+8]
        imgs=[]
        for key in keys:
            if key in crops:
                imgs.append((f'Query crop {key}',crops[key]))
            for row in candidates[key][:1]:
                thumb=bank.directory(row['asset_id'])/'Thumbnail.jpg'
                if thumb.exists():
                    imgs.append((f'Catalog thumbnail for {key}: {row["asset_id"]}',thumb))
        desire=[dict(query=describe(queries[key]),candidates=[describe(x) for x in candidates[key]]) for key in keys]
        r=api.ask('match',{'requests':desire,'threshold':GEOM_NEAR},imgs,tokens=3800)
        raw.update({x['query_id']:x for x in r.get('decisions',[]) if x['query_id'] in keys})
    if api is None:
        raw={}
    dispatch={}
    logs=[]
    for key,q in queries.items():
        proposal=raw.get(key,{})
        candidate=by_id.get(proposal.get('asset_id'))
        same=candidates[key]
        geom=[r for r in same if r['procedural_hash']==q['procedural_hash']]
        exact=[r for r in geom if material_distance(q['material'],r['material'])<=MAT_NEAR]
        prefabs=[r for r in same if r.get('public_prefab')]
        if q['semantic']=='desk' and prefabs:
            # Video desks are often distinguished by finish: a white laminate
            # desk must not lose to a closer-size honey-wood desk.
            prefabs=sorted(prefabs,key=lambda x:(material_distance(q['material'],x['material']),
                                                 prefab_distance(q,x)))
        public_candidate=candidate if candidate in prefabs else None
        fixed_semantics={'monitor','keyboard','mouse','laptop','teddy_bear','toy_blocks','sport_ball','microwave','washer','fridge'}
        if q.get('public_required'):
            if not prefabs:
                raise ValueError(f'No imported human prefab for {key}; run the library intake first.')
            source=next((x for x in prefabs if x.get('preferred_pose')),prefabs[0])
            action='REUSE'
            reason='human classification uses the downloaded rigged-source idle-pose asset'
        elif public_candidate and proposal.get('action') in {'REUSE','MATERIAL_VARIANT'} and (
             float(proposal.get('geometry_similarity',0))>=.72 and prefab_distance(q,public_candidate)<.60):
            source=public_candidate
            action='REUSE'
            reason='public match passed silhouette review and metric size gate'
        elif prefabs and q['semantic'] in fixed_semantics and prefab_distance(q,prefabs[0])<.53:
            source=prefabs[0]
            action='REUSE'
            reason='strong canonical subtype with compatible measured bounds'
        elif prefabs and prefab_distance(q,prefabs[0])<.53:
            source=prefabs[0]
            action='REUSE'
            reason='offline cognitive gate: scene/role/material compatible and metric size within 0.53 log-distance'
        elif exact:
            source=exact[0]
            action='REUSE'
            reason='identical procedural fingerprint and compatible visual finish'
        elif geom and q['material'] is not None:
            source=geom[0]
            action='MATERIAL_VARIANT'
            reason='identical procedural fingerprint and distinct visual finish'
        elif candidate and candidate in same and proposal.get('action') in {'REUSE','MATERIAL_VARIANT'} and (
            float(proposal.get('geometry_similarity',0))>=GEOM_NEAR and
            geometrical_distance(q,candidate)<.12 and token_overlap(q,candidate)>=.38):
            source=candidate
            action='REUSE' if material_distance(q['material'],source['material'])<=MAT_NEAR else 'MATERIAL_VARIANT'
            reason='visual judgment passed independent metric-size and topology gates'
        else:
            source=None
            action='BUILD'
            reason='no candidate passed geometry gates and requested finish policy'
        if action=='MATERIAL_VARIANT' and not q.get('replace_materials'):
            q['replace_materials']=source.get('replace_materials',[])
            if not q['replace_materials']:
                # New procedural meshes have one body material. Keep structural finishes intact.
                q['replace_materials']=[name for name in source['material_definitions'] if name.startswith('finish_')]
            if not q['replace_materials']:
                source=None
                action='BUILD'
                reason='no editable compatible surface role in the proposed original'
        entry={**q,'branch':action}
        if source:
            entry.update(source_asset_id=source['asset_id'],source_key=source['prototype_key'],
                         blend_path=str(bank.directory(source['asset_id'])/'Prototype.blend'),
                         material_definitions=source['material_definitions'])
        entry['asset_id']=source['asset_id'] if action=='REUSE' else 'AT_'+canonical_hash([
            q['procedural_hash'],q['material'],q['motion'],
            source['geometry_sha256'] if source else None])[:24]
        logs.append(dict(query_id=key,action=action,asset_id=entry['asset_id'],
                         source_asset_id=source['asset_id'] if source else None,resolution=reason,
                         candidate_ids=[x['asset_id'] for x in same],model_proposal=proposal,
                         source_provider=source.get('provider') if source else None,
                         public_prefab=bool(source and source.get('public_prefab'))))
        dispatch[key]=entry
    put_json(job/'Retrieval.json',logs)
    return dispatch


def copy_reused_textures(dispatch,bank,to):
    texture_dir=to/'Textures'
    texture_dir.mkdir(parents=True,exist_ok=True)
    for entry in dispatch.values():
        if entry['branch']=='BUILD': continue
        root=bank.directory(entry['source_asset_id'])/'Textures'
        if not root.exists(): continue
        for f in root.glob('*.png'):
            p=texture_dir/f.name
            if p.exists() and sha(f)!=sha(p):
                raise ValueError(f'Catalog texture collision: {f.name}')
            if not p.exists():
                shutil.copyfile(f,p)
