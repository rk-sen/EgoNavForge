"""Publish staged prefabs in the agent's native USD/Blender asset database."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
from catalog import Catalog,canonical_hash
from evidence import read_json,put_json

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('staging',type=Path)
    ap.add_argument('catalog',type=Path)
    args=ap.parse_args()
    bank=Catalog(args.catalog)
    rows=sorted(args.staging.glob('*/Intake.json'))
    known={r['asset_id'] for r in bank.entries()}
    def add(f):
        p=read_json(f)
        key=p['prototype_key']
        p['procedural_hash']=canonical_hash(['public_v1',key,p['input_sha256']])
        p['asset_id']='AT_'+p['procedural_hash'][:24]
        if p['asset_id'] in known: return p['asset_id']
        bank.publish(f.parent,key,p)
        print('published',key,flush=True)
        return p['asset_id']
    with ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(add,rows))
    entries=bank.entries()
    put_json(args.catalog/'PublicInventory.json',[{k:r.get(k) for k in
          ['asset_id','prototype_key','provider','license','source_page','source_file','input_sha256','semantic','dimensions_m','mesh_count']}
          for r in entries if r.get('public_prefab')])
    print(len(rows),'public prefabs addressed')

if __name__=='__main__': main()
