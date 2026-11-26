"""Download curated CC0 packs and verify Poly Haven file digests."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,json
from pathlib import Path
from urllib.request import Request,urlopen
import zipfile
import argparse

HERE=Path(__file__).resolve().parent
HEADERS={'User-Agent':'IndoorTwin-Research/1.0 (Powered by Poly Haven)'}
EXTRA_URLS={
    'curtain-panels-on-a-rod-fa3bf80c':
        'https://files.innerscene.com/library/curtain-panels-on-a-rod-fa3bf80c/56047dae8237e9b74553/curtain-panels-on-a-rod.glb',
    'basketball-3c93d030':
        'https://files.innerscene.com/library/basketball-3c93d030/bf33964b0fb04ad8379c/basketball.glb',
}
PACKS={
    'kenney': 'https://opengameart.org/sites/default/files/kenney_furniturePack.zip',
    'human': 'https://opengameart.org/sites/default/files/kenney_animated-characters-1.zip'
}

def get(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers=HEADERS),timeout=35) as r:
                return r.read()
        except OSError:
            if attempt==2: raise

def fetch_one(root,key):
    config=json.loads(get(f'https://api.polyhaven.com/files/{key}'))
    f=config['gltf']['1k']['gltf']
    out=root/'polyhaven'/key
    out.mkdir(parents=True,exist_ok=True)
    payloads={f['url'].split('/')[-1]:f,**f.get('include',{})}
    for relative,attrs in payloads.items():
        path=out/relative
        path.parent.mkdir(exist_ok=True,parents=True)
        digest=attrs.get('md5')
        raw=path.read_bytes() if path.exists() else get(attrs['url'])
        if digest and hashlib.md5(raw).hexdigest()!=digest:
            raise ValueError(f'MD5 mismatch at {path}.')
        if not path.exists(): path.write_bytes(raw)
    return key

def main():
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    args=p.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    assets=json.loads((HERE/'asset_sources.json').read_text())
    for key,url in PACKS.items():
        d=out/key
        if d.exists() and any(d.iterdir()): continue
        raw=get(url)
        file=out/(key+'.zip')
        file.write_bytes(raw)
        with zipfile.ZipFile(file) as z:
            for name in z.namelist():
                path=(d/name).resolve()
                if not path.is_relative_to(d): raise ValueError('Invalid bundled path')
            z.extractall(d)
    for name in assets.get('innerscene',{}):
        folder=out/'innerscene'/name
        folder.mkdir(exist_ok=True,parents=True)
        f=folder/'model.glb'
        if not f.exists(): f.write_bytes(get(EXTRA_URLS[name]))
    failures=[]
    def task(key):
        try: print('ok',fetch_one(out,key),flush=True)
        except Exception as e: failures.append((key,str(e))); print(key,e,flush=True)
    with ThreadPoolExecutor(max_workers=10) as pool: list(pool.map(task,assets['polyhaven']))
    (out/'download_failures.json').write_text(json.dumps(failures,indent=2))
    if failures: raise SystemExit(2)

if __name__=='__main__': main()
