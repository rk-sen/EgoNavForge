"""The only model adapter. Use selected JPEGs and JSON over OpenAI-compatible HTTPS."""
import base64
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

MODEL = 'kimi-k2.6'
ORIGIN = 'https://ai.zj-computility.com/maas/v1'
PROMPTS = Path(__file__).with_name('prompts')


class Kimi:
    def __init__(self, logs, endpoint=ORIGIN, model=MODEL, key=None):
        self.key = key or os.environ.get('KIMI_API_KEY')
        if not self.key:
            raise ValueError('Set KIMI_API_KEY in the process environment.')
        if urlsplit(endpoint).scheme != 'https':
            raise ValueError('An HTTPS API endpoint is required.')
        self.url = endpoint.rstrip('/')+'/chat/completions'
        self.model = model
        self.logs = Path(logs)
        self.logs.mkdir(parents=True, exist_ok=True)

    def ask(self, step, payload, images=(), tokens=3600):
        system = (PROMPTS/'system.txt').read_text(encoding='utf8')
        task = (PROMPTS/f'{step}.txt').read_text(encoding='utf8')
        content = [{'type':'text','text':task+'\nINPUT_JSON\n'+json.dumps(payload,ensure_ascii=False,separators=(',',':'))}]
        image_log=[]
        for label,path in images:
            path=Path(path)
            raw=path.read_bytes()
            media='image/png' if path.suffix.lower()=='.png' else 'image/jpeg'
            content.extend([{'type':'text','text':label},
                            {'type':'image_url','image_url':{'url':f'data:{media};base64,'+base64.b64encode(raw).decode()}}])
            image_log.append({'label':label,'filename':path.name,'sha256':hashlib.sha256(raw).hexdigest()})
        request = dict(model=self.model, temperature=.2, max_tokens=tokens, stream=False, thinking={"type":"disabled"},
                       messages=[dict(role='system',content=system),dict(role='user',content=content)])
        log_id=len(list(self.logs.glob('*.request.json')))+1
        suffix=self.logs/f'{log_id:03}_{step}'
        # Preserve all outbound text and hashes. Keep the credential and image base64 out of logs.
        suffix.with_suffix('.request.json').write_text(json.dumps(dict(
            url=self.url,model=self.model,max_tokens=tokens,temperature=.2,stream=False,
            thinking={"type":"disabled"},system=system,user_text=task,user_content_sequence=[{'type':'text','text':part['text']} if part['type']=='text'
            else {'type':'image','label':image_log[n]['label'],'sha256':image_log[n]['sha256']}
            for part,n in zip(content,[0]+[i for i in range(len(image_log)) for _ in range(2)])],
            input=payload,images=image_log),ensure_ascii=False,indent=2),encoding='utf8')
        wire=Request(self.url,data=json.dumps(request).encode(),headers={
            'Authorization':'Bearer '+self.key,'Content-Type':'application/json'},method='POST')
        last=None
        for retry in range(3):
            try:
                with urlopen(wire,timeout=150) as res:
                    raw=json.load(res)
                break
            except HTTPError as e:
                last=f'{e.code}: {e.read(2048).decode(errors="replace").replace(self.key,"[redacted]")}'
                if e.code not in {429,500,502,503,504} or retry==2:
                    suffix.with_suffix('.error.txt').write_text(last,encoding='utf8')
                    raise RuntimeError(f'Kimi request failed ({step}): {last}') from None
                time.sleep(2**retry)
            except (URLError, TimeoutError, ConnectionError, OSError) as e:
                last=str(e).replace(self.key,'[redacted]')
                if retry==2:
                    suffix.with_suffix('.error.txt').write_text(last,encoding='utf8')
                    raise RuntimeError(f'Kimi transport failed ({step}): {last}') from None
                time.sleep(2**retry)
        choice=raw['choices'][0]
        result=choice['message'].get('content') or ''
        envelope=dict(id=raw.get('id'),model=raw.get('model'),usage=raw.get('usage'),
                      finish_reason=choice.get('finish_reason'),content=result)
        suffix.with_suffix('.response.json').write_text(json.dumps(envelope,ensure_ascii=False,indent=2),encoding='utf8')
        if choice.get('finish_reason')=='length':
            raise ValueError(f'{step} exhausted model token budget.')
        text=result.strip()
        if text.startswith('\x60\x60\x60'):
            text=text.split('\n',1)[1].rsplit('\x60\x60\x60',1)[0]
        return json.loads(text)
