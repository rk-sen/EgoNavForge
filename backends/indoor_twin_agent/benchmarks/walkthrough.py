"""Render a synchronized original-video and Blender trajectory comparison."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from PIL import Image,ImageDraw,ImageFont
import numpy as np

TOOLS=Path(__file__).resolve().parent
sys.path.insert(0,str(TOOLS.parent))
from evidence import media,put_json,decode


def font(size):
    if os.name=='nt':
        try:return ImageFont.truetype('arial.ttf',size)
        except OSError: pass
    try:return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
    except OSError:return ImageFont.load_default()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--blend',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--label',default='Indoor twin')
    p.add_argument('--fps',type=int,default=6)
    p.add_argument('--width',type=int,default=352)
    p.add_argument('--threads',type=int,default=3)
    p.add_argument('--samples',type=int,default=8)
    p.add_argument('--blender',default='blender')
    a=p.parse_args()
    a.work=a.work.resolve()
    a.output=a.output.resolve()
    a.work.mkdir(parents=True,exist_ok=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    pts,W,H,_=media(a.video)
    video_duration=float(pts[-1]+np.median(np.diff(pts)))
    slots=np.arange(0,video_duration-1e-7,1/a.fps)
    rows=[]
    for i,t in enumerate(slots):
        k=int(np.argmin(abs(pts-t)))
        rows.append(dict(output_index=i,output_pts_seconds=float(t),source_index=k,
                         source_pts_seconds=float(pts[k]),interpolation_time_error_seconds=float(pts[k]-t)))
    schedule=a.work/'Schedule.json'
    put_json(schedule,rows)
    goal_indexes=sorted({x['source_index'] for x in rows})
    source_folder=a.work/'Source'
    if len(goal_indexes)>45:
        # Dense replay: one decode avoids deeply nested FFmpeg filter expressions.
        source_folder.mkdir(exist_ok=True)
        subprocess.run(['ffmpeg','-v','error','-y','-threads','1','-i',str(a.video),
                        '-an','-vf',f'scale={a.width}:-2','-fps_mode','passthrough','-q:v','3',
                        '-threads','1','-start_number','0',str(source_folder/'dense_%06d.jpg')],check=True)
        photos={i:source_folder/f'dense_{i:06d}.jpg' for i in goal_indexes}
    else:
        photos=decode(a.video,goal_indexes,source_folder)
    render_dir=a.work/'Render'
    call=[a.blender,'-b','-t',str(a.threads),str(a.blend.resolve()),'--python-exit-code','4',
          '--python',str(TOOLS/'render_blender.py'),'--','--schedule',str(schedule),
          '--output',str(render_dir),'--samples',str(a.samples),'--width',str(a.width)]
    with (a.work/'BlenderRender.log').open('wb') as log:
        subprocess.run(call,stdout=log,stderr=subprocess.STDOUT,check=True)
    w=a.width
    h=round(a.width*H/W)
    edge=10
    merged=a.work/'Merged'
    merged.mkdir(exist_ok=True)
    title_font=font(15); label_font=font(12)
    for row in rows:
        i=row['output_index']
        source=Image.open(photos[row['source_index']]).convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
        twin=Image.open(render_dir/f'image_{i:05}.png').convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
        canvas=Image.new('RGB',(2*w+edge*3,h+70),'#172630')
        d=ImageDraw.Draw(canvas)
        d.text((edge,6),a.label,fill='#eff7fb',font=title_font)
        d.text((edge,30),'SOURCE VIDEO',fill='#89d6c3',font=label_font)
        d.text((w+2*edge,30),'TWIN / SAME CAPTURED CAMERA',fill='#89d6c3',font=label_font)
        canvas.paste(source,(edge,49));canvas.paste(twin,(w+2*edge,49))
        d.text((edge,h+53),f'Original frame {row["source_index"]:03}   |   {row["source_pts_seconds"]:05.2f} s',
               fill='#dae5e8',font=label_font)
        canvas.save(merged/f'image_{i:05}.jpg',quality=93,subsampling=0)
    subprocess.run(['ffmpeg','-v','error','-y','-framerate',str(a.fps),'-i',str(merged/'image_%05d.jpg'),
        '-frames:v',str(len(rows)),'-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags',
        '+faststart','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',
        str(a.output)],check=True)
    with a.output.with_suffix('.frames.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader();writer.writerows(rows)
    print(a.output,len(rows),'paired frames',max(abs(r['interpolation_time_error_seconds']) for r in rows))


if __name__=='__main__':
    main()
