"""Render selected physical timestamps inside Blender."""
import argparse
import json
import math
from pathlib import Path
import sys
import bpy

p=argparse.ArgumentParser()
p.add_argument('--schedule',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--width',type=int,default=352)
p.add_argument('--samples',type=int,default=8)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
a.output.mkdir(parents=True,exist_ok=True)
s=bpy.context.scene
if s.camera is None or s.camera.name!='TrajectoryCam':
    raise ValueError('Open an assembled Twin scene with TrajectoryCam active.')
s.render.resolution_percentage=100
s.render.resolution_y=int(round(a.width*s.render.resolution_y/s.render.resolution_x))
s.render.resolution_x=a.width
s.render.image_settings.file_format='PNG'
s.cycles.samples=a.samples
s.cycles.use_denoising=False
s.cycles.adaptive_threshold=.14
s.render.use_file_extension=True
s.render.film_transparent=False
for row in json.loads(a.schedule.read_text()):
    f=1+row['source_pts_seconds']*s.render.fps/s.render.fps_base
    n=math.floor(f+1e-7)
    s.frame_set(n,subframe=max(0,min(.999999,f-n)))
    target=a.output/f'image_{row["output_index"]:05}.png'
    if target.is_file():
        continue
    s.render.filepath=str(target)
    bpy.ops.render.render(write_still=True)
