"""Normalize input media, calibration, metric trajectory and optional colored PLY."""
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

FLIP=np.diag([1,-1,-1,1])
PLY_TYPES=dict(char='i1',uchar='u1',int8='i1',uint8='u1',short='i2',ushort='u2',
               int='i4',uint='u4',float='f4',float32='f4',double='f8',float64='f8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def put_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def sha(path):
    # hashlib.file_digest is 3.11+. The Pi3 environment here is 3.10.
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv):
    return subprocess.run([str(x) for x in argv],check=True,capture_output=True).stdout


def media(video):
    args=['ffprobe','-v','error','-select_streams','v:0','-show_entries',
          'stream=width,height:frame=best_effort_timestamp_time','-of','json',video]
    j=json.loads(run(args))
    pts=np.array([float(f['best_effort_timestamp_time']) for f in j['frames']])
    offset=float(pts[0])
    pts-=offset
    if len(pts)<2 or np.any(np.diff(pts)<=0):
        raise ValueError('Require at least two monotonically timestamped video frames.')
    return pts,j['streams'][0]['width'],j['streams'][0]['height'],offset


def read_cloud(path):
    with Path(path).open('rb') as f:
        if f.readline().strip()!=b'ply':
            raise ValueError('Expected PLY.')
        fmt=f.readline().decode().split()[1]
        properties=[]
        count=None
        in_vertex=False
        for line in f:
            tokens=line.decode('ascii').split()
            if not tokens: continue
            if tokens[0]=='end_header': break
            if tokens[0]=='element':
                in_vertex=tokens[1]=='vertex'
                if in_vertex: count=int(tokens[2])
                elif count is None and int(tokens[2])>0:
                    raise ValueError('PLY requires vertex data first.')
            if in_vertex and tokens[0]=='property':
                if tokens[1] not in PLY_TYPES: raise ValueError('Unsupported vertex PLY property type.')
                properties.append((tokens[-1],PLY_TYPES[tokens[1]]))
        if not count or not all(name in [p[0] for p in properties] for name in ('x','y','z')):
            raise ValueError('PLY has no XYZ vertices.')
        if fmt=='ascii':
            x=np.loadtxt(f,max_rows=count,ndmin=2)
            data={name:x[:,i] for i,(name,_) in enumerate(properties)}
        elif fmt in {'binary_little_endian','binary_big_endian'}:
            prefix='<' if fmt=='binary_little_endian' else '>'
            data=np.fromfile(f,dtype=[(name,prefix+code) for name,code in properties],count=count)
        else:
            raise ValueError(f'Unsupported PLY encoding: {fmt}')
        xyz=np.column_stack([data[k] for k in ('x','y','z')])
        keys=[p[0] for p in properties]
        rgb=np.column_stack([data[k] for k in ('red','green','blue')]) if 'red' in keys else np.full(xyz.shape,155)
        if len(rgb) and float(rgb.max())<=1:
            rgb=255*rgb
        valid=np.isfinite(xyz).all(axis=1)&np.isfinite(rgb).all(axis=1)
        return xyz[valid].astype(float),np.clip(rgb[valid],0,255).astype(np.uint8)


def write_cloud(path, xyz, color):
    data=np.zeros(len(xyz),dtype=[('xyz','<f8',(3,)),('normal','<f8',(3,)),('color','u1',(3,))])
    data['xyz'],data['color']=xyz,color
    with Path(path).open('wb') as f:
        f.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(data)}\n'+
                 ''.join(f'property double {n}\n' for n in ('x','y','z','nx','ny','nz'))+
                 ''.join(f'property uchar {n}\n' for n in ('red','green','blue'))+'end_header\n').encode())
        data.tofile(f)


def normalize_camera(path,W,H):
    j=read_json(path)
    k=np.asarray(j.get('K_pinhole_pipeline',j.get('K')),dtype=float)
    ow,oh=j.get('pipeline_wh',[j.get('width',W),j.get('height',H)])
    if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1],ow,oh)<=0:
        raise ValueError('Invalid pinhole calibration.')
    if abs(k[0,1])>1e-7 or not np.allclose(k[2],[0,0,1]):
        raise ValueError('Camera skew is unsupported.')
    if np.any(np.abs(j.get('distortion',[]))>1e-8):
        raise ValueError('Supply the rectified video and its matching pinhole K.')
    k=np.diag([W/ow,H/oh,1])@k
    return {'K_pinhole_pipeline':k.tolist(),'pipeline_wh':[W,H]}


def normalize_track(path, pts):
    j=read_json(path)
    if j.get('units','meters') not in {'m','meters','metres'}:
        raise ValueError('Camera trajectory must be metric.')
    is_pi3='key_frames' in j
    if not is_pi3 and j.get('convention') not in {'blender_c2w','opencv_c2w','opencv_w2c'}:
        raise ValueError('Declare trajectory convention: blender_c2w, opencv_c2w or opencv_w2c.')
    anchors={}
    pos={}
    if is_pi3:
        stride=int(j.get('image_stride',1))
        for item in j.get('subsample_xyz',[]):
            n=int(item['row'])
            index=n*stride
            if not 0<=index<len(pts): raise ValueError('Pi3 position maps outside the video.')
            pos[n]={'row':n,'xyz':item['xyz'],'time_s':float(pts[index]),'video_index':index}
        entries=j['key_frames']
    else:
        stride=1
        entries=j['poses']
    for order,item in enumerate(entries):
        row=int(item['pose_row']) if is_pi3 else order
        if is_pi3 and not 0<=row*stride<len(pts): raise ValueError('Pi3 pose maps outside the video.')
        t=float(pts[row*stride]) if is_pi3 else float(item['time_s'])
        if not 0<=t<=pts[-1]+.0001:
            raise ValueError('Pose timestamps must be relative to the first decoded video frame.')
        m=np.array(item['blender_matrix_world'] if is_pi3 else item['matrix'],dtype=float)
        if not is_pi3 and j.get('convention')=='opencv_w2c':
            m=np.linalg.inv(m)
        if not is_pi3 and j.get('convention') in {'opencv_c2w','opencv_w2c'}:
            m=m@FLIP
        if m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1]):
            raise ValueError('Invalid camera matrix.')
        if abs(np.linalg.det(m[:3,:3])-1)>.01 or not np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=.01):
            raise ValueError('Camera matrix must be a proper rigid transform.')
        if row in anchors and not np.allclose(anchors[row]['blender_matrix_world'],m,atol=1e-6):
            raise ValueError(f'Conflicting pose at row {row}.')
        index=int(np.argmin(abs(pts-t)))
        anchors[row]={'pose_row':row,'time_s':t,'video_index':index,'blender_matrix_world':m.tolist()}
        pos[row]={'row':row,'time_s':t,'video_index':index,'xyz':m[:3,3].tolist()}
    aa=sorted(anchors.values(),key=lambda x:x['time_s'])
    pp=sorted(pos.values(),key=lambda x:x['time_s'])
    if len(aa)<2 or np.any(np.diff([x['time_s'] for x in aa])<=0):
        raise ValueError('At least two distinct complete timestamped poses are required.')
    if np.any(np.diff([x['time_s'] for x in pp])<=0):
        raise ValueError('Position times must strictly increase.')
    up=j.get('world_up')
    if up is not None:
        up=np.asarray(up,dtype=float)
        if up.shape!=(3,) or not np.isfinite(up).all() or np.linalg.norm(up)<.1:
            raise ValueError('world_up must be a finite three-vector.')
        up=(up/np.linalg.norm(up)).tolist()
    return {'key_frames':aa,'subsample_xyz':pp,'units':'meters','source_format':'pi3' if is_pi3 else 'timestamped',
            'world_up':up,'n_poses':j.get('n_poses',len(aa)),'image_stride':stride,
            'path_length_m':j.get('path_length_m',float(np.linalg.norm(np.diff([r['xyz'] for r in pp],axis=0),axis=1).sum()))}


def ray_pose(track, t, R, origin):
    pp=track['subsample_xyz']
    times=np.array([q['time_s'] for q in pp])
    p=np.asarray([q['xyz'] for q in pp])
    loc=R@(np.array([np.interp(t,times,p[:,d]) for d in range(3)])-origin)
    aa=track['key_frames']
    at=np.array([q['time_s'] for q in aa])
    rot=Slerp(at,Rotation.from_matrix(np.asarray([q['blender_matrix_world'] for q in aa])[:,:3,:3]))
    return loc,R@rot(np.clip(t,at[0],at[-1])).as_matrix()


def level(track,xyz,camera_height):
    mats=np.array([m['blender_matrix_world'] for m in track['key_frames']])
    positions=np.array([p['xyz'] for p in track['subsample_xyz']])
    up=np.mean(mats[:,:3,1],axis=0)
    up=up/max(np.linalg.norm(up),1e-6)
    declared=track.get('world_up')
    if declared is not None:
        up=np.asarray(declared)
    center=np.median(positions,axis=0)
    support=0.
    below=None
    floor=None
    if xyz is not None and len(xyz)>40:
        p=xyz[::max(1,len(xyz)//18000)]
        y=p@up
        center_h=np.median(positions@up)
        radius=np.linalg.norm((p-center)-(y-center@up)[:,None]*up,axis=1)
        near=p[(y<center_h-.14)&(y>center_h-3.20)&(radius<5)]
        if len(near)>60 and declared is not None:
            y=near@up
            lower=float(np.quantile(y,.04))
            upper=float(min(np.quantile(y,.30)+.035,center_h-.21))
            candidates=np.arange(lower-.03,upper+.034,.022)
            options=[]
            for height in candidates:
                inliers=abs(y-height)<.055
                interference=float(np.mean(y<height-.115))
                if inliers.sum()<max(30,len(y)*.035) or interference>.085:
                    continue
                xy=near[inliers]-height*up
                n=np.std(xy,axis=0)
                spread=float(np.linalg.norm(n))
                options.append((float(inliers.mean())*min(1,spread/.80),height,inliers))
            if options:
                _,h,hit=max(options)
                h=float(np.median(y[hit]))
                floor=(up,h)
                support=float(hit.mean())
                below=float(np.mean(y<h-.115))
        if len(near)>60 and declared is None:
            rng=np.random.default_rng(17)
            best=0.
            for _ in range(490):
                a,b,c=near[rng.choice(len(near),3,replace=False)]
                n=np.cross(b-a,c-a);norm=np.linalg.norm(n)
                if norm<.17:continue
                n=n/norm
                if n@up<0:n=-n
                if n@up<.87:continue
                h=float(a@n)
                residual=near@n-h
                fraction=float(np.mean(abs(residual)<.061))
                tail=float(np.mean(residual<-.13))
                if fraction<.095 or tail>.12:
                    continue
                zspan=np.linalg.norm(np.std(near[abs(residual)<.061],axis=0))
                value=fraction*min(1.,zspan/.70)-.9*tail
                if value>best:
                    best=value
                    floor=(n,h)
                    support=fraction
                    below=tail
    if floor is None:
        origin=mats[0,:3,3]-up*(.65 if camera_height is None else camera_height)
        mode='weak_camera_height_prior_0.65m' if camera_height is None else 'first_camera_height_prior'
    else:
        up,offset=floor
        origin=center+(offset-up@center)*up
        mode='low_cloud_band_with_known_gravity' if declared is not None else 'low_cloud_plane'
    x=mats[0,:3,0].copy()
    x-=up*(x@up)
    if np.linalg.norm(x)<.05:
        x=np.eye(3)[np.argmin(np.abs(up))]
        x-=up*(x@up)
    x/=np.linalg.norm(x)
    y=np.cross(up,x)
    return np.stack([x,y,up]),origin,{'floor_method':mode,'local_floor_support_fraction':support,
        'dense_geometry_below_floor_fraction':below,'world_up_declared':declared is not None,
        'floor_coordinate_in_source_m':float(origin@up)}


def select_frames(pts,track,limit):
    n=len(pts)
    pp=track['subsample_xyz']
    t=[p['time_s'] for p in pp]
    v=np.array([p['xyz'] for p in pp])
    ds=np.r_[0,np.cumsum(np.linalg.norm(np.diff(v,axis=0),axis=1))]
    motion=np.interp(pts,t,ds)
    span=pts[-1] or 1
    mix=.56*pts/span+.44*motion/max(float(ds[-1]),.01)
    # Budget is constant; time and motion coverage scale to the input.
    return sorted({int(np.argmin(abs(mix-x))) for x in np.linspace(mix[0],mix[-1],min(limit,n))})


def decode(video, indices, out):
    out.mkdir(parents=True,exist_ok=True)
    paths={}
    ids=sorted(set(indices))
    expr='+'.join(f'eq(n\\,{n})' for n in ids)
    for f in out.glob('temporary_*.jpg'): f.unlink()
    run(['ffmpeg','-v','error','-threads','1','-i',video,'-an','-vf',
         f'select={expr},scale=w=min(800\\,iw):h=-2','-vsync','0','-q:v','3',
         '-threads','1','-start_number','0',out/'temporary_%04d.jpg'])
    for i,n in enumerate(ids):
        p=out/f'frame_{n:06}.jpg'
        (out/f'temporary_{i:04}.jpg').replace(p)
        paths[n]=p
    return paths


def prepare(args, job, seed=None):
    out=job/'Prepared'
    out.mkdir(exist_ok=True)
    pts,W,H,start=media(args.video)
    cam=normalize_camera(args.intrinsics,W,H)
    track=normalize_track(args.trajectory,pts)
    xyz=colors=None
    if args.pointcloud:
        xyz,colors=read_cloud(args.pointcloud)
        write_cloud(out/'reference.ply',xyz,colors)
    if seed:
        R=np.array(seed['source_to_room_rotation']); origin=np.array(seed['room_origin_in_source_m'])
        audit={'floor_method':'supplied_survey_registration'}
    else:
        R,origin,audit=level(track,xyz,args.camera_height)
    inputs={k:{'name':str(Path(getattr(args,k)).name),'sha256':sha(getattr(args,k))}
            for k in ('video','trajectory','intrinsics','pointcloud') if getattr(args,k)}
    movie=out/'source.mp4'
    if not movie.exists(): shutil.copyfile(args.video,movie)
    put_json(out/'camera_pinhole.json',cam)
    put_json(out/'source_trajectory.json',track)
    (out/'video_clock.csv').write_text(''.join(f'{t:.8f}\n' for t in pts))
    candidates=select_frames(pts,track,args.frames)
    if seed and 'preview_indices' in seed:
        candidates=sorted(set(candidates)|{int(v) for v in seed['preview_indices'] if 0<=v<len(pts)})
    photos=decode(args.video,candidates,job/'Evidence')
    q=(xyz-origin)@R.T if xyz is not None else None
    if q is not None:
        np.savez_compressed(out/'aligned_points.npz',xyz=q,rgb=colors)
    audit.update(dict(original_inputs=inputs,video_pts_offset=start,video_width=W,video_height=H,decoded_frames=len(pts),
                      duration_s=float(pts[-1]),selected_indices=candidates,cloud_points=0 if xyz is None else len(xyz),
                      matrix_count=len(track['key_frames']),positions=len(track['subsample_xyz'])))
    put_json(job/'InputAudit.json',audit)
    return dict(pts=pts, K=np.array(cam['K_pinhole_pipeline']), W=W,H=H, track=track,
                R=R,origin=origin,xyz=q,colors=colors,photos=photos,prepared=out,audit=audit)
