"""Create a timestamped video, trajectory, pinhole K and a measured sparse RGB-D cloud."""
import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation, Slerp

K_TUM={
    1:([517.3,516.5,318.6,255.3],[.2624,-.9531,-.0054,.0026,1.1633]),
    2:([520.9,521.0,325.1,249.7],[.2312,-.7849,-.0033,-.0001,.9172]),
    3:([535.4,539.2,320.1,247.6],[0,0,0,0,0]),
}


def index(path):
    pairs=[line.split()[:2] for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    pairs.sort(key=lambda p:float(p[0]))
    return np.array([float(p[0]) for p in pairs]),[p[1] for p in pairs]


def nearest(ts,t):
    k=int(np.searchsorted(ts,t))
    return min([max(0,k-1),min(len(ts)-1,k)],key=lambda i:abs(ts[i]-t))


def distortion_map(K,dist,w,h):
    fx,fy,cx,cy=K
    k1,k2,p1,p2,k3=dist
    ys,xs=np.indices((h,w),dtype=np.float32)
    x=(xs-cx)/fx;y=(ys-cy)/fy
    rr=x*x+y*y
    s=1+rr*(k1+rr*(k2+rr*k3))
    xx=(x*s+2*p1*x*y+p2*(rr+2*x*x))*fx+cx
    yy=(y*s+p1*(rr+2*y*y)+2*p2*x*y)*fy+cy
    return [yy,xx]


def cloud_ply(path,pts):
    # Keep one measured XYZ/RGB point per 3.2 cm voxel.
    q,c=pts
    _,inds=np.unique(np.floor(q/.032).astype(np.int32),axis=0,return_index=True)
    if len(inds)>120000:
        inds=inds[np.random.default_rng(840).choice(len(inds),120000,replace=False)]
    rec=np.empty(len(inds),dtype=[('xyz','<f4',(3,)),('rgb','u1',(3,))])
    rec['xyz'],rec['rgb']=q[inds],c[inds]
    with path.open('wb') as f:
        header=f'ply\nformat binary_little_endian 1.0\nelement vertex {len(inds)}\n'
        header+=''.join(f'property float {k}\n' for k in 'xyz')
        header+='property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
        f.write(header.encode())
        rec.tofile(f)
    return len(inds)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--fr',type=int,choices=[1,2,3],required=True)
    p.add_argument('--start',type=float,required=True)
    p.add_argument('--duration',type=float,required=True)
    p.add_argument('--fps',type=int,default=10)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    rt,rf=index(a.dataset/'rgb.txt')
    dt,df=index(a.dataset/'depth.txt')
    gt=np.loadtxt(a.dataset/'groundtruth.txt',ndmin=2)
    gt=gt[np.argsort(gt[:,0])]
    gt=gt[np.r_[True,np.diff(gt[:,0])>0]]
    tracks=Slerp(gt[:,0],Rotation.from_quat(gt[:,4:8]))
    start=rt[0]+a.start
    samples=[]
    for k in range(int(a.duration*a.fps)):
        rid=nearest(rt,start+k/a.fps)
        if not gt[0,0]<=rt[rid]<=gt[-1,0]:
            raise ValueError(f'Clip includes an RGB frame outside motion capture at t={rt[rid]-rt[0]:.2f}s.')
        if abs(rt[rid]-(start+k/a.fps))>.098:
            raise ValueError('Clip includes a long gap in the RGB capture.')
        if abs(gt[nearest(gt[:,0],rt[rid]),0]-rt[rid])>.095:
            raise ValueError('Clip includes a gap in motion capture.')
        samples.append(rid)
    actual_times=rt[samples]
    loc=np.stack([np.interp(actual_times,gt[:,0],gt[:,j]) for j in (1,2,3)],axis=1)
    rots=tracks(actual_times).as_matrix()
    w,h=Image.open(a.dataset/rf[samples[0]]).size
    intr,dist=K_TUM[a.fr]
    fx,fy,cx,cy=intr
    maps=distortion_map(intr,dist,w,h) if any(dist) else None
    yy,xx=np.mgrid[:h:8,:w:8]
    records=[]
    all_pts=[]; all_rgb=[]; dtd=[]
    with TemporaryDirectory(dir=a.output) as tmp:
        tmp=Path(tmp)
        for i,rid in enumerate(samples):
            rgb=np.asarray(Image.open(a.dataset/rf[rid]).convert('RGB'))
            if maps is not None:
                rgb=np.stack([map_coordinates(rgb[...,j],maps,order=1,mode='nearest') for j in (0,1,2)],axis=-1)
            Image.fromarray(rgb).save(tmp/f'image_{i:05}.jpg',quality=94)
            M=np.eye(4);M[:3,:3]=rots[i];M[:3,3]=loc[i]
            records.append({'time_s':i/a.fps,'matrix':M.tolist()})
            if i % max(1,a.fps//3): continue
            did=nearest(dt,rt[rid])
            gap=abs(dt[did]-rt[rid])
            if gap>.045: continue
            z=np.asarray(Image.open(a.dataset/df[did]))
            if maps is not None:
                z=map_coordinates(z,maps,order=0,mode='constant',cval=0)
            z=z[yy,xx].astype(np.float32)/5000.
            valid=(z>.35)&(z<6.5)
            local=np.column_stack(((xx[valid]-cx)*z[valid]/fx,(yy[valid]-cy)*z[valid]/fy,z[valid]))
            q=local@rots[i].T+loc[i]
            all_pts.append(q)
            all_rgb.append(rgb[yy,xx][valid])
            dtd.append(float(gap))
        command=['ffmpeg','-v','error','-y','-framerate',str(a.fps),'-i',str(tmp/'image_%05d.jpg'),
                 '-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                 str(a.output/'source.mp4')]
        subprocess.run(command,check=True)
    cloud_count=cloud_ply(a.output/'cloud.ply',(np.concatenate(all_pts),np.concatenate(all_rgb)))
    def save(name,data):
        (a.output/name).write_text(json.dumps(data,indent=2)+'\n')
    save('trajectory.json',{'units':'meters','convention':'opencv_c2w','world_up':[0,0,1],'poses':records})
    save('intrinsics.json',{'K':[[fx,0,cx],[0,fy,cy],[0,0,1]],'width':w,'height':h})
    save('SourceMetadata.json',dict(source=a.dataset.name,source_project='TUM RGB-D Benchmark',
        license='CC BY 4.0',authors='J. Sturm, N. Engelhard, F. Endres, W. Burgard and D. Cremers',
        clip_start_after_first_rgb_seconds=a.start,clip_duration_seconds=len(records)/a.fps,video_fps=a.fps,
        frames=len(records),ground_truth_motion_length_m=float(np.linalg.norm(np.diff(loc,axis=0),axis=1).sum()),
        registered_depth_stride_px=8,depth_match_max_seconds=max(dtd),voxel_m=.032,cloud_count=cloud_count,
        rgb_and_registered_depth_remapped_to_pinhole=maps is not None))
    print(a.output, len(records), cloud_count)


if __name__=='__main__':
    main()
