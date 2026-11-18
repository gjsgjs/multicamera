"""Batch dark-channel-prior dehazing, with shared airlight for camera mosaics.

References: He, Sun, Tang, CVPR 2009; Guided Image Filtering, ECCV 2010.
Only photometric processing is performed; geometry and output dimensions stay fixed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from ges.cli import natural, read
import cv2
import numpy as np


def small_image(image, size):
    h,w=image.shape[:2]
    scale=min(1.,size/max(h,w))
    return cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)


def dark_channel(image, patch):
    return cv2.erode(image.min(axis=2),np.ones((patch,patch),np.uint8),borderType=cv2.BORDER_REPLICATE)


def air_samples(image, patch):
    floating=image.astype(np.float32)/255.
    dark=dark_channel(floating,patch).ravel()
    count=max(1,math.ceil(len(dark)*.001))
    indices=np.argpartition(dark,-count)[-count:]
    return floating.reshape(-1,3)[indices],dark[indices]


def estimate_air(samples, scores):
    values=np.concatenate(samples); dark=np.concatenate(scores)
    # Use the hazy end of the pooled candidates, so clear foreground tiles do
    # not lower the common airlight estimate for the whole camera group.
    count=max(1,min(len(dark),512,math.ceil(len(dark)*.1)))
    indices=np.argpartition(dark,-count)[-count:]
    candidates=values[indices]
    bright=candidates.sum(axis=1)
    chosen=candidates[bright>=np.percentile(bright,75)]
    return np.clip(chosen.mean(axis=0),.05,1.).astype(np.float32)


def guided_filter(guide, source, radius, epsilon=1e-3):
    kernel=(radius*2+1,radius*2+1)
    mean=lambda value:cv2.boxFilter(value,-1,kernel,borderType=cv2.BORDER_REFLECT)
    mean_g,mean_p=mean(guide),mean(source)
    variance=np.maximum(mean(guide*guide)-mean_g*mean_g,0)
    covariance=mean(guide*source)-mean_g*mean_p
    a=covariance/(variance+epsilon)
    b=mean_p-a*mean_g
    return mean(a)*guide+mean(b)


def dehaze_image(image, airlight, *, omega=.9, tmin=.15, patch=15, radius=30,
                 work_size=1024, clahe=0.):
    if (image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3
            or not 0<=omega<=1 or not 0<tmin<=1 or patch<1 or patch%2!=1
            or radius<1 or work_size<32 or not math.isfinite(clahe) or clahe<0):
        raise ValueError('Invalid image or dehazing parameters.')
    airlight=np.asarray(airlight,dtype=np.float32)
    if airlight.shape!=(3,) or not np.isfinite(airlight).all() or np.any(airlight<=0) or np.any(airlight>1):
        raise ValueError('Airlight must contain three finite BGR values in (0, 1].')
    small=small_image(image,work_size).astype(np.float32)/255.
    transmission=1-omega*dark_channel(small/airlight,patch)
    guide=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    transmission=guided_filter(guide,transmission,radius)
    np.clip(transmission,tmin,1,out=transmission)
    h,w=image.shape[:2]
    transmission=cv2.resize(transmission,(w,h),interpolation=cv2.INTER_LINEAR)
    result=np.empty_like(image)
    # Recover row blocks rather than allocating several full-sized float RGB images.
    for top in range(0,h,256):
        block=image[top:top+256].astype(np.float32)/255.
        block-=airlight
        block/=transmission[top:top+256,:,None]
        block+=airlight
        block*=255
        np.clip(block,0,255,out=block); np.rint(block,out=block)
        result[top:top+256]=block
    if clahe>0:
        lab=cv2.cvtColor(result,cv2.COLOR_BGR2LAB)
        lab[:,:,0]=cv2.createCLAHE(clipLimit=clahe,tileGridSize=(8,8)).apply(lab[:,:,0])
        result=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
    return result,dict(transmission_min=float(transmission.min()),transmission_mean=float(transmission.mean()))


def match_summary(paths, dimension=800):
    from ges.model import matches
    images=[read(path) for path in paths]
    diagnostics=[]
    pairs=matches(images,dimension,diagnostics)
    adjacency=[set() for _ in images]
    for pair in pairs:
        adjacency[pair['i']].add(pair['j']); adjacency[pair['j']].add(pair['i'])
    components=[]; remaining=set(range(len(images)))
    while remaining:
        start=min(remaining); reached={start}; pending=[start]
        while pending:
            for v in adjacency[pending.pop()]-reached:
                reached.add(v); pending.append(v)
        components.append(sorted(reached)); remaining-=reached
    return dict(feature_size=dimension,connected=len(components)==1,components=components,
                accepted_pairs=len(pairs),inliers=sum(len(p['p']) for p in pairs),pairs=diagnostics,
                note='Matching connectivity is not proof of correct geometry or final stitching success.')


def run(args):
    source=Path(args.input).resolve()
    output=Path(args.output).resolve() if args.output else source.with_name(source.name+'_dehazed')
    if output==source or source.is_relative_to(output):
        raise ValueError('Use a separate output folder; original images must not be overwritten.')
    extensions={'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}
    paths=sorted((p for p in source.glob(args.pattern) if p.is_file() and p.suffix.lower() in extensions
                  and not re.match(r'^(ground[_-]?truth|gt)\.',p.name,re.I)),key=lambda p:natural(p.name))
    if not paths: raise ValueError(f'No images found in {source}')
    if len({p.stem.casefold() for p in paths})!=len(paths):
        raise ValueError('Input names would collide after PNG conversion.')
    destinations=[output/(p.stem+'.png') for p in paths]
    reserved=destinations+[output/'dehaze-report.json',output/'diagnostics'/'comparison.jpg']
    if any(p.exists() for p in reserved) and not args.overwrite:
        raise ValueError('Generated output already exists. Use a new --output folder or --overwrite.')
    samples=[]; scores=[]; local_air=[]
    print(f'Estimating airlight for {len(paths)} images...',flush=True)
    for path in paths:
        values,dark=air_samples(small_image(read(path),args.work_size),args.patch)
        samples.append(values); scores.append(dark)
        local_air.append(estimate_air([values],[dark]))
    shared=estimate_air(samples,scores)
    report=dict(method='Dark channel prior + grayscale guided filter',input=str(source),output=str(output),
                count=len(paths),airlight_mode=args.airlight,shared_airlight_bgr=shared.tolist(),
                parameters={k:getattr(args,k) for k in ('omega','tmin','patch','radius','work_size','clahe')},
                complete=False,images=[])
    output.mkdir(parents=True,exist_ok=True)
    preview=[]
    try:
        for i,(path,destination) in enumerate(zip(paths,destinations)):
            original=read(path)
            air=shared if args.airlight=='shared' else local_air[i]
            processed,stats=dehaze_image(original,air,omega=args.omega,tmin=args.tmin,patch=args.patch,
                                        radius=args.radius,work_size=args.work_size,clahe=args.clahe)
            ok,data=cv2.imencode('.png',processed)
            if not ok: raise ValueError(f'Cannot encode {destination}')
            data.tofile(destination)
            report['images'].append(dict(input=path.name,output=destination.name,
                input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size=[original.shape[1],original.shape[0]],
                airlight_bgr=air.tolist(),**stats))
            if i<4:
                before=small_image(original,420); after=small_image(processed,420)
                row=np.hstack([before,after])
                cv2.putText(row,path.name+'  |  before / after',(10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),1)
                preview.append(row)
            print(f'{i+1}/{len(paths)}: {path.name} -> {destination.name}',flush=True)
            del original,processed
        if preview:
            # Keep diagnostic images in a subfolder, outside the stitch input glob.
            width=max(row.shape[1] for row in preview)
            padded=[cv2.copyMakeBorder(row,0,0,0,width-row.shape[1],cv2.BORDER_CONSTANT) for row in preview]
            directory=output/'diagnostics'; directory.mkdir(exist_ok=True)
            ok,data=cv2.imencode('.jpg',np.vstack(padded))
            if ok: data.tofile(directory/'comparison.jpg')
        if args.check_matches:
            print('Checking original overlap graph...',flush=True)
            report['matching_before']=match_summary(paths,args.feature_size)
            print('Checking dehazed overlap graph...',flush=True)
            report['matching_after']=match_summary(destinations,args.feature_size)
            for key in ('matching_before','matching_after'):
                info=report[key]
                print(f"{key}: components={info['components']}, accepted pairs={info['accepted_pairs']}, inliers={info['inliers']}",flush=True)
        report['complete']=True
    finally:
        (output/'dehaze-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Saved {len(paths)} dehazed images: {output}',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default=str(Path(__file__).resolve().parent/'datasets/frog'))
    parser.add_argument('--output',help='Defaults to a sibling folder named <input>_dehazed')
    parser.add_argument('--pattern',default='*.png')
    parser.add_argument('--omega',type=float,default=.9,help='Dehazing strength, 0 to 1')
    parser.add_argument('--tmin',type=float,default=.15,help='Transmission floor, >0 to 1')
    parser.add_argument('--patch',type=int,default=15)
    parser.add_argument('--radius',type=int,default=30)
    parser.add_argument('--work-size',type=int,default=1024)
    parser.add_argument('--clahe',type=float,default=0.,help='Optional local contrast clip limit; 0 disables it')
    parser.add_argument('--airlight',choices=['shared','individual'],default='shared')
    parser.add_argument('--check-matches',action='store_true')
    parser.add_argument('--feature-size',type=int,default=800)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args(argv)
    if (not math.isfinite(args.omega) or not 0<=args.omega<=1 or not 0<args.tmin<=1
            or args.patch<1 or args.patch%2!=1 or args.radius<1 or args.work_size<32
            or not math.isfinite(args.clahe) or args.clahe<0 or args.feature_size<128):
        parser.error('Invalid strength, transmission, patch, radius or image-size parameter.')
    cv2.setNumThreads(1)
    try:
        run(args)
    except (ValueError,OSError,cv2.error,MemoryError) as exc:
        print(f'Error: {exc}',file=sys.stderr); return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
