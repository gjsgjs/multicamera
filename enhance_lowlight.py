"""V2: LIME-inspired illumination refinement and signal-aware natural rendering.

Independent implementation of a weighted illumination least-squares model.
References and deviations from LIME/DUAL/Zero-DCE are documented in LOWLIGHT.md.
Registration-only images are never used as final panorama textures.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from ges.cli import read,natural
from ges import calibrate,render,save
from dehaze import small_image,match_summary
import cv2
import numpy as np
from scipy.sparse import coo_matrix,eye
from scipy.sparse.linalg import spsolve


def refine_illumination(initial, regularization=.15):
    """Weighted least squares on illumination, without periodic image borders."""
    h,w=initial.shape
    indices=np.arange(h*w).reshape(h,w)
    u=np.concatenate([indices[:,:-1].ravel(),indices[:-1,:].ravel()])
    v=np.concatenate([indices[:,1:].ravel(),indices[1:,:].ravel()])
    dx=np.abs(np.diff(initial,axis=1)); dy=np.abs(np.diff(initial,axis=0))
    wx=1/(cv2.GaussianBlur(dx,(5,5),1)+.01)
    wy=1/(cv2.GaussianBlur(dy,(5,5),1)+.01)
    weights=regularization*np.concatenate([wx.ravel(),wy.ravel()])
    rows=np.concatenate([u,v,u,v]); columns=np.concatenate([u,v,v,u])
    values=np.concatenate([weights,weights,-weights,-weights])
    matrix=eye(h*w,format='csr')+coo_matrix((values,(rows,columns)),shape=(h*w,h*w)).tocsr()
    result=spsolve(matrix,initial.ravel()).reshape(h,w)
    if not np.isfinite(result).all(): raise ValueError('Illumination solver produced invalid values.')
    return np.clip(result,.01,1).astype(np.float32)


def enhance_image(image, *, strength=.45, max_gain=3., denoise=3., work_size=192):
    if (image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<16
            or not all(math.isfinite(x) for x in (strength,max_gain,denoise))
            or not 0<=strength<=1 or not 1<=max_gain<=8 or not 0<=denoise<=20 or work_size<32):
        raise ValueError('Invalid image or natural-enhancement parameters.')
    if strength==0: return image.copy()
    h,w=image.shape[:2]
    clean=cv2.fastNlMeansDenoisingColored(image,None,denoise,denoise,7,21) if denoise>0 else image
    if denoise>0:
        # OpenCV's Lab conversion can move even a constant white field by one
        # code value. Fade denoising out in highlights, preserving clipped light.
        amount=np.clip((255-image.max(axis=2).astype(np.float32))/64,0,1)
        clean=np.uint8(np.rint(image.astype(np.float32)+(clean.astype(np.float32)-image)*amount[:,:,None]))
    small=small_image(clean,work_size).astype(np.float32)/255.
    illumination=refine_illumination(small.max(axis=2))
    illumination=cv2.resize(illumination,(w,h),interpolation=cv2.INTER_LINEAR)
    # Use the smooth illumination, rather than local edge variance, as the
    # signal confidence. Edge-dependent gain creates visible glowing outlines.
    # The noise floor limits amplification of nearly black, quantized shadows.
    confidence=illumination/(illumination+.04)
    gain=np.minimum(illumination**(-strength),max_gain)
    gain=1+confidence*(gain-1)
    # Apply one scalar to all color channels, avoiding YCrCb gray/color shifts.
    base=clean.astype(np.float32)/255.
    brightest=base.max(axis=2)
    gain=np.minimum(gain,1/np.maximum(brightest,1e-6))
    result=base*gain[:,:,None]
    return np.uint8(np.clip(np.rint(result*255),0,255))


def registration_view(image,gamma):
    """Strong visibility aid for SIFT only, never a source of final output pixels."""
    clean=cv2.fastNlMeansDenoisingColored(image,None,1.5,1.5,7,21)
    ycc=cv2.cvtColor(clean,cv2.COLOR_BGR2YCrCb); old=ycc[:,:,0]
    x=np.arange(256,dtype=np.float32)
    lut=np.uint8(np.clip(np.rint(np.minimum((x/255.)**gamma*255,x*12)),0,255))
    bright=cv2.LUT(old,lut)
    bright=cv2.createCLAHE(clipLimit=4,tileGridSize=(8,8)).apply(bright)
    ycc[:,:,0]=np.minimum(bright.astype(np.float32),old.astype(np.float32)*12).astype(np.uint8)
    return cv2.cvtColor(ycc,cv2.COLOR_YCrCb2BGR)


def write_png(path,image):
    ok,data=cv2.imencode('.png',image)
    if not ok: raise ValueError(f'Cannot encode {path}')
    data.tofile(path)


def run(args):
    source=Path(args.input).resolve(); output=Path(args.output).resolve()
    if output==source or source.is_relative_to(output): raise ValueError('Output must be a separate directory.')
    paths=sorted((p for p in source.glob(args.pattern) if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}
                  and not re.match(r'^(ground[_-]?truth|gt)\.',p.name,re.I)),key=lambda p:natural(p.name))
    if not paths: raise ValueError('No supported input images.')
    if len({p.stem.casefold() for p in paths})!=len(paths): raise ValueError('Duplicate PNG output names.')
    if output.exists() and any(output.iterdir()) and not args.overwrite: raise ValueError('Output directory is not empty; choose a new one or use --overwrite.')
    if args.stitch and any(Path(args.stitch).with_suffix(s).exists() for s in ('.png','.npz','.report.json')):
        raise ValueError('Panorama or calibration output already exists; choose a new --stitch filename.')
    output.mkdir(parents=True,exist_ok=True)
    (output/'registration').mkdir(exist_ok=True); (output/'diagnostics').mkdir(exist_ok=True)
    histogram=np.zeros(256,np.int64)
    for p in paths:
        y=cv2.cvtColor(small_image(read(p),512),cv2.COLOR_BGR2YCrCb)[:,:,0]
        histogram+=np.bincount(y.ravel(),minlength=256)
    median=np.searchsorted(histogram.cumsum(),histogram.sum()*.5)
    gamma=float(np.clip(np.log(.35)/np.log(max(1,median)/255.),.3,1)) if median<255 else 1.
    report=dict(version=2,method='LIME-inspired weighted illumination + signal-aware RGB gain',
                input=str(source),output=str(output),parameters=vars(args),registration_gamma=gamma,
                complete=False,images=[],references=['https://github.com/pvnieo/Low-light-Image-Enhancement',
                                                    'https://github.com/Li-Chongyi/Zero-DCE'])
    previews=[]; enhanced_paths=[]; registration_paths=[]
    try:
        for i,path in enumerate(paths):
            original=read(path)
            natural_image=enhance_image(original,strength=args.strength,max_gain=args.max_gain,
                                        denoise=args.denoise,work_size=args.work_size)
            helper=registration_view(original,gamma)
            enhanced_path=output/(path.stem+'.png'); registration_path=output/'registration'/(path.stem+'.png')
            write_png(enhanced_path,natural_image); write_png(registration_path,helper)
            enhanced_paths.append(enhanced_path); registration_paths.append(registration_path)
            report['images'].append(dict(input=path.name,output=enhanced_path.name,size=[original.shape[1],original.shape[0]],
                                         source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            if i in {0,len(paths)//3,2*len(paths)//3,len(paths)-1}:
                row=np.hstack([small_image(original,420),small_image(natural_image,420)])
                row=cv2.copyMakeBorder(row,28,0,0,0,cv2.BORDER_CONSTANT)
                cv2.putText(row,f'{path.name} | Original / Natural v2',(10,20),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                previews.append(row)
            print(f'{i+1}/{len(paths)}: {path.name} (natural + registration-only)',flush=True)
        width=max(p.shape[1] for p in previews)
        comparison=np.vstack([cv2.copyMakeBorder(p,0,0,0,width-p.shape[1],cv2.BORDER_CONSTANT) for p in previews])
        ok,data=cv2.imencode('.jpg',comparison)
        if ok: data.tofile(output/'diagnostics/comparison.jpg')
        if args.check_matches:
            for label,items in [('original',paths),('natural',enhanced_paths),('registration',registration_paths)]:
                info=match_summary(items,args.feature_size); report['matching_'+label]=info
                print(f"{label}: components={info['components']}, accepted pairs={info['accepted_pairs']}",flush=True)
        if args.stitch:
            images=[read(p) for p in enhanced_paths]
            helpers=[read(p) for p in registration_paths]
            print('Joint mesh: matching uses registration helpers; HED and final pixels use natural images.',flush=True)
            model=calibrate(images,matching_images=helpers,dimension=args.feature_size,edges='hed',
                            model_dir=str(Path(__file__).resolve().parent/'models'))
            destination=Path(args.stitch).resolve(); destination.parent.mkdir(parents=True,exist_ok=True)
            save(model,destination.with_suffix('.npz'))
            write_png(destination,render(model,images))
            destination.with_suffix('.report.json').write_text(json.dumps(model['report'],indent=2),encoding='utf-8')
            report['panorama']=dict(path=str(destination),report=model['report'])
            print(f'Panorama saved: {destination} ({model["canvas_size"][0]} x {model["canvas_size"][1]})',flush=True)
        report['complete']=True
    finally:
        (output/'enhancement-report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Natural images: {output}',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default=str(Path(__file__).resolve().parent/'datasets/dark'))
    parser.add_argument('--output',default=str(Path(__file__).resolve().parent/'datasets/dark_natural_v2'))
    parser.add_argument('--pattern',default='*.png')
    parser.add_argument('--strength',type=float,default=.45)
    parser.add_argument('--max-gain',type=float,default=3.)
    parser.add_argument('--denoise',type=float,default=3.)
    parser.add_argument('--work-size',type=int,default=192)
    parser.add_argument('--feature-size',type=int,default=800)
    parser.add_argument('--check-matches',action='store_true')
    parser.add_argument('--stitch',metavar='OUTPUT.png',help='Use helpers for matching and natural images for rendering')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args(argv)
    cv2.setNumThreads(1)
    try:
        if args.feature_size<128 or (args.stitch and Path(args.stitch).suffix.lower()!='.png'):
            raise ValueError('Feature size must be >=128 and panorama output must use .png.')
        enhance_image(np.zeros((16,16,3),np.uint8),strength=args.strength,max_gain=args.max_gain,
                      denoise=args.denoise,work_size=args.work_size)
        run(args)
    except (ValueError,OSError,cv2.error,MemoryError) as exc:
        print(f'Error: {exc}',file=sys.stderr); return 1
    return 0


if __name__=='__main__': raise SystemExit(main())
