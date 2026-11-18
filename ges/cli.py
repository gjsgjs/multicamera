import argparse
import json
from pathlib import Path
import re
import sys
import cv2
import numpy as np
from .model import calibrate,layout,load,save
from .render import render


def read(path):
    im=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if im is None: raise ValueError(f'Cannot decode {path}')
    return im


def write(path,image):
    path=Path(path)
    if path.exists(): raise ValueError(f'Output exists: {path}; select a new filename.')
    path.parent.mkdir(parents=True,exist_ok=True)
    ok,data=cv2.imencode(path.suffix,image)
    if not ok: raise ValueError(f'Cannot encode {path}')
    data.tofile(path)


def natural(path):
    return [(1,int(s)) if s.isdigit() else (0,s.casefold()) for s in re.split(r'(\d+)',path)]


def main(argv=None):
    parser=argparse.ArgumentParser(description='GES-GSP mesh-based camera mosaics; original-resolution output.')
    sub=parser.add_subparsers(dest='command',required=True)
    fit=sub.add_parser('stitch',help='Joint mesh calibration and PNG output')
    fit.add_argument('--images',nargs='+',required=True)
    fit.add_argument('--matching-images',nargs='+',help='Optional photometric helpers in the same camera order; only used for SIFT, never for structures or rendering')
    fit.add_argument('--edges',choices=['hed','canny'],default='hed')
    fit.add_argument('--model-dir',default=str(Path(__file__).resolve().parents[1]/'models'))
    fit.add_argument('--cell',type=int,default=128)
    fit.add_argument('--feature-size',type=int,default=800)
    fit.add_argument('--reference',type=int,default=None,help='Reference camera index; default selects the overlap-graph center')
    fit.add_argument('--local-weight',type=float,default=.75)
    fit.add_argument('--ges-weight',type=float,default=1.5)
    fit.add_argument('--save-model',help='Reusable mesh .npz; defaults to output path with .npz suffix')
    fit.add_argument('--layout-grid',type=int,metavar='COLUMNS',
                     help='Explicit row-major grid for adjacent crops without overlap; skips automatic alignment, keeps original pixels')
    img=sub.add_parser('images',help='Apply saved mesh to one frame group')
    img.add_argument('--images',nargs='+',required=True)
    seq=sub.add_parser('sequence',help='Apply fixed mesh to synchronized camera folders')
    seq.add_argument('--dirs',nargs='+',required=True)
    seq.add_argument('--pattern',default='*.png')
    seq.add_argument('--max-frames',type=int)
    for p in (img,seq): p.add_argument('--calibration','-c',required=True)
    for p in (fit,img,seq):
        p.add_argument('--output','-o',required=True)
        p.add_argument('--blend',choices=['sharp','feather'],default='sharp')
        p.add_argument('--tile',type=int,default=512)
        p.add_argument('--max-canvas-pixels',type=int,default=120_000_000,
                       help='Canvas pixel budget (default: 120000000); never resizes the output')
    args=parser.parse_args(argv)
    cv2.setNumThreads(1)
    try:
        if args.max_canvas_pixels<4: raise ValueError('--max-canvas-pixels must be at least 4.')
        if Path(args.output).exists(): raise ValueError('Output already exists; choose a new output path.')
        if args.command=='stitch':
            if args.matching_images is not None and args.layout_grid is not None:
                raise ValueError('--matching-images is for automatic alignment; omit it with --layout-grid.')
            images=[read(p) for p in args.images]
            path=Path(args.save_model) if args.save_model else Path(args.output).with_suffix('.npz')
            if path.exists(): raise ValueError(f'Mesh calibration already exists: {path}')
            if args.layout_grid is not None:
                print('Explicit grid placement; no feature matching, warping or resizing.',flush=True)
                model=layout(images,columns=args.layout_grid,cell=args.cell,max_canvas_pixels=args.max_canvas_pixels)
            else:
                print('Camera indices: '+', '.join(f'{i}={p}' for i,p in enumerate(args.images)),flush=True)
                print('SIFT matching, HED/edge structures and joint GES-GSP mesh optimization...',flush=True)
                helpers=[read(p) for p in args.matching_images] if args.matching_images is not None else None
                if helpers is not None:
                    print('SIFT uses registration helpers; edge structures and final panorama use rendering images.',flush=True)
                model=calibrate(images,cell=args.cell,dimension=args.feature_size,reference=args.reference,
                                edges=args.edges,model_dir=args.model_dir,local_weight=args.local_weight,ges_weight=args.ges_weight,
                                max_canvas_pixels=args.max_canvas_pixels,matching_images=helpers)
            path.parent.mkdir(parents=True,exist_ok=True); save(model,path)
            report_path=path.with_suffix('.report.json')
            report_path.write_text(json.dumps(model['report'],indent=2),encoding='utf-8')
            print(json.dumps(model['report'],indent=2),flush=True)
            print(f'Mesh saved: {path}. Rendering original-size panorama...',flush=True)
            write(args.output,render(model,images,args.blend,args.tile))
        elif args.command=='images':
            model=load(args.calibration,max_canvas_pixels=args.max_canvas_pixels)
            write(args.output,render(model,[read(p) for p in args.images],args.blend,args.tile))
        else:
            model=load(args.calibration,max_canvas_pixels=args.max_canvas_pixels)
            if len(args.dirs)!=len(model['meshes']): raise ValueError('Folder count must match calibration.')
            if args.max_frames is not None and args.max_frames<1: raise ValueError('--max-frames must be positive.')
            roots=[Path(p).resolve() for p in args.dirs]
            destination=Path(args.output).resolve()
            if any(destination==p or destination.is_relative_to(p) for p in roots):
                raise ValueError('Output must be outside the input folders.')
            tables=[{p.name:p for p in root.glob(args.pattern) if p.is_file()} for root in roots]
            if not tables[0] or any(set(t)!=set(tables[0]) for t in tables):
                raise ValueError('Each camera folder must have the same nonempty set of frame filenames.')
            names=sorted(tables[0],key=natural)
            if args.max_frames: names=names[:args.max_frames]
            destination.mkdir(parents=True)
            count=0
            metadata=dict(canvas_size=model['canvas_size'],expected_frames=len(names),written_frames=0,complete=False)
            try:
                with (destination/'frames.jsonl').open('w',encoding='utf-8') as manifest:
                    for index,name in enumerate(names):
                        paths=[t[name] for t in tables]
                        images=[read(p) for p in paths]
                        out=f'frame_{index:06d}.png'
                        panorama=render(model,images,args.blend,args.tile)
                        write(destination/out,panorama)
                        manifest.write(json.dumps(dict(output=out,inputs=[str(p) for p in paths]))+'\n'); manifest.flush()
                        count+=1
                        print(f'{count}/{len(names)}: {out}',flush=True)
                        del images,panorama
                metadata['complete']=True
            finally:
                metadata['written_frames']=count
                (destination/'sequence.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print(f'Saved: {args.output}')
        return 0
    except (ValueError,OSError,cv2.error,MemoryError) as exc:
        print(f'Error: {exc}',file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Interrupted; completed files retained.',file=sys.stderr)
        return 130
