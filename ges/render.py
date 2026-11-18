"""Piecewise-affine mesh rasterizer. Work buffers are bounded by tile size."""
import cv2
import numpy as np


def render(model, images, blend='sharp', tile=512):
    if blend not in ('sharp','feather') or not 32<=tile<=2048:
        raise ValueError('Invalid blend mode or tile size.')
    if len(images)!=len(model['meshes']): raise ValueError('Camera count differs from calibration.')
    w,h=map(int,model['canvas_size'])
    result=np.zeros((h,w,3),np.uint8)
    prepared=[]
    for image,m in zip(images,model['meshes']):
        if image.dtype!=np.uint8 or image.shape!=(int(m['size'][1]),int(m['size'][0]),3):
            raise ValueError('Input size/order must match calibration; images are not automatically resized.')
        dst=m['warped'][m['faces']]; src=m['vertices'][m['faces']]
        bounds=np.column_stack((np.floor(dst.min(axis=1)),np.ceil(dst.max(axis=1))))
        transforms=[cv2.getAffineTransform(d.astype(np.float32),s.astype(np.float32)) for d,s in zip(dst,src)]
        bary=[cv2.getAffineTransform(d.astype(np.float32),np.float32([[0,0],[1,0],[0,1]])) for d in dst]
        prepared.append((bounds,transforms,bary))
    for top in range(0,h,tile):
        for left in range(0,w,tile):
            bottom,right=min(h,top+tile),min(w,left+tile)
            shape=(bottom-top,right-left)
            accumulator=np.zeros((*shape,3),np.float32)
            total=np.zeros(shape,np.float32)
            target=result[top:bottom,left:right]
            for image,(bounds,transforms,bary) in zip(images,prepared):
                mapx=np.full(shape,-10,np.float32); mapy=np.full(shape,-10,np.float32)
                covered=np.zeros(shape,bool)
                candidates=np.flatnonzero((bounds[:,0]<right)&(bounds[:,2]>=left)&(bounds[:,1]<bottom)&(bounds[:,3]>=top))
                for k in candidates:
                    x0,y0=np.maximum([left,top],bounds[k,:2]).astype(int)
                    x1,y1=np.minimum([right,bottom],bounds[k,2:]+1).astype(int)
                    xx=np.arange(x0,x1,dtype=float)[None,:]; yy=np.arange(y0,y1,dtype=float)[:,None]
                    B=bary[k]; u=B[0,0]*xx+B[0,1]*yy+B[0,2]; v=B[1,0]*xx+B[1,1]*yy+B[1,2]
                    inside=(u>=-1e-6)&(v>=-1e-6)&(u+v<=1.000001)
                    region=(slice(y0-top,y1-top),slice(x0-left,x1-left))
                    T=transforms[k]
                    mapx[region][inside]=(T[0,0]*xx+T[0,1]*yy+T[0,2])[inside]
                    mapy[region][inside]=(T[1,0]*xx+T[1,1]*yy+T[1,2])[inside]
                    covered[region]|=inside
                ih,iw=image.shape[:2]
                valid=covered&(mapx>=-.01)&(mapx<=iw-1+.01)&(mapy>=-.01)&(mapy<=ih-1+.01)
                weights=np.minimum.reduce([mapx+1,iw-mapx,mapy+1,ih-mapy])
                weights=np.maximum(weights,0)*valid
                warped=cv2.remap(image,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                if blend=='sharp':
                    choose=weights>total
                    target[choose]=warped[choose]; np.maximum(total,weights,out=total)
                else:
                    # Narrow border feathering; large interior weights remain bounded.
                    np.minimum(weights,16,out=weights)
                    accumulator+=warped.astype(np.float32)*weights[...,None]
                    total+=weights
            if blend=='feather':
                total[total==0]=1
                accumulator/=total[...,None]
                np.rint(accumulator,out=accumulator); np.clip(accumulator,0,255,out=accumulator)
                target[:]=accumulator
    return result
