"""Independent implementation of GSP local/global terms and weighted GES triangles.

Reference: Du et al., CVPR 2022, equations (5)-(9).
This is an engineering reproduction, not a claim of identical author results.
"""
from pathlib import Path
import json
import heapq
import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr


def grid(size, cell):
    w, h = size
    xs = np.linspace(0, w - 1, max(2, int(np.ceil(w / cell)) + 1))
    ys = np.linspace(0, h - 1, max(2, int(np.ceil(h / cell)) + 1))
    vertices = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    nx, ny = len(xs), len(ys)
    faces = []
    neighbors = [set() for _ in vertices]
    edges = set()
    for y in range(ny - 1):
        for x in range(nx - 1):
            a = y * nx + x
            quad = [a, a + 1, a + nx + 1, a + nx]
            faces.extend([[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]])
            for u, v in zip(quad, quad[1:] + quad[:1]):
                edges.add(tuple(sorted((u, v))))
                neighbors[u].add(v)
                neighbors[v].add(u)
    return dict(size=size, vertices=vertices, nx=nx, ny=ny, faces=np.array(faces),
                edges=sorted(edges), neighbors=neighbors)


def interpolate(mesh, p):
    w, h = mesh['size']
    tx = np.clip(p[0] / (w - 1) * (mesh['nx'] - 1), 0, mesh['nx'] - 1)
    ty = np.clip(p[1] / (h - 1) * (mesh['ny'] - 1), 0, mesh['ny'] - 1)
    x, y = min(int(tx), mesh['nx'] - 2), min(int(ty), mesh['ny'] - 2)
    u, v = tx - x, ty - y
    a = y * mesh['nx'] + x
    return [(a, (1-u)*(1-v)), (a+1, u*(1-v)),
            (a+mesh['nx'], (1-u)*v), (a+mesh['nx']+1, u*v)]


def matches(images, dimension, diagnostics=None):
    sift = cv2.SIFT_create(nfeatures=5000)
    features = []
    for image in images:
        h, w = image.shape[:2]
        s = min(1., dimension / max(h, w))
        small = cv2.resize(image, (max(2, round(w*s)), max(2, round(h*s))))
        kp, desc = sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), None)
        xy = np.float32([k.pt for k in kp]).reshape(-1, 2)
        xy *= [w / small.shape[1], h / small.shape[0]]
        features.append((xy, desc))
    result = []
    bf = cv2.BFMatcher()
    for i, (p, d) in enumerate(features):
        for j in range(i+1, len(images)):
            q, e = features[j]
            detail=dict(cameras=[i,j],keypoints=[len(p),len(q)],candidates=0,inliers=0,accepted=False)
            if diagnostics is not None: diagnostics.append(detail)
            if d is None or e is None or min(len(d), len(e)) < 2:
                detail['reason']='insufficient keypoints'
                continue
            reverse = {(a.trainIdx, a.queryIdx) for pair in bf.knnMatch(e, d, k=2)
                       if len(pair) == 2 for a,b in [pair] if a.distance < .72*b.distance}
            good = [a for pair in bf.knnMatch(d, e, k=2) if len(pair) == 2
                    for a,b in [pair] if a.distance < .72*b.distance and (a.queryIdx,a.trainIdx) in reverse]
            detail['candidates']=len(good)
            if len(good) < 4:
                detail['reason']='fewer than 4 candidate correspondences'
                continue
            src = np.float32([p[a.queryIdx] for a in good])
            dst = np.float32([q[a.trainIdx] for a in good])
            threshold = 3 * max(images[i].shape[:2]) / min(dimension, max(images[i].shape[:2]))
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, threshold)
            detail['inliers']=0 if mask is None else int(mask.sum())
            if H is None or mask is None or mask.sum() < 15 or mask.mean() < .3:
                detail['reason']='requires at least 15 inliers and 30% inlier ratio'
                continue
            keep = mask.ravel().astype(bool)
            detail['accepted']=True
            detail['reason']='accepted'
            result.append(dict(i=i, j=j, p=src[keep], q=dst[keep], H=H))
    return result


def initialize(n, pairs, reference):
    transforms = [None]*n
    transforms[reference] = np.eye(3)
    queue = []
    def push(u):
        for k, pair in enumerate(pairs):
            if u == pair['i']:
                heapq.heappush(queue, (-len(pair['p']), k, u, pair['j'], 0))
            elif u == pair['j']:
                heapq.heappush(queue, (-len(pair['p']), k, u, pair['i'], 1))
    push(reference)
    while queue:
        _, k, u, v, direction = heapq.heappop(queue)
        if transforms[v] is not None:
            continue
        H = pairs[k]['H'] if direction else np.linalg.inv(pairs[k]['H'])
        transforms[v] = transforms[u] @ H
        transforms[v] /= transforms[v][2,2]
        push(v)
    if any(H is None for H in transforms):
        missing=[i for i,H in enumerate(transforms) if H is None]
        raise ValueError(f'Overlap graph is disconnected; cameras {missing} cannot connect to reference {reference}. '
                         'Use textured overlapping images. For adjacent crops without overlap, explicitly select --layout-grid 2.')
    return transforms


def layout(images, columns=2, cell=128, max_canvas_pixels=120_000_000):
    """Explicit row-major placement, with original pixels and no inferred geometry."""
    if len(images)<2 or not isinstance(columns,int) or not 1<=columns<=len(images) or cell<16:
        raise ValueError('Need at least two images, valid layout column count and cell>=16.')
    if any(im is None or im.dtype!=np.uint8 or im.ndim!=3 or im.shape[2]!=3 or min(im.shape[:2])<2 for im in images):
        raise ValueError('Layout requires nonempty uint8 BGR images.')
    rows=(len(images)+columns-1)//columns
    widths=[max(im.shape[1] for i,im in enumerate(images) if i%columns==c) for c in range(columns)]
    heights=[max(im.shape[0] for i,im in enumerate(images) if i//columns==r) for r in range(rows)]
    xs=np.cumsum([0]+widths[:-1]); ys=np.cumsum([0]+heights[:-1])
    meshes=[]; positions=[]
    for i,im in enumerate(images):
        mesh=grid((im.shape[1],im.shape[0]),cell)
        offset=[int(xs[i%columns]),int(ys[i//columns])]
        mesh['warped']=mesh['vertices']+offset
        meshes.append(mesh); positions.append(offset)
    canvas=[sum(widths),sum(heights)]
    if canvas[0]*canvas[1]>max_canvas_pixels: raise ValueError(f'Layout canvas {canvas} exceeds {max_canvas_pixels:,} pixels; increase --max-canvas-pixels.')
    report=dict(algorithm='manual_grid_no_registration',canvas_size=canvas,columns=columns,
                positions=positions,input_sizes=[list(m['size']) for m in meshes],
                note='Original pixels; unequal image sizes leave black padding. No inferred alignment or GES optimization.')
    return dict(meshes=meshes,canvas_size=canvas,report=report)


class Equations:
    def __init__(self, n):
        self.n = n
        self.rows, self.cols, self.values, self.rhs = [], [], [], []
    def add(self, coefficients, rhs=0., weight=1.):
        r = len(self.rhs)
        for col, value in coefficients.items():
            if value:
                self.rows.append(r); self.cols.append(col); self.values.append(value*weight)
        self.rhs.append(rhs*weight)
    def solve(self):
        A = coo_matrix((self.values, (self.rows, self.cols)), shape=(len(self.rhs), self.n)).tocsr()
        b = np.array(self.rhs)
        solution = lsqr(A, b, atol=1e-7, btol=1e-7, iter_lim=8000)
        if solution[1] not in (1,2,4,5):
            raise ValueError(f'Mesh solver did not converge (LSQR status {solution[1]}).')
        return solution[0], dict(equations=A.shape[0], unknowns=A.shape[1],
                                iterations=int(solution[2]), residual=float(solution[3]))


def initialize_similarity(meshes, pairs, reference):
    """Jointly fit orientation-preserving similarities on the full overlap graph.

    Unlike a chain of projective transforms, these maps have no projective pole.
    Coordinates are centered/scaled before solving, and the reference is removed
    from the unknowns exactly so the graph cannot collapse to the zero solution.
    """
    n=len(meshes)
    adjacency=[set() for _ in meshes]
    for pair in pairs:
        adjacency[pair['i']].add(pair['j']); adjacency[pair['j']].add(pair['i'])
    reached={reference}; pending=[reference]
    while pending:
        for v in adjacency[pending.pop()]-reached:
            reached.add(v); pending.append(v)
    missing=sorted(set(range(n))-reached)
    if missing:
        raise ValueError(f'Overlap graph is disconnected; cameras {missing} cannot connect to reference {reference}. '
                         'Use overlapping images, or --layout-grid for known adjacent crops.')
    centers=[(np.array(m['size'],dtype=float)-1)/2 for m in meshes]
    unit=float(np.median([max(m['size']) for m in meshes]))
    slots={i:4*k for k,i in enumerate(i for i in range(n) if i!=reference)}
    equations=Equations(4*(n-1))
    for pair in pairs:
        i,j=pair['i'],pair['j']
        for p,q in zip(pair['p'],pair['q']):
            p=(p-centers[i])/unit; q=(q-centers[j])/unit
            for dim in (0,1):
                row={}; rhs=0.
                for camera,point,sign in ((i,p,1.),(j,q,-1.)):
                    x,y=point
                    coefficients=[x,-y,1.,0.] if dim==0 else [y,x,0.,1.]
                    if camera==reference:
                        rhs-=sign*coefficients[0]
                    else:
                        for axis,value in enumerate(coefficients): row[slots[camera]+axis]=sign*value
                equations.add(row,rhs)
    # Weak unit-scale/zero-rotation regularization protects poorly conditioned
    # overlap strips while allowing the data to determine each camera pose.
    for start in slots.values():
        equations.add({start:1.},rhs=1.,weight=.01)
        equations.add({start+1:1.},weight=.01)
    solution,info=equations.solve()
    transforms=[]; scales=[]
    for i in range(n):
        a,b,tx,ty=(1.,0.,0.,0.) if i==reference else solution[slots[i]:slots[i]+4]
        scale=float(np.hypot(a,b))
        if not np.isfinite(scale) or not .02<scale<50:
            raise ValueError(f'Unreliable global similarity scale for camera {i}: {scale}. Check overlap correspondences.')
        linear=np.array([[a,-b],[b,a]])
        H=np.eye(3); H[:2,:2]=linear
        H[:2,2]=centers[reference]+unit*np.array([tx,ty])-linear@centers[i]
        transforms.append(H); scales.append(scale)
    info.update(method='joint_similarity',scales=scales)
    return transforms,info


def central_reference(n, pairs):
    """Choose the overlap-graph center to limit distortion of long image chains."""
    adjacency=[set() for _ in range(n)]
    for pair in pairs:
        adjacency[pair['i']].add(pair['j']); adjacency[pair['j']].add(pair['i'])
    scores=[]
    for start in range(n):
        distance={start:0}; queue=[start]
        for u in queue:
            for v in adjacency[u]-distance.keys():
                distance[v]=distance[u]+1; queue.append(v)
        scores.append((n-len(distance),max(distance.values()),sum(distance.values()),start))
    return min(scores)[-1]


def shape_rows(eq, mesh, offset, a, b, p, weight):
    """V(p) = (1-u)V(a)+u V(b)+v R90(V(b)-V(a)), paper Eq. (5)."""
    edge = b-a
    length = float(edge@edge)
    if length < 1:
        return
    u = float((p-a)@edge / length)
    delta=p-a
    v = float((edge[0]*delta[1]-edge[1]*delta[0]) / length)
    ia, ib, ip = interpolate(mesh,a), interpolate(mesh,b), interpolate(mesh,p)
    for dim in (0,1):
        row = {}
        def add(indices, axis, factor):
            for index, w in indices:
                col = offset + 2*index+axis
                row[col] = row.get(col,0.) + factor*w
        add(ip,dim,1); add(ia,dim,-(1-u)); add(ib,dim,-u)
        # R90(x,y)=(-y,x).
        sign = 1 if dim == 0 else -1
        add(ib,1-dim,sign*v); add(ia,1-dim,-sign*v)
        eq.add(row, weight=weight)


def structures(image, mode, model_dir, max_dimension=512):
    h,w = image.shape[:2]
    s = min(1., max_dimension/max(h,w))
    small = cv2.resize(image, (round(w*s),round(h*s)))
    if mode == 'hed':
        proto = Path(model_dir)/'deploy.prototxt'
        weights = Path(model_dir)/'hed_pretrained_bsds.caffemodel'
        if not proto.exists() or not weights.exists():
            raise ValueError('HED model missing. Run download_model.py, or explicitly select --edges canny (approximation).')
        class Crop:
            def __init__(self, params, blobs): pass
            def getMemoryShapes(self, inputs):
                self.y = (inputs[0][2]-inputs[1][2])//2
                self.x = (inputs[0][3]-inputs[1][3])//2
                self.h,self.w = inputs[1][2:]
                return [[inputs[0][0],inputs[0][1],inputs[1][2],inputs[1][3]]]
            def forward(self, inputs):
                return [inputs[0][:,:,self.y:self.y+self.h,self.x:self.x+self.w]]
        cv2.dnn_registerLayer('Crop', Crop)
        try:
            # OpenCV's native Caffe file loader cannot open some Windows Unicode
            # paths. Python reads them correctly, including relocated demo folders.
            net = cv2.dnn.readNetFromCaffe(np.frombuffer(proto.read_bytes(), dtype=np.uint8),
                                            np.frombuffer(weights.read_bytes(), dtype=np.uint8))
            blob = cv2.dnn.blobFromImage(small,1.,(small.shape[1],small.shape[0]),
                                       (104.00698793,116.66876762,122.67891434),swapRB=False,crop=False)
            net.setInput(blob)
            probability = net.forward()[0,0]
            edge = np.uint8(cv2.resize(probability,(small.shape[1],small.shape[0]))>.2)*255
        finally:
            cv2.dnn_unregisterLayer('Crop')
    else:
        edge = cv2.Canny(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),70,150)
    contours,_ = cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    curves = []
    for contour in sorted(contours,key=lambda c:cv2.arcLength(c,False),reverse=True)[:120]:
        if len(contour)<30 or cv2.arcLength(contour,False)<50:
            continue
        points = contour[:,0,:].astype(float)
        # Far-apart endpoint anchors avoid degenerate closed-contour endpoints.
        a = int(np.argmax(np.linalg.norm(points-points[0],axis=1)))
        b = int(np.argmax(np.linalg.norm(points-points[a],axis=1)))
        if np.linalg.norm(points[a]-points[b])<30:
            continue
        samples = points[np.linspace(0,len(points)-1,min(30,len(points))).astype(int)]
        factor = np.array([w/small.shape[1],h/small.shape[0]])
        curves.append((points[a]*factor,points[b]*factor,samples*factor))
    return curves


def calibrate(images, cell=128, dimension=800, reference=None, edges='canny', model_dir='models',
              local_weight=.75, ges_weight=1.5, max_canvas_pixels=120_000_000, matching_images=None):
    cv2.setNumThreads(1)
    if len(images)<2 or (reference is not None and not 0<=reference<len(images)) or cell<16 or dimension<128:
        raise ValueError('Need two or more images, valid reference, cell>=16, dimension>=128.')
    if edges not in ('hed','canny') or local_weight<0 or ges_weight<0:
        raise ValueError('Invalid edge mode or energy weights.')
    if any(im is None or im.ndim!=3 or im.dtype!=np.uint8 or min(im.shape[:2])<16 for im in images):
        raise ValueError('Inputs must be uint8 BGR images, at least 16x16.')
    meshes = [grid((im.shape[1],im.shape[0]),cell) for im in images]
    if matching_images is not None:
        if len(matching_images)!=len(images) or any(a is None or a.shape!=b.shape or a.dtype!=np.uint8 for a,b in zip(matching_images,images)):
            raise ValueError('Matching views must have the same count, order and dimensions as rendering views.')
    pair_diagnostics=[]
    pairs = matches(images if matching_images is None else matching_images,dimension,diagnostics=pair_diagnostics)
    for detail in pair_diagnostics:
        print(f"Pair {detail['cameras']}: {detail['candidates']} candidates, {detail['inliers']} inliers; {detail['reason']}",flush=True)
    if reference is None:
        reference=central_reference(len(images),pairs)
        print(f'Automatic reference camera: {reference} (overlap-graph center)',flush=True)
    Hs,initialization = initialize_similarity(meshes,pairs,reference)
    initial = [cv2.perspectiveTransform(m['vertices'][None].astype(float),H)[0] for m,H in zip(meshes,Hs)]
    if any(not np.isfinite(v).all() or np.max(np.abs(v))>1e6 for v in initial):
        raise ValueError('Invalid initial geometry.')
    offsets = np.cumsum([0]+[len(m['vertices'])*2 for m in meshes]).tolist()
    eq = Equations(offsets[-1])
    overlap = [np.zeros((m['ny'],m['nx']),np.uint8) for m in meshes]
    for pair in pairs:
        for i,pts in ((pair['i'],pair['p']),(pair['j'],pair['q'])):
            m=meshes[i]
            xy = np.int32(pts / [m['size'][0]-1,m['size'][1]-1] * [m['nx']-1,m['ny']-1])
            hull = cv2.convexHull(xy)
            cv2.fillConvexPoly(overlap[i],hull,1)
        # All reliable overlaps participate in the joint solve, including loops.
        for p,q in zip(pair['p'],pair['q']):
            for dim in (0,1):
                row={}
                for index,w in interpolate(meshes[pair['i']],p): row[offsets[pair['i']]+2*index+dim]=w
                for index,w in interpolate(meshes[pair['j']],q): row[offsets[pair['j']]+2*index+dim]=-w
                eq.add(row)
    structure_count=0
    for i,m in enumerate(meshes):
        verts=m['vertices']; offset=offsets[i]
        # Global similarity estimated from initial image geometry (2-D variant).
        z=verts[:,0]+1j*verts[:,1]; target=initial[i][:,0]+1j*initial[i][:,1]
        complex_scale=np.vdot(z-z.mean(),target-target.mean())/np.vdot(z-z.mean(),z-z.mean())
        # GSP edge-neighborhood least-squares local similarity, cf. author implementation.
        for a,b in m['edges']:
            neighbors=sorted((m['neighbors'][a]|m['neighbors'][b])-{a})
            D=np.zeros((2*len(neighbors),2))
            for k,v in enumerate(neighbors):
                dx,dy=verts[v]-verts[a]
                D[2*k:2*k+2]=[[dx,dy],[dy,-dx]]
            G=np.linalg.pinv(D)
            dx,dy=verts[b]-verts[a]
            L=np.array([[dx,dy],[dy,-dx]])@G
            for dim in (0,1):
                row={offset+2*b+dim:1.,offset+2*a+dim:-1.}
                prior={}
                for k,v in enumerate(neighbors):
                    for axis in (0,1):
                        c=offset+2*v+axis; ca=offset+2*a+axis
                        value=L[dim,2*k+axis]
                        row[c]=row.get(c,0)-value; row[ca]=row.get(ca,0)+value
                        value=G[dim,2*k+axis]
                        prior[c]=prior.get(c,0)+value; prior[ca]=prior.get(ca,0)-value
                eq.add(row,weight=local_weight)
                mid=(verts[a]+verts[b])/2
                gx=round(mid[0]/(m['size'][0]-1)*(m['nx']-1))
                gy=round(mid[1]/(m['size'][1]-1)*(m['ny']-1))
                strength=cell*(.015 if overlap[i][gy,gx] else .08)
                eq.add(prior, rhs=complex_scale.real if dim==0 else -complex_scale.imag, weight=strength)
        curves=structures(images[i],edges,model_dir)
        # Eq. (8): cosine weights from distance to overlap boundary and image boundary.
        inside=cv2.distanceTransform(np.pad(overlap[i],1),cv2.DIST_L2,3)[1:-1,1:-1]
        boundary=cv2.distanceTransform(np.pad(np.ones_like(overlap[i]),1),cv2.DIST_L2,3)[1:-1,1:-1]
        for a,b,samples in curves:
            structure_count+=1
            for p in samples:
                gx=int(np.clip(round(p[0]/(m['size'][0]-1)*(m['nx']-1)),0,m['nx']-1))
                gy=int(np.clip(round(p[1]/(m['size'][1]-1)*(m['ny']-1)),0,m['ny']-1))
                gamma=inside[gy,gx]/max(inside[gy,gx]+boundary[gy,gx],1e-8)
                weight=max(.01,.5*(np.cos(gamma*np.pi)+1))
                shape_rows(eq,m,offset,a,b,p,ges_weight*weight)
        if i==reference:
            for j,v in enumerate(verts):
                for dim in (0,1): eq.add({offset+2*j+dim:1},rhs=v[dim],weight=10)
    solution,report=eq.solve()
    deformed=[solution[offsets[i]:offsets[i+1]].reshape(-1,2) for i in range(len(images))]
    for m,v in zip(meshes,deformed):
        tri=v[m['faces']]
        e=tri[:,1]-tri[:,0]; f=tri[:,2]-tri[:,0]
        cross=e[:,0]*f[:,1]-e[:,1]*f[:,0]
        if np.any(cross<=1e-3): raise ValueError('Folded mesh detected. Increase cell size or reduce GES weight.')
    lower=np.floor(np.concatenate(deformed).min(axis=0)); upper=np.ceil(np.concatenate(deformed).max(axis=0))
    canvas=(upper-lower+1).astype(int)
    if np.prod(canvas)>max_canvas_pixels:
        raise ValueError(f'Canvas {canvas.tolist()} ({int(np.prod(canvas)):,} pixels) exceeds {max_canvas_pixels:,}; '
                         'increase --max-canvas-pixels after checking geometry.')
    for m,v in zip(meshes,deformed): m['warped']=v-lower
    errors=[]
    initial_errors=[]
    for pair in pairs:
        pi=cv2.perspectiveTransform(pair['p'][None].astype(float),Hs[pair['i']])[0]
        qi=cv2.perspectiveTransform(pair['q'][None].astype(float),Hs[pair['j']])[0]
        initial_errors.extend(np.linalg.norm(pi-qi,axis=1).tolist())
        for p,q in zip(pair['p'],pair['q']):
            a=sum(w*meshes[pair['i']]['warped'][idx] for idx,w in interpolate(meshes[pair['i']],p))
            b=sum(w*meshes[pair['j']]['warped'][idx] for idx,w in interpolate(meshes[pair['j']],q))
            errors.append(np.linalg.norm(a-b))
    report.update(canvas_size=canvas.tolist(),edge_detector=edges,structures=structure_count,
                  matches=len(errors),alignment_rmse=float(np.sqrt(np.mean(np.square(errors)))),
                  median_alignment_error=float(np.median(errors)),algorithm='GES-GSP engineering reproduction',
                  initial_alignment_rmse=float(np.sqrt(np.mean(np.square(initial_errors)))),
                  initialization=initialization,
                  matching_views='rendering_images' if matching_images is None else 'separate_photometric_views',
                  max_canvas_pixels=max_canvas_pixels,
                  evaluation='In-sample retained SIFT correspondences; not an independent image-quality metric',
                  local_weight=local_weight,ges_weight=ges_weight,cell=cell,reference=reference)
    return dict(meshes=meshes,canvas_size=canvas.tolist(),report=report)


def save(model,path):
    arrays={'info':np.array(json.dumps(model['report'])),'canvas':np.array(model['canvas_size']),
            'count':np.array(len(model['meshes']))}
    for i,m in enumerate(model['meshes']):
        for key in ('vertices','warped','faces','size'): arrays[f'{i}_{key}']=np.asarray(m[key])
    np.savez_compressed(path,**arrays)


def load(path, max_canvas_pixels=120_000_000):
    with np.load(path,allow_pickle=False) as data:
        meshes=[{key:data[f'{i}_{key}'].copy() for key in ('vertices','warped','faces','size')}
                for i in range(int(data['count']))]
        model=dict(meshes=meshes,canvas_size=data['canvas'].tolist(),report=json.loads(str(data['info'])))
    if len(meshes)<2 or len(model['canvas_size'])!=2 or min(model['canvas_size'])<2 or np.prod(model['canvas_size'])>max_canvas_pixels:
        raise ValueError('Invalid saved mesh canvas.')
    for m in meshes:
        if m['vertices'].shape!=m['warped'].shape or not np.isfinite(m['warped']).all():
            raise ValueError('Invalid mesh vertices.')
    return model
