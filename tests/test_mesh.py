import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
from ges import calibrate,render,save,load
from ges.model import grid,shape_rows,Equations,layout,matches,initialize_similarity,central_reference
import cv2
import numpy as np


class MeshTests(unittest.TestCase):
    def test_joint_initialization_ignores_projective_poles_and_preserves_scale(self):
        meshes=[grid((400,300),64) for _ in range(8)]
        points=np.float32([[220,30],[260,90],[300,130],[370,210],[240,260]])
        # Retained scene correspondences describe stable translations, while a
        # pair homography has an extrapolation pole inside the source image.
        pole=np.array([[1.,0,0],[0,1,0],[-.005,0,1]])
        pairs=[dict(i=i,j=i+1,p=points,q=points-[200,0],H=pole) for i in range(7)]
        transforms,info=initialize_similarity(meshes,pairs,reference=0)
        self.assertEqual(info['method'],'joint_similarity')
        for i,H in enumerate(transforms):
            np.testing.assert_allclose(H[2],[0,0,1],atol=1e-10)
            projected=cv2.perspectiveTransform(meshes[i]['vertices'][None],H)[0]
            np.testing.assert_allclose(projected,meshes[i]['vertices']+[200*i,0],atol=.01)
        # A different reference only changes the coordinate origin for this case.
        transforms,_=initialize_similarity(meshes,pairs,reference=3)
        np.testing.assert_allclose(transforms[3],np.eye(3),atol=1e-10)

    def test_canvas_budget_is_explicit_and_does_not_resize(self):
        images=[np.zeros((40,50,3),np.uint8)]*2
        with self.assertRaisesRegex(ValueError,'max-canvas-pixels'):
            layout(images,max_canvas_pixels=3999)
        self.assertEqual(layout(images,max_canvas_pixels=4000)['canvas_size'],[100,40])

    def test_reference_uses_the_overlap_graph_center(self):
        pairs=[dict(i=i,j=i+1) for i in range(7)]
        pairs.extend([dict(i=1,j=3),dict(i=2,j=4)])
        self.assertEqual(central_reference(8,pairs),4)

    def test_ges_is_zero_for_a_similarity(self):
        mesh=grid((100,100),25)
        eq=Equations(len(mesh['vertices'])*2)
        shape_rows(eq,mesh,0,np.array([4.,7.]),np.array([80.,90.]),np.array([25.,64.]),1.)
        angle=.31; R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])*1.3
        transformed=mesh['vertices']@R.T+[42,-12]
        residual=np.zeros(len(eq.rhs))
        for r,c,v in zip(eq.rows,eq.cols,eq.values): residual[r]+=v*transformed.ravel()[c]
        np.testing.assert_allclose(residual,0,atol=1e-10)

    def test_exact_pixels_and_tile_continuity(self):
        rng=np.random.default_rng(3)
        world=rng.integers(0,256,(120,200,3),dtype=np.uint8)
        meshes=[]; images=[]
        for x in (0,60):
            mesh=grid((140,120),32); mesh['warped']=mesh['vertices']+[x,0]
            meshes.append(mesh); images.append(world[:,x:x+140])
        model=dict(meshes=meshes,canvas_size=[200,120],report={})
        a=render(model,images,tile=32); b=render(model,images,tile=128)
        np.testing.assert_array_equal(a,world)
        np.testing.assert_array_equal(a,b)
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'m.npz'; save(model,p)
            np.testing.assert_array_equal(render(load(p),images),world)
        with self.assertRaises(ValueError): render(model,[im[:60] for im in images])

    def test_joint_mesh_on_four_low_resolution_views(self):
        rng=np.random.default_rng(5)
        world=cv2.GaussianBlur(rng.integers(0,256,(360,480,3),dtype=np.uint8),(3,3),0)
        positions=[(0,0),(180,0),(0,140),(180,140)]
        images=[world[y:y+220,x:x+300].copy() for x,y in positions]
        model=calibrate(images,cell=64,dimension=512,edges='canny')
        self.assertLess(model['report']['alignment_rmse'],1)
        result=render(model,images)
        self.assertGreater(result.shape[1],450)
        self.assertGreater(result.shape[0],330)
        self.assertEqual(result.dtype,np.uint8)

    def test_disconnected_views_fail(self):
        with self.assertRaisesRegex(ValueError,'disconnected'):
            calibrate([np.zeros((100,100,3),np.uint8)]*2,edges='canny')

    def test_adjacent_grid_preserves_pixels_and_explicit_padding(self):
        images=[np.random.default_rng(i).integers(0,256,(h,w,3),dtype=np.uint8)
                for i,(h,w) in enumerate([(56,96),(58,92),(62,94),(58,88)])]
        model=layout(images,columns=2,cell=32)
        result=render(model,images,tile=32)
        self.assertEqual(result.shape,(120,188,3))
        expected=np.zeros_like(result)
        for im,(x,y) in zip(images,model['report']['positions']):
            expected[y:y+im.shape[0],x:x+im.shape[1]]=im
        np.testing.assert_array_equal(result,expected)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'grid.npz'; save(model,path)
            np.testing.assert_array_equal(render(load(path),images),expected)
        self.assertEqual(model['report']['algorithm'],'manual_grid_no_registration')

    def test_diagnostics_include_each_rejected_pair(self):
        diagnostics=[]
        pairs=matches([np.zeros((80,80,3),np.uint8)]*4,800,diagnostics)
        self.assertEqual(pairs,[])
        self.assertEqual(len(diagnostics),6)
        self.assertTrue(all(not d['accepted'] and d['reason'] for d in diagnostics))


if __name__=='__main__': unittest.main()
